"""Summarize frozen PROSTATEx baseline runs with patient-level bootstrap intervals."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


SENSITIVITY_KEY = "ProstateX-0052|1|26.6078 10.1553 50.5636"
EXPECTED_CONFIGURATIONS = {
    ("t2", "primary"): 306,
    ("adc", "primary"): 306,
    ("adc", "sensitivity"): 305,
    ("paired", "primary"): 306,
    ("paired", "sensitivity"): 305,
}
SEEDS = (1729, 2718, 3141)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def roc_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = labels == 1
    positive_count = int(positives.sum())
    negative_count = len(labels) - positive_count
    require(positive_count and negative_count, "ROC-AUC requires both classes")
    ranks = rankdata(scores, method="average")
    return float((ranks[positives].sum() - positive_count * (positive_count + 1) / 2)
                 / (positive_count * negative_count))


def average_precision(labels, scores):
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    positive_count = int(sorted_labels.sum())
    require(positive_count, "Average precision requires a positive class")
    cumulative_true = np.cumsum(sorted_labels)
    precision = cumulative_true / np.arange(1, len(labels) + 1)
    return float(precision[sorted_labels == 1].sum() / positive_count)


def metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=np.uint8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= 0.5).astype(np.uint8)
    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    true_negative = int(np.sum((labels == 0) & (predictions == 0)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)
          if precision + sensitivity else 0.0)
    return {
        "ROC_AUC": roc_auc(labels, probabilities),
        "average_precision": average_precision(labels, probabilities),
        "accuracy": float(np.mean(labels == predictions)),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "F1": float(f1),
        "Brier_score": float(np.mean((probabilities - labels) ** 2)),
        "confusion_matrix": {
            "TN": true_negative,
            "FP": false_positive,
            "FN": false_negative,
            "TP": true_positive,
        },
    }


def read_predictions(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label"] = int(row["label"])
        row["probability"] = float(row["probability"])
        row["outer_fold"] = int(row["outer_fold"])
    return rows


def load_configuration(run_root, modality, variant):
    rows = []
    per_run_metrics = []
    for test_fold in range(5):
        for seed in SEEDS:
            run_name = f"{modality}_{variant}_fold{test_fold}_seed{seed}"
            directory = Path(run_root) / run_name
            result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            require(result["run_name"] == run_name, "Run identity changed")
            predictions = read_predictions(directory / "test_predictions.csv")
            require(all(row["outer_fold"] == test_fold for row in predictions), "Prediction fold changed")
            for row in predictions:
                row["seed"] = seed
                row["run_name"] = run_name
            rows.extend(predictions)
            per_run_metrics.append(result["metrics"])
    expected = EXPECTED_CONFIGURATIONS[(modality, variant)]
    require(len(rows) == expected * len(SEEDS), "Unexpected prediction count")
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(row)
    require(set(by_seed) == set(SEEDS), "Unexpected seed set")
    for seed, seed_rows in by_seed.items():
        require(len(seed_rows) == expected, f"Seed {seed} lacks complete OOF predictions")
        require(len({row["full_key"] for row in seed_rows}) == expected, "Duplicate OOF key")
    return rows, per_run_metrics


def ensemble_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["full_key"]].append(row)
    output = []
    for full_key, values in grouped.items():
        require(len(values) == len(SEEDS), f"Expected three seed scores for {full_key}")
        reference = values[0]
        require(all(value["label"] == reference["label"]
                    and value["patient"] == reference["patient"]
                    and value["outer_fold"] == reference["outer_fold"] for value in values),
                "Seed-level identity mismatch")
        output.append({
            "full_key": full_key,
            "patient": reference["patient"],
            "fid": reference["fid"],
            "outer_fold": reference["outer_fold"],
            "label": reference["label"],
            "probability": float(np.mean([value["probability"] for value in values])),
            "seed_probability_standard_deviation": float(np.std(
                [value["probability"] for value in values], ddof=1
            )),
        })
    output.sort(key=lambda row: row["full_key"])
    return output


def patient_bootstrap(rows, replicates=2000, seed=8675309):
    by_patient = defaultdict(list)
    for row in rows:
        by_patient[row["patient"]].append(row)
    patients = sorted(by_patient)
    generator = np.random.default_rng(seed)
    values = defaultdict(list)
    valid = 0
    for _ in range(replicates):
        sampled = generator.choice(patients, size=len(patients), replace=True)
        sampled_rows = [row for patient in sampled for row in by_patient[patient]]
        labels = np.asarray([row["label"] for row in sampled_rows], dtype=np.uint8)
        if len(np.unique(labels)) < 2:
            continue
        probabilities = np.asarray([row["probability"] for row in sampled_rows])
        result = metrics(labels, probabilities)
        for name, value in result.items():
            if name != "confusion_matrix":
                values[name].append(value)
        valid += 1
    require(valid >= int(replicates * 0.95), "Too many invalid bootstrap replicates")
    return {
        name: {
            "lower_95": float(np.percentile(metric_values, 2.5)),
            "upper_95": float(np.percentile(metric_values, 97.5)),
        }
        for name, metric_values in values.items()
    }, valid


def summarize(run_root, output_dir):
    output_dir = Path(output_dir)
    require(not output_dir.exists(), f"Output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    summaries = []
    ensemble_by_configuration = {}
    for modality, variant in EXPECTED_CONFIGURATIONS:
        rows, per_run = load_configuration(run_root, modality, variant)
        ensemble = ensemble_rows(rows)
        expected = EXPECTED_CONFIGURATIONS[(modality, variant)]
        require(len(ensemble) == expected, "Unexpected ensemble size")
        if variant == "sensitivity":
            require(SENSITIVITY_KEY not in {row["full_key"] for row in ensemble},
                    "Sensitivity finding was not removed")
        labels = np.asarray([row["label"] for row in ensemble])
        probabilities = np.asarray([row["probability"] for row in ensemble])
        pooled_metrics = metrics(labels, probabilities)
        intervals, valid_bootstraps = patient_bootstrap(ensemble)
        metric_names = [name for name in pooled_metrics if name != "confusion_matrix"]
        run_summary = {
            name: {
                "mean": float(np.mean([run[name] for run in per_run])),
                "standard_deviation": float(np.std([run[name] for run in per_run], ddof=1)),
            }
            for name in metric_names
        }
        summary = {
            "modality": modality,
            "variant": variant,
            "findings": len(ensemble),
            "patients": len({row["patient"] for row in ensemble}),
            "per_run_summary": run_summary,
            "three_seed_ensemble_OOF_metrics": pooled_metrics,
            "patient_bootstrap_95_CI": intervals,
            "valid_bootstrap_replicates": valid_bootstraps,
        }
        summaries.append(summary)
        ensemble_by_configuration[(modality, variant)] = ensemble
        path = output_dir / f"{modality}_{variant}_ensemble_predictions.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ensemble[0]))
            writer.writeheader()
            writer.writerows(ensemble)

    sensitivity_comparisons = []
    for modality in ("adc", "paired"):
        primary = {row["full_key"]: row for row in ensemble_by_configuration[(modality, "primary")]}
        sensitivity = ensemble_by_configuration[(modality, "sensitivity")]
        common_primary = [primary[row["full_key"]] for row in sensitivity]
        primary_metrics = metrics(
            [row["label"] for row in common_primary],
            [row["probability"] for row in common_primary],
        )
        sensitivity_metrics = metrics(
            [row["label"] for row in sensitivity],
            [row["probability"] for row in sensitivity],
        )
        sensitivity_comparisons.append({
            "modality": modality,
            "common_findings": len(sensitivity),
            "primary_model_on_common_findings": primary_metrics,
            "retrained_sensitivity_model": sensitivity_metrics,
            "metric_difference_sensitivity_minus_primary": {
                name: float(sensitivity_metrics[name] - primary_metrics[name])
                for name in primary_metrics if name != "confusion_matrix"
            },
        })

    report = {
        "protocol_id": "PROSTATEx-modeling-v0.1",
        "status": "COMPLETE",
        "configurations": summaries,
        "sensitivity_comparisons": sensitivity_comparisons,
        "bootstrap": {
            "unit": "patient",
            "replicates": 2000,
            "seed": 8675309,
        },
    }
    (output_dir / "baseline_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("Saved:", output_dir / "baseline_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", default="prostatex_baseline_summary_v0_1")
    arguments = parser.parse_args()
    summarize(arguments.run_root, arguments.output_dir)