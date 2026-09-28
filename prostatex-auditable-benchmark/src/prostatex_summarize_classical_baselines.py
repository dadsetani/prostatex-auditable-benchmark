"""Summarize frozen classical PROSTATEx baselines with patient bootstrap intervals."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


MODELS = ("logistic", "rbf_svm", "random_forest")
CONFIGURATIONS = {
    (model, modality, variant): 306 if variant == "primary" else 305
    for model in MODELS
    for modality, variants in {
        "t2": ("primary",),
        "adc": ("primary", "sensitivity"),
        "paired": ("primary", "sensitivity"),
    }.items()
    for variant in variants
}


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
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    ordered = labels[order]
    cumulative_true = np.cumsum(ordered)
    precision = cumulative_true / np.arange(1, len(ordered) + 1)
    return float(precision[ordered == 1].sum() / ordered.sum())


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
    precision = (true_positive / (true_positive + false_positive)
                 if true_positive + false_positive else 0.0)
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
            "TN": true_negative, "FP": false_positive,
            "FN": false_negative, "TP": true_positive,
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


def load_configuration(run_root, model, modality, variant):
    rows = []
    selected_parameters = []
    for test_fold in range(5):
        run_name = f"{model}_{modality}_{variant}_fold{test_fold}"
        directory = Path(run_root) / run_name
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        require(result["run_name"] == run_name, "Run identity changed")
        predictions = read_predictions(directory / "test_predictions.csv")
        require(all(row["outer_fold"] == test_fold for row in predictions),
                "Prediction fold changed")
        rows.extend(predictions)
        selected_parameters.append({
            "test_fold": test_fold,
            "parameters": result["selected_parameters"],
            "best_validation_ROC_AUC": result["metrics"]["best_validation_ROC_AUC"],
        })
    expected = CONFIGURATIONS[(model, modality, variant)]
    require(len(rows) == expected, "Unexpected out-of-fold prediction count")
    require(len({row["full_key"] for row in rows}) == expected, "Duplicate out-of-fold key")
    rows.sort(key=lambda row: row["full_key"])
    return rows, selected_parameters


def patient_bootstrap(rows, replicates=2000, seed=97531):
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
    for model, modality, variant in CONFIGURATIONS:
        rows, selected_parameters = load_configuration(run_root, model, modality, variant)
        labels = np.asarray([row["label"] for row in rows])
        probabilities = np.asarray([row["probability"] for row in rows])
        pooled_metrics = metrics(labels, probabilities)
        intervals, valid_bootstraps = patient_bootstrap(rows)
        summary = {
            "model": model,
            "modality": modality,
            "variant": variant,
            "findings": len(rows),
            "patients": len({row["patient"] for row in rows}),
            "selected_parameters_by_fold": selected_parameters,
            "out_of_fold_metrics": pooled_metrics,
            "patient_bootstrap_95_CI": intervals,
            "valid_bootstrap_replicates": valid_bootstraps,
        }
        summaries.append(summary)
        path = output_dir / f"{model}_{modality}_{variant}_predictions.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = {
        "protocol_id": "PROSTATEx-classical-baselines-v0.1",
        "status": "COMPLETE",
        "configurations": summaries,
        "bootstrap": {"unit": "patient", "replicates": 2000, "seed": 97531},
    }
    output = output_dir / "classical_summary.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Saved:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", default="prostatex_classical_summary_v0_1")
    arguments = parser.parse_args()
    summarize(arguments.run_root, arguments.output_dir)