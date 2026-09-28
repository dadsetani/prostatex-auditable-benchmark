"""Audit PROSTATEx classical summaries and compare them with fixed CNN baselines."""

import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np


MODELS = ("logistic", "rbf_svm", "random_forest")
MODALITIES = ("t2", "adc", "paired")
CNN_FILES = {
    "t2": "t2_primary_ensemble_predictions.csv",
    "adc": "adc_primary_ensemble_predictions.csv",
    "paired": "paired_primary_ensemble_predictions.csv",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv_bytes(raw):
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    parsed = {}
    for row in rows:
        key = row["full_key"]
        require(key not in parsed, f"Duplicate finding key: {key}")
        probability = float(row["probability"])
        parsed[key] = {
            "patient": row["patient"],
            "label": int(row["label"]),
            "probability": probability,
            "prediction": int(row.get("prediction", int(probability >= 0.5))),
        }
    return parsed


def auc(labels, probabilities):
    labels = np.asarray(labels, dtype=np.uint8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    require(positives > 0 and negatives > 0, "AUC requires both classes")
    _, inverse = np.unique(probabilities, return_inverse=True)
    positive_counts = np.bincount(inverse, weights=labels)
    negative_counts = np.bincount(inverse, weights=1 - labels)
    lower_negatives = np.cumsum(negative_counts) - negative_counts
    concordance = np.sum(positive_counts * (lower_negatives + 0.5 * negative_counts))
    return float(concordance / (positives * negatives))


def average_precision(labels, probabilities):
    labels = np.asarray(labels, dtype=np.uint8)
    order = np.argsort(-np.asarray(probabilities, dtype=np.float64), kind="stable")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered == 1].sum() / ordered.sum())


def metrics(rows):
    ordered = [rows[key] for key in sorted(rows)]
    labels = np.asarray([row["label"] for row in ordered], dtype=np.uint8)
    probabilities = np.asarray([row["probability"] for row in ordered], dtype=np.float64)
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
        "ROC_AUC": auc(labels, probabilities),
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


def require_close(observed, expected, context):
    if isinstance(expected, dict):
        require(observed == expected, f"Metric mismatch for {context}")
    else:
        require(np.isclose(observed, expected, rtol=0, atol=1e-12),
                f"Metric mismatch for {context}: {observed} != {expected}")


def validate_classical_archive(path):
    configurations = {}
    with ZipFile(path) as archive:
        require(archive.testzip() is None, "Classical summary ZIP CRC failure")
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate classical ZIP member")
        report = json.loads(archive.read("classical_summary.json"))
        require(report["protocol_id"] == "PROSTATEx-classical-baselines-v0.1",
                "Unexpected classical protocol")
        require(report["status"] == "COMPLETE", "Classical summary is incomplete")
        require(len(report["configurations"]) == 15, "Expected 15 classical configurations")
        for summary in report["configurations"]:
            model = summary["model"]
            modality = summary["modality"]
            variant = summary["variant"]
            filename = f"{model}_{modality}_{variant}_predictions.csv"
            rows = read_csv_bytes(archive.read(filename))
            expected_count = 306 if variant == "primary" else 305
            require(len(rows) == expected_count, f"Unexpected count for {filename}")
            require(len({row["patient"] for row in rows.values()}) == summary["patients"],
                    f"Patient count mismatch for {filename}")
            require(all(0 <= row["probability"] <= 1 for row in rows.values()),
                    f"Probability outside [0,1] for {filename}")
            require(all(row["prediction"] == int(row["probability"] >= 0.5)
                        for row in rows.values()), f"Threshold mismatch for {filename}")
            recomputed = metrics(rows)
            for name, expected in summary["out_of_fold_metrics"].items():
                require_close(recomputed[name], expected, f"{filename}:{name}")
            configurations[(model, modality, variant)] = {
                "rows": rows,
                "summary": summary,
            }
    primary_keys = [set(configurations[(model, modality, "primary")]["rows"])
                    for model in MODELS for modality in MODALITIES]
    require(all(keys == primary_keys[0] for keys in primary_keys[1:]),
            "Primary classical configurations have different finding keys")
    sensitivity_keys = [set(configurations[(model, modality, "sensitivity")]["rows"])
                        for model in MODELS for modality in ("adc", "paired")]
    require(all(keys == sensitivity_keys[0] for keys in sensitivity_keys[1:]),
            "Sensitivity classical configurations have different finding keys")
    require(len(primary_keys[0] - sensitivity_keys[0]) == 1,
            "Sensitivity analysis must remove exactly one finding")
    return report, configurations


def load_cnn_archive(path):
    with ZipFile(path) as archive:
        require(archive.testzip() is None, "CNN summary ZIP CRC failure")
        return {
            modality: read_csv_bytes(archive.read(filename))
            for modality, filename in CNN_FILES.items()
        }


def paired_bootstrap(reference, comparator, replicates, seed):
    keys = sorted(reference)
    require(set(keys) == set(comparator), "Paired configurations have different keys")
    require(all(reference[key]["patient"] == comparator[key]["patient"]
                and reference[key]["label"] == comparator[key]["label"] for key in keys),
            "Paired configurations have inconsistent identities")
    by_patient = defaultdict(list)
    for key in keys:
        by_patient[reference[key]["patient"]].append(key)
    patients = sorted(by_patient)
    generator = np.random.default_rng(seed)
    auc_differences = []
    ap_differences = []
    for _ in range(replicates):
        sampled = generator.choice(patients, size=len(patients), replace=True)
        sampled_keys = [key for patient in sampled for key in by_patient[patient]]
        labels = np.asarray([reference[key]["label"] for key in sampled_keys])
        if len(np.unique(labels)) < 2:
            continue
        reference_probabilities = [reference[key]["probability"] for key in sampled_keys]
        comparator_probabilities = [comparator[key]["probability"] for key in sampled_keys]
        auc_differences.append(
            auc(labels, reference_probabilities) - auc(labels, comparator_probabilities)
        )
        ap_differences.append(
            average_precision(labels, reference_probabilities)
            - average_precision(labels, comparator_probabilities)
        )
    require(len(auc_differences) >= int(replicates * 0.95), "Too many invalid bootstraps")
    return {
        "valid_bootstrap_replicates": len(auc_differences),
        "ROC_AUC_difference_95_CI": [
            float(np.percentile(auc_differences, 2.5)),
            float(np.percentile(auc_differences, 97.5)),
        ],
        "average_precision_difference_95_CI": [
            float(np.percentile(ap_differences, 2.5)),
            float(np.percentile(ap_differences, 97.5)),
        ],
    }


def forest_plot(primary, output):
    labels = []
    values = []
    lower = []
    upper = []
    names = {"logistic": "Logistic", "rbf_svm": "RBF-SVM", "random_forest": "Random forest"}
    modalities = {"t2": "T2", "adc": "ADC", "paired": "T2/ADC"}
    for model in MODELS:
        for modality in MODALITIES:
            row = primary[(model, modality)]
            ci = row["patient_bootstrap_95_CI"]["ROC_AUC"]
            labels.append(f"{names[model]}: {modalities[modality]}")
            values.append(row["out_of_fold_metrics"]["ROC_AUC"])
            lower.append(ci["lower_95"])
            upper.append(ci["upper_95"])
    positions = np.arange(len(labels))[::-1]
    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    values_array = np.asarray(values)
    axis.errorbar(values_array, positions,
                  xerr=[values_array - np.asarray(lower), np.asarray(upper) - values_array],
                  fmt="o", color="#336699", ecolor="#7799BB", capsize=3)
    axis.axvline(0.5, linestyle="--", color="0.55", linewidth=1)
    axis.set_yticks(positions, labels)
    axis.set(xlabel="Out-of-fold ROC AUC (patient-bootstrap 95% CI)", xlim=(0.45, 0.82),
             title="PROSTATEx classical baseline discrimination")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=300)
    plt.close(figure)


def compare(classical_archive, cnn_archive, output_dir, replicates=10000, seed=86420):
    classical_archive = Path(classical_archive)
    cnn_archive = Path(cnn_archive)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report, configurations = validate_classical_archive(classical_archive)
    cnn = load_cnn_archive(cnn_archive)
    primary = {
        (model, modality): configurations[(model, modality, "primary")]["summary"]
        for model in MODELS for modality in MODALITIES
    }
    comparisons = []
    for model in MODELS:
        for modality in MODALITIES:
            classical_rows = configurations[(model, modality, "primary")]["rows"]
            cnn_rows = cnn[modality]
            classical_metrics = metrics(classical_rows)
            cnn_metrics = metrics(cnn_rows)
            result = paired_bootstrap(classical_rows, cnn_rows, replicates, seed)
            result.update({
                "model": model,
                "modality": modality,
                "reference": "classical",
                "comparator": "fixed_CNN",
                "observed_ROC_AUC_difference": (
                    classical_metrics["ROC_AUC"] - cnn_metrics["ROC_AUC"]
                ),
                "observed_average_precision_difference": (
                    classical_metrics["average_precision"] - cnn_metrics["average_precision"]
                ),
            })
            comparisons.append(result)
    best = max(primary.items(), key=lambda item: item[1]["out_of_fold_metrics"]["ROC_AUC"])
    forest_plot(primary, output_dir / "prostatex_classical_auc_forest.png")
    audit = {
        "audit_id": "PROSTATEx-classical-baseline-results-v0.1",
        "checked_date": "2026-09-28",
        "status": "COMPLETE",
        "source_archive": {
            "path": str(classical_archive),
            "bytes": classical_archive.stat().st_size,
            "sha256": sha256_path(classical_archive),
            "zip_crc_check_passed": True,
        },
        "cnn_source_archive": {
            "path": str(cnn_archive),
            "sha256": sha256_path(cnn_archive),
        },
        "protocol_id": report["protocol_id"],
        "configurations": 15,
        "independent_checks": {
            "all_prediction_tables_present": True,
            "all_reported_metrics_recomputed": True,
            "primary_keys_consistent": True,
            "sensitivity_keys_consistent": True,
            "sensitivity_finding_removed": True,
        },
        "primary_results": [primary[(model, modality)] for model in MODELS for modality in MODALITIES],
        "best_observed_primary_configuration": {
            "model": best[0][0],
            "modality": best[0][1],
            "ROC_AUC": best[1]["out_of_fold_metrics"]["ROC_AUC"],
            "ROC_AUC_95_CI": [
                best[1]["patient_bootstrap_95_CI"]["ROC_AUC"]["lower_95"],
                best[1]["patient_bootstrap_95_CI"]["ROC_AUC"]["upper_95"],
            ],
        },
        "paired_classical_minus_CNN_comparisons": {
            "bootstrap_unit": "patient",
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
            "comparisons": comparisons,
            "interpretation_rule": (
                "A difference is not treated as established when its percentile interval includes zero."
            ),
        },
    }
    output = output_dir / "prostatex_classical_baseline_results_audit_v0_1.json"
    output.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "best": audit["best_observed_primary_configuration"],
        "comparisons": comparisons,
    }, indent=2))
    print("Saved:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classical-archive", required=True)
    parser.add_argument("--cnn-archive", required=True)
    parser.add_argument("--output-dir", default="analysis/prostatex_classical_comparison_v0_1")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=86420)
    arguments = parser.parse_args()
    compare(arguments.classical_archive, arguments.cnn_archive, arguments.output_dir,
            arguments.replicates, arguments.seed)