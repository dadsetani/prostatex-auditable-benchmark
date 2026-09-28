"""Compare primary PROSTATEx CNN baselines and generate evaluation figures."""

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


CONFIGURATIONS = {
    "T2": "t2_primary_ensemble_predictions.csv",
    "ADC": "adc_primary_ensemble_predictions.csv",
    "T2/ADC": "paired_primary_ensemble_predictions.csv",
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


def read_archive_csv(archive, name):
    rows = list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8"))))
    return {
        row["full_key"]: {
            "patient": row["patient"],
            "label": int(row["label"]),
            "probability": float(row["probability"]),
        }
        for row in rows
    }


def auc(labels, probabilities):
    labels = np.asarray(labels, dtype=np.uint8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    require(positives > 0 and negatives > 0, "AUC requires both classes")
    _, inverse = np.unique(probabilities, return_inverse=True)
    positive_counts = np.bincount(inverse, weights=labels)
    negative_counts = np.bincount(inverse, weights=1 - labels)
    lower_negative_counts = np.cumsum(negative_counts) - negative_counts
    concordance = np.sum(
        positive_counts * (lower_negative_counts + 0.5 * negative_counts)
    )
    return float(concordance / (positives * negatives))


def average_precision(labels, probabilities):
    labels = np.asarray(labels, dtype=np.uint8)
    order = np.argsort(-np.asarray(probabilities), kind="stable")
    ordered = labels[order]
    true_positives = np.cumsum(ordered)
    precision = true_positives / np.arange(1, len(ordered) + 1)
    return float(precision[ordered == 1].sum() / ordered.sum())


def roc_curve(labels, probabilities):
    labels = np.asarray(labels, dtype=np.uint8)
    order = np.argsort(-np.asarray(probabilities), kind="stable")
    ordered = labels[order]
    true_positives = np.r_[0, np.cumsum(ordered)]
    false_positives = np.r_[0, np.cumsum(1 - ordered)]
    return false_positives / false_positives[-1], true_positives / true_positives[-1]


def precision_recall_curve(labels, probabilities):
    labels = np.asarray(labels, dtype=np.uint8)
    order = np.argsort(-np.asarray(probabilities), kind="stable")
    ordered = labels[order]
    true_positives = np.cumsum(ordered)
    false_positives = np.cumsum(1 - ordered)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / true_positives[-1]
    return np.r_[0, recall], np.r_[1, precision]


def patient_bootstrap_difference(reference, comparator, replicates, seed):
    keys = sorted(reference)
    require(set(keys) == set(comparator), "Configurations have different finding keys")
    require(all(reference[key]["patient"] == comparator[key]["patient"]
                and reference[key]["label"] == comparator[key]["label"] for key in keys),
            "Configurations have inconsistent identities")
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
        "ROC_AUC_difference": float(np.mean(auc_differences)),
        "ROC_AUC_difference_95_CI": [
            float(np.percentile(auc_differences, 2.5)),
            float(np.percentile(auc_differences, 97.5)),
        ],
        "average_precision_difference": float(np.mean(ap_differences)),
        "average_precision_difference_95_CI": [
            float(np.percentile(ap_differences, 2.5)),
            float(np.percentile(ap_differences, 97.5)),
        ],
    }


def arrays(rows):
    ordered = [rows[key] for key in sorted(rows)]
    return (
        np.asarray([row["label"] for row in ordered], dtype=np.uint8),
        np.asarray([row["probability"] for row in ordered], dtype=np.float64),
    )


def plot_curves(configurations, output_dir):
    colors = {"T2": "#4477AA", "ADC": "#EE6677", "T2/ADC": "#228833"}
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    for name, rows in configurations.items():
        labels, probabilities = arrays(rows)
        false_positive_rate, true_positive_rate = roc_curve(labels, probabilities)
        axis.plot(false_positive_rate, true_positive_rate, linewidth=2,
                  color=colors[name], label=f"{name} (AUC={auc(labels, probabilities):.3f})")
    axis.plot([0, 1], [0, 1], "--", color="0.55", linewidth=1)
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate",
             xlim=(0, 1), ylim=(0, 1), title="PROSTATEx out-of-fold ROC curves")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "prostatex_roc.png", dpi=300)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    prevalence = None
    for name, rows in configurations.items():
        labels, probabilities = arrays(rows)
        recall, precision = precision_recall_curve(labels, probabilities)
        prevalence = float(labels.mean())
        axis.plot(recall, precision, linewidth=2, color=colors[name],
                  label=f"{name} (AP={average_precision(labels, probabilities):.3f})")
    axis.axhline(prevalence, linestyle="--", color="0.55", linewidth=1,
                 label=f"Prevalence={prevalence:.3f}")
    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1),
             title="PROSTATEx out-of-fold precision--recall curves")
    axis.legend(loc="upper right")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "prostatex_precision_recall.png", dpi=300)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    edges = np.linspace(0, 1, 6)
    for name, rows in configurations.items():
        labels, probabilities = arrays(rows)
        predicted, observed = [], []
        for lower, upper in zip(edges[:-1], edges[1:]):
            mask = ((probabilities >= lower) & (probabilities < upper))
            if upper == 1:
                mask |= probabilities == 1
            if mask.any():
                predicted.append(float(probabilities[mask].mean()))
                observed.append(float(labels[mask].mean()))
        axis.plot(predicted, observed, marker="o", linewidth=2, color=colors[name], label=name)
    axis.plot([0, 1], [0, 1], "--", color="0.55", linewidth=1)
    axis.set(xlabel="Mean predicted probability", ylabel="Observed fraction",
             xlim=(0, 1), ylim=(0, 1), title="PROSTATEx calibration plot")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "prostatex_calibration.png", dpi=300)
    plt.close(figure)


def compare(summary_archive, output_dir, replicates=10000, seed=24681357):
    summary_archive = Path(summary_archive)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(summary_archive) as archive:
        require(archive.testzip() is None, "Summary archive CRC failure")
        configurations = {
            name: read_archive_csv(archive, filename)
            for name, filename in CONFIGURATIONS.items()
        }
    require(all(len(rows) == 306 for rows in configurations.values()),
            "Unexpected primary configuration size")
    comparisons = []
    for reference, comparator in (("T2/ADC", "ADC"), ("T2/ADC", "T2"), ("ADC", "T2")):
        result = patient_bootstrap_difference(
            configurations[reference], configurations[comparator], replicates, seed
        )
        labels, reference_probabilities = arrays(configurations[reference])
        _, comparator_probabilities = arrays(configurations[comparator])
        result.update({
            "reference": reference,
            "comparator": comparator,
            "observed_ROC_AUC_difference": (
                auc(labels, reference_probabilities) - auc(labels, comparator_probabilities)
            ),
            "observed_average_precision_difference": (
                average_precision(labels, reference_probabilities)
                - average_precision(labels, comparator_probabilities)
            ),
        })
        comparisons.append(result)
    plot_curves(configurations, output_dir)
    report = {
        "analysis_id": "PROSTATEx-primary-CNN-paired-comparison-v0.1",
        "checked_date": "2026-09-28",
        "source_archive": {
            "path": str(summary_archive),
            "sha256": sha256_path(summary_archive),
        },
        "bootstrap": {"unit": "patient", "replicates": replicates, "seed": seed},
        "comparisons": comparisons,
        "interpretation_rule": (
            "A difference is not treated as established when its percentile interval includes zero."
        ),
    }
    output = output_dir / "prostatex_primary_model_comparisons.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(comparisons, indent=2))
    print("Saved:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-archive", required=True)
    parser.add_argument("--output-dir", default="analysis/prostatex_cnn_comparison_v0_1")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=24681357)
    arguments = parser.parse_args()
    compare(arguments.summary_archive, arguments.output_dir,
            arguments.replicates, arguments.seed)