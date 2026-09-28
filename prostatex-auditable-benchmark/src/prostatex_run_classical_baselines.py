"""Run frozen classical PROSTATEx baselines on engineered patch features."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


PROTOCOL_ID = "PROSTATEx-classical-baselines-v0.1"
SENSITIVITY_KEY = "ProstateX-0052|1|26.6078 10.1553 50.5636"
EXPECTED_COUNTS = {0: 64, 1: 57, 2: 62, 3: 62, 4: 61}
MODALITY_CHANNELS = {"t2": [0], "adc": [1], "paired": [0, 1]}
MODELS = ("logistic", "rbf_svm", "random_forest")
SEED = 1729


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def load_fold(patch_root, fold):
    patch_root = Path(patch_root)
    directory = patch_root / f"prostatex_patches_fold_{fold}_v0_3"
    bundle = directory.with_suffix(".zip")
    if (directory / "patches.npz").is_file() and (directory / "metadata.json").is_file():
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        with np.load(directory / "patches.npz", allow_pickle=False) as loaded:
            arrays = {name: loaded[name].copy() for name in loaded.files}
    elif bundle.is_file():
        with ZipFile(bundle) as archive:
            require(archive.testzip() is None, f"Fold {fold} ZIP CRC failure")
            metadata = json.loads(archive.read("metadata.json"))
            with np.load(io.BytesIO(archive.read("patches.npz")), allow_pickle=False) as loaded:
                arrays = {name: loaded[name].copy() for name in loaded.files}
    else:
        raise FileNotFoundError(
            f"Missing fold {fold} patches. Expected {directory}/patches.npz or {bundle}."
        )
    required = {
        "patches", "labels", "patient_ids", "finding_ids",
        "positions_mm", "full_keys", "outer_folds",
    }
    require(required.issubset(arrays), f"Fold {fold} NPZ fields changed")
    count = EXPECTED_COUNTS[fold]
    require(arrays["patches"].shape == (count, 2, 100, 100), f"Fold {fold} shape changed")
    require(arrays["patches"].dtype == np.float32, f"Fold {fold} dtype changed")
    require(np.isfinite(arrays["patches"]).all(), f"Fold {fold} contains nonfinite pixels")
    require(np.array_equal(arrays["outer_folds"], np.full(count, fold, dtype=np.uint8)),
            f"Fold {fold} assignment changed")
    require(sha256_bytes(arrays["patches"].tobytes(order="C"))
            == metadata["patch_array_sha256"], f"Fold {fold} patch hash changed")
    require(len(set(arrays["full_keys"].tolist())) == count, f"Fold {fold} duplicate key")
    return arrays


def load_all_folds(patch_root):
    folds = {fold: load_fold(patch_root, fold) for fold in range(5)}
    keys = [key for arrays in folds.values() for key in arrays["full_keys"].tolist()]
    require(len(keys) == len(set(keys)) == 306, "All-fold keys changed")
    labels = np.concatenate([folds[index]["labels"] for index in range(5)])
    require((int(labels.sum()), int((labels == 0).sum())) == (65, 241),
            "All-fold labels changed")
    return folds


def valid_variants(modality):
    return ("primary",) if modality == "t2" else ("primary", "sensitivity")


def subset_fold(arrays, modality, variant):
    keep = np.ones(len(arrays["labels"]), dtype=bool)
    if variant == "sensitivity":
        require(modality in {"adc", "paired"}, "Sensitivity applies only to ADC inputs")
        keep &= arrays["full_keys"] != SENSITIVITY_KEY
    channels = MODALITY_CHANNELS[modality]
    return {
        "patches": arrays["patches"][keep][:, channels].astype(np.float32, copy=True),
        "labels": arrays["labels"][keep].astype(np.uint8, copy=True),
        "patient_ids": arrays["patient_ids"][keep].astype(str),
        "finding_ids": arrays["finding_ids"][keep].astype(str),
        "positions_mm": arrays["positions_mm"][keep].astype(np.float64, copy=True),
        "full_keys": arrays["full_keys"][keep].astype(str),
        "outer_folds": arrays["outer_folds"][keep].astype(np.uint8, copy=True),
    }


def concatenate(parts):
    return {name: np.concatenate([part[name] for part in parts], axis=0) for name in parts[0]}


def engineer_features(patches):
    require(patches.ndim == 4 and patches.shape[2:] == (100, 100), "Unexpected patches")
    sample_count, channels = patches.shape[:2]
    blocks = patches.reshape(sample_count, channels, 10, 10, 10, 10)
    block_means = blocks.mean(axis=(3, 5), dtype=np.float64)
    block_standard_deviations = blocks.std(axis=(3, 5), dtype=np.float64)
    flat = patches.reshape(sample_count, channels, -1).astype(np.float64)
    quantiles = np.quantile(flat, (0.1, 0.25, 0.5, 0.75, 0.9), axis=2)
    global_features = np.concatenate([
        flat.mean(axis=2)[:, :, None],
        flat.std(axis=2)[:, :, None],
        flat.min(axis=2)[:, :, None],
        flat.max(axis=2)[:, :, None],
        np.moveaxis(quantiles, 0, 2),
        (flat == 0).mean(axis=2)[:, :, None],
    ], axis=2)
    features = np.concatenate([
        block_means.reshape(sample_count, -1),
        block_standard_deviations.reshape(sample_count, -1),
        global_features.reshape(sample_count, -1),
    ], axis=1).astype(np.float32)
    require(features.shape[1] == channels * 210, "Unexpected engineered feature count")
    require(np.isfinite(features).all(), "Feature extraction produced nonfinite values")
    return features


def make_cycle(folds, test_fold, modality, variant):
    validation_fold = (test_fold + 1) % 5
    training_folds = [fold for fold in range(5) if fold not in {test_fold, validation_fold}]
    train = concatenate([subset_fold(folds[fold], modality, variant) for fold in training_folds])
    validation = subset_fold(folds[validation_fold], modality, variant)
    test = subset_fold(folds[test_fold], modality, variant)
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        require(len(split["labels"]) and len(np.unique(split["labels"])) == 2,
                f"{name} split lacks both classes")
    require(set(train["patient_ids"]).isdisjoint(validation["patient_ids"]),
            "Train/validation leakage")
    require(set(train["patient_ids"]).isdisjoint(test["patient_ids"]), "Train/test leakage")
    require(set(validation["patient_ids"]).isdisjoint(test["patient_ids"]),
            "Validation/test leakage")
    for split in (train, validation, test):
        split["features"] = engineer_features(split.pop("patches"))
    return train, validation, test, {
        "training_folds": training_folds,
        "validation_fold": validation_fold,
        "test_fold": test_fold,
        "features_per_channel": 210,
        "feature_definition": (
            "10x10 block means, 10x10 block standard deviations, and ten global "
            "statistics per channel (mean, standard deviation, min, max, five quantiles, "
            "and exact-zero fraction)"
        ),
    }


def binary_metrics(labels, probabilities):
    from sklearn.metrics import (
        accuracy_score, average_precision_score, balanced_accuracy_score,
        brier_score_loss, confusion_matrix, f1_score, precision_score, roc_auc_score,
    )
    labels = np.asarray(labels, dtype=np.uint8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= 0.5).astype(np.uint8)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    return {
        "ROC_AUC": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(true_positive / (true_positive + false_negative)),
        "specificity": float(true_negative / (true_negative + false_positive)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "F1": float(f1_score(labels, predictions, zero_division=0)),
        "Brier_score": float(brier_score_loss(labels, probabilities)),
        "confusion_matrix": {
            "TN": int(true_negative), "FP": int(false_positive),
            "FN": int(false_negative), "TP": int(true_positive),
        },
    }


def candidate_grid(model_name):
    if model_name == "logistic":
        return [{"C": value} for value in (0.001, 0.01, 0.1, 1.0, 10.0)]
    if model_name == "rbf_svm":
        return [
            {"C": cost, "gamma": gamma}
            for cost in (0.1, 1.0, 10.0)
            for gamma in ("scale", 0.01)
        ]
    if model_name == "random_forest":
        return [
            {"max_depth": depth, "min_samples_leaf": leaf}
            for depth in (None, 6, 12)
            for leaf in (1, 4)
        ]
    raise ValueError(f"Unknown model: {model_name}")


def build_model(model_name, parameters):
    if model_name == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=parameters["C"], class_weight="balanced", solver="liblinear",
                max_iter=5000, random_state=SEED,
            )),
        ])
    if model_name == "rbf_svm":
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(
                C=parameters["C"], gamma=parameters["gamma"], kernel="rbf",
                class_weight="balanced", probability=True, random_state=SEED,
            )),
        ])
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=500, max_depth=parameters["max_depth"],
            min_samples_leaf=parameters["min_samples_leaf"], max_features="sqrt",
            class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {model_name}")


def fit_one(train, validation, test, model_name, modality, variant, test_fold,
            output_root, split_metadata):
    from joblib import dump
    from sklearn.metrics import roc_auc_score
    candidate_rows = []
    best_auc = -np.inf
    best_parameters = None
    best_model = None
    for candidate_index, parameters in enumerate(candidate_grid(model_name)):
        model = build_model(model_name, parameters)
        model.fit(train["features"], train["labels"])
        probabilities = model.predict_proba(validation["features"])[:, 1]
        validation_auc = float(roc_auc_score(validation["labels"], probabilities))
        candidate_rows.append({
            "candidate_index": candidate_index,
            "parameters": json.dumps(parameters, sort_keys=True),
            "validation_ROC_AUC": validation_auc,
        })
        if validation_auc > best_auc + 1e-12:
            best_auc = validation_auc
            best_parameters = parameters
            best_model = model
    require(best_parameters is not None and best_model is not None, "No candidate was selected")
    test_probabilities = best_model.predict_proba(test["features"])[:, 1]
    metrics = binary_metrics(test["labels"], test_probabilities)
    metrics["best_validation_ROC_AUC"] = best_auc

    run_name = f"{model_name}_{modality}_{variant}_fold{test_fold}"
    output = Path(output_root) / run_name
    require(not output.exists(), f"Output already exists: {output}")
    output.mkdir(parents=True)
    dump(best_model, output / "model.joblib")
    with (output / "candidate_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "full_key", "patient", "fid", "pos_x_mm", "pos_y_mm", "pos_z_mm",
            "outer_fold", "label", "probability", "prediction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, probability in enumerate(test_probabilities):
            position = test["positions_mm"][index]
            writer.writerow({
                "full_key": test["full_keys"][index],
                "patient": test["patient_ids"][index],
                "fid": test["finding_ids"][index],
                "pos_x_mm": position[0], "pos_y_mm": position[1], "pos_z_mm": position[2],
                "outer_fold": int(test["outer_folds"][index]),
                "label": int(test["labels"][index]),
                "probability": float(probability),
                "prediction": int(probability >= 0.5),
            })
    result = {
        "protocol_id": PROTOCOL_ID,
        "run_name": run_name,
        "model": model_name,
        "modality": modality,
        "variant": variant,
        "test_fold": test_fold,
        "validation_fold": split_metadata["validation_fold"],
        "seed": SEED,
        "selected_parameters": best_parameters,
        "feature_metadata": split_metadata,
        "counts": {
            "model_selection_train_findings": len(train["labels"]),
            "model_selection_validation_findings": len(validation["labels"]),
            "final_training_findings": len(train["labels"]),
            "test_findings": len(test["labels"]),
            "test_patients": len(set(test["patient_ids"])),
        },
        "metrics": metrics,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def requests_all():
    return [
        (model_name, modality, variant, test_fold)
        for model_name in MODELS
        for modality in MODALITY_CHANNELS
        for variant in valid_variants(modality)
        for test_fold in range(5)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-root", required=True)
    parser.add_argument("--output-root", default="prostatex_classical_baselines_v0_1")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--modality", choices=MODALITY_CHANNELS)
    parser.add_argument("--variant", choices=("primary", "sensitivity"))
    parser.add_argument("--test-fold", type=int, choices=range(5))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-inputs", action="store_true")
    arguments = parser.parse_args()

    folds = load_all_folds(arguments.patch_root)
    if arguments.validate_inputs:
        for modality in MODALITY_CHANNELS:
            for variant in valid_variants(modality):
                for test_fold in range(5):
                    train, validation, test, metadata = make_cycle(
                        folds, test_fold, modality, variant
                    )
                    print(modality, variant, test_fold, train["features"].shape,
                          validation["features"].shape, test["features"].shape,
                          metadata["features_per_channel"])
        print("Input validation complete; no classical model was fitted.")
        return

    if arguments.all:
        requests = requests_all()
    else:
        require(arguments.model is not None, "Set --model or --all")
        require(arguments.modality is not None, "Set --modality or --all")
        require(arguments.variant is not None, "Set --variant or --all")
        require(arguments.test_fold is not None, "Set --test-fold or --all")
        require(arguments.variant in valid_variants(arguments.modality),
                "T2-only has no sensitivity variant")
        requests = [(
            arguments.model, arguments.modality, arguments.variant, arguments.test_fold
        )]

    output_root = Path(arguments.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for model_name, modality, variant, test_fold in requests:
        run_name = f"{model_name}_{modality}_{variant}_fold{test_fold}"
        existing = output_root / run_name / "result.json"
        if arguments.resume and existing.is_file():
            result = json.loads(existing.read_text(encoding="utf-8"))
            require(result["run_name"] == run_name, "Existing run identity changed")
            results.append(result)
            print("Skipped completed run:", run_name)
            continue
        train, validation, test, metadata = make_cycle(folds, test_fold, modality, variant)
        result = fit_one(
            train, validation, test, model_name, modality, variant, test_fold,
            output_root, metadata,
        )
        results.append(result)
        print(run_name, json.dumps(result["metrics"], sort_keys=True))
    (output_root / "run_index.json").write_text(
        json.dumps({"protocol_id": PROTOCOL_ID, "runs": results},
                   indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()