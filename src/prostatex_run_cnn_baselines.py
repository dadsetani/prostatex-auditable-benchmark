"""Run frozen T2-only, ADC-only, and paired PROSTATEx CNN baselines."""

import argparse
import csv
import hashlib
import io
import json
import random
from pathlib import Path
from zipfile import ZipFile

import numpy as np


PROTOCOL_ID = "PROSTATEx-modeling-v0.1"
SENSITIVITY_KEY = "ProstateX-0052|1|26.6078 10.1553 50.5636"
EXPECTED_COUNTS = {0: 64, 1: 57, 2: 62, 3: 62, 4: 61}
MODALITY_CHANNELS = {
    "t2": [0],
    "adc": [1],
    "paired": [0, 1],
}


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
    keys = [key for fold in folds.values() for key in fold["full_keys"].tolist()]
    require(len(keys) == len(set(keys)) == 306, "All-fold keys changed")
    labels = np.concatenate([folds[index]["labels"] for index in range(5)])
    require((int(labels.sum()), int((labels == 0).sum())) == (65, 241), "All-fold labels changed")
    return folds


def subset_fold(arrays, modality, sensitivity):
    indices = MODALITY_CHANNELS[modality]
    keep = np.ones(len(arrays["labels"]), dtype=bool)
    if sensitivity:
        require(modality in {"adc", "paired"}, "Sensitivity exclusion applies only to ADC models")
        keep &= arrays["full_keys"] != SENSITIVITY_KEY
    return {
        "x": arrays["patches"][keep][:, indices].astype(np.float32, copy=True),
        "y": arrays["labels"][keep].astype(np.float32, copy=True),
        "patient_ids": arrays["patient_ids"][keep].astype(str),
        "finding_ids": arrays["finding_ids"][keep].astype(str),
        "positions_mm": arrays["positions_mm"][keep].astype(np.float64, copy=True),
        "full_keys": arrays["full_keys"][keep].astype(str),
        "outer_folds": arrays["outer_folds"][keep].astype(np.uint8, copy=True),
    }


def concatenate(parts):
    return {name: np.concatenate([part[name] for part in parts], axis=0) for name in parts[0]}


def make_cycle(folds, test_fold, modality, variant):
    sensitivity = variant == "sensitivity"
    validation_fold = (test_fold + 1) % 5
    training_folds = [fold for fold in range(5) if fold not in {test_fold, validation_fold}]
    train = concatenate([subset_fold(folds[fold], modality, sensitivity) for fold in training_folds])
    validation = subset_fold(folds[validation_fold], modality, sensitivity)
    test = subset_fold(folds[test_fold], modality, sensitivity)
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        require(len(split["y"]) > 0 and len(np.unique(split["y"])) == 2,
                f"{name} split lacks both classes")
        require(len(set(split["patient_ids"])) > 0, f"{name} split has no patients")
    require(set(train["patient_ids"]).isdisjoint(validation["patient_ids"]), "Train/validation leakage")
    require(set(train["patient_ids"]).isdisjoint(test["patient_ids"]), "Train/test leakage")
    require(set(validation["patient_ids"]).isdisjoint(test["patient_ids"]), "Validation/test leakage")
    mean = train["x"].mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    standard_deviation = train["x"].std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    require(np.all(standard_deviation >= 1e-6), "Training channel standard deviation is too small")
    for split in (train, validation, test):
        split["x"] = ((split["x"] - mean[None, :, None, None])
                      / standard_deviation[None, :, None, None]).astype(np.float32)
        require(np.isfinite(split["x"]).all(), "Normalization produced nonfinite values")
    return train, validation, test, {
        "training_folds": training_folds,
        "validation_fold": validation_fold,
        "test_fold": test_fold,
        "channel_mean": mean.tolist(),
        "channel_standard_deviation": standard_deviation.tolist(),
    }


def binary_metrics(labels, probabilities):
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_score,
        roc_auc_score,
    )

    labels = np.asarray(labels, dtype=np.uint8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= 0.5).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "ROC_AUC": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "F1": float(f1_score(labels, predictions, zero_division=0)),
        "Brier_score": float(brier_score_loss(labels, probabilities)),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }


def seed_everything(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def train_one(train, validation, test, modality, variant, test_fold, seed, output_root):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from sklearn.metrics import roc_auc_score

    seed_everything(seed, torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class ArrayDataset(Dataset):
        def __init__(self, split):
            self.x = torch.from_numpy(split["x"])
            self.y = torch.from_numpy(split["y"])

        def __len__(self):
            return len(self.y)

        def __getitem__(self, index):
            return self.x[index], self.y[index]

    class FixedCNN(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(channels, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.25),
                nn.Linear(32, 1),
            )

        def forward(self, inputs):
            return self.network(inputs).squeeze(1)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        ArrayDataset(train), batch_size=32, shuffle=True, generator=generator,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(ArrayDataset(validation), batch_size=64, shuffle=False)
    test_loader = DataLoader(ArrayDataset(test), batch_size=64, shuffle=False)
    model = FixedCNN(train["x"].shape[1]).to(device)
    negatives = float(np.sum(train["y"] == 0))
    positives = float(np.sum(train["y"] == 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def predict(loader):
        model.eval()
        labels, probabilities, losses = [], [], []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                losses.append(float(criterion(logits, targets).item()) * len(targets))
                labels.append(targets.cpu().numpy())
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels = np.concatenate(labels)
        probabilities = np.concatenate(probabilities)
        return labels, probabilities, sum(losses) / len(labels)

    best_auc = -np.inf
    best_epoch = None
    best_state = None
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, 101):
        model.train()
        training_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            training_loss += float(loss.item()) * len(targets)
        validation_labels, validation_probabilities, validation_loss = predict(validation_loader)
        validation_auc = float(roc_auc_score(validation_labels, validation_probabilities))
        history.append({
            "epoch": epoch,
            "training_loss": training_loss / len(train["y"]),
            "validation_loss": validation_loss,
            "validation_ROC_AUC": validation_auc,
        })
        if validation_auc > best_auc + 1e-12:
            best_auc = validation_auc
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 15:
                break
    require(best_state is not None, "No best model was selected")
    model.load_state_dict(best_state)
    test_labels, test_probabilities, test_loss = predict(test_loader)
    require(np.array_equal(test_labels, test["y"]), "Test order changed")
    metrics = binary_metrics(test_labels, test_probabilities)
    metrics["test_loss"] = test_loss
    metrics["best_validation_ROC_AUC"] = best_auc
    metrics["best_epoch"] = best_epoch

    run_name = f"{modality}_{variant}_fold{test_fold}_seed{seed}"
    output = Path(output_root) / run_name
    require(not output.exists(), f"Output already exists: {output}")
    output.mkdir(parents=True)
    torch.save(best_state, output / "model_state.pt")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    prediction_fields = [
        "full_key", "patient", "fid", "pos_x_mm", "pos_y_mm", "pos_z_mm",
        "outer_fold", "label", "probability", "prediction",
    ]
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        for index, probability in enumerate(test_probabilities):
            position = test["positions_mm"][index]
            writer.writerow({
                "full_key": test["full_keys"][index],
                "patient": test["patient_ids"][index],
                "fid": test["finding_ids"][index],
                "pos_x_mm": position[0],
                "pos_y_mm": position[1],
                "pos_z_mm": position[2],
                "outer_fold": int(test["outer_folds"][index]),
                "label": int(test_labels[index]),
                "probability": float(probability),
                "prediction": int(probability >= 0.5),
            })
    result = {
        "protocol_id": PROTOCOL_ID,
        "run_name": run_name,
        "modality": modality,
        "variant": variant,
        "test_fold": test_fold,
        "validation_fold": (test_fold + 1) % 5,
        "seed": seed,
        "device": str(device),
        "counts": {
            "train_findings": len(train["y"]),
            "validation_findings": len(validation["y"]),
            "test_findings": len(test["y"]),
            "train_patients": len(set(train["patient_ids"])),
            "validation_patients": len(set(validation["patient_ids"])),
            "test_patients": len(set(test["patient_ids"])),
        },
        "metrics": metrics,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def valid_variants(modality):
    return ["primary"] if modality == "t2" else ["primary", "sensitivity"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-root", required=True)
    parser.add_argument("--output-root", default="prostatex_baselines_v0_1")
    parser.add_argument("--modality", choices=MODALITY_CHANNELS)
    parser.add_argument("--variant", choices=("primary", "sensitivity"))
    parser.add_argument("--test-fold", type=int, choices=range(5))
    parser.add_argument("--seed", type=int, choices=(1729, 2718, 3141))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-inputs", action="store_true")
    arguments = parser.parse_args()

    folds = load_all_folds(arguments.patch_root)
    if arguments.validate_inputs:
        for modality in MODALITY_CHANNELS:
            for variant in valid_variants(modality):
                for test_fold in range(5):
                    train, validation, test, normalization = make_cycle(
                        folds, test_fold, modality, variant
                    )
                    print(modality, variant, test_fold,
                          len(train["y"]), len(validation["y"]), len(test["y"]),
                          normalization["channel_mean"], normalization["channel_standard_deviation"])
        print("Input validation complete; no model was fitted.")
        return

    if arguments.all:
        requests = [
            (modality, variant, test_fold, seed)
            for modality in MODALITY_CHANNELS
            for variant in valid_variants(modality)
            for test_fold in range(5)
            for seed in (1729, 2718, 3141)
        ]
    else:
        require(arguments.modality is not None, "Set --modality or --all")
        require(arguments.variant is not None, "Set --variant or --all")
        require(arguments.test_fold is not None, "Set --test-fold or --all")
        require(arguments.seed is not None, "Set --seed or --all")
        require(arguments.variant in valid_variants(arguments.modality),
                "T2-only has no sensitivity variant")
        requests = [(arguments.modality, arguments.variant, arguments.test_fold, arguments.seed)]

    output_root = Path(arguments.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for modality, variant, test_fold, seed in requests:
        run_name = f"{modality}_{variant}_fold{test_fold}_seed{seed}"
        existing_result = output_root / run_name / "result.json"
        if arguments.resume and existing_result.is_file():
            result = json.loads(existing_result.read_text(encoding="utf-8"))
            require(result["run_name"] == run_name, "Existing result identity changed")
            results.append(result)
            print("Skipped completed run:", run_name)
            continue
        train, validation, test, normalization = make_cycle(
            folds, test_fold, modality, variant
        )
        result = train_one(
            train, validation, test, modality, variant, test_fold, seed, output_root
        )
        result["normalization"] = normalization
        (output_root / result["run_name"] / "result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        results.append(result)
        print(result["run_name"], json.dumps(result["metrics"], sort_keys=True))
    (output_root / "run_index.json").write_text(
        json.dumps({"protocol_id": PROTOCOL_ID, "runs": results}, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()