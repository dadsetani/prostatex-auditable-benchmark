"""Audit all five frozen PROSTATEx patch bundles against the eligibility register."""

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


REGISTER_SHA256 = "af602758925b35388a29322a17366dc3057729efcec824221456c3073ca06891"
EXPECTED_FOLD_COUNTS = {0: 64, 1: 57, 2: 62, 3: 62, 4: 61}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return hashlib.sha256(
        (json.dumps(data, indent=2, allow_nan=False) + "\n").encode("utf-8")
    ).hexdigest()


def audit(analysis_dir="analysis"):
    analysis_dir = Path(analysis_dir)
    register_path = analysis_dir / "prostatex_eligibility_register_v0_3.json"
    require(canonical_json_sha256(register_path) == REGISTER_SHA256,
            "Eligibility register content changed")
    register = json.loads(register_path.read_text(encoding="utf-8"))
    expected = {
        row["full_key"]: row for row in register["rows"] if row["eligibility"] == "ACCEPTED"
    }
    require(len(expected) == 306, "Unexpected accepted-register size")

    observed = {}
    fold_results = []
    all_series = set()
    maximum_snap = 0.0
    for fold, expected_count in EXPECTED_FOLD_COUNTS.items():
        directory = analysis_dir / f"prostatex_patches_fold_{fold}_v0_3"
        bundle = directory.with_suffix(".zip")
        require(directory.is_dir() and bundle.is_file(), f"Missing fold {fold} output")
        artifact_hashes = json.loads((directory / "artifact_hashes.json").read_text())
        for name, item in artifact_hashes.items():
            path = directory / name
            require(path.stat().st_size == item["bytes"], f"Fold {fold} artifact size changed")
            require(sha256_path(path) == item["sha256"], f"Fold {fold} artifact hash changed")
        with ZipFile(bundle) as archive:
            require(archive.testzip() is None, f"Fold {fold} bundle CRC failure")
            require(len(archive.namelist()) == len(set(archive.namelist())), "Duplicate bundle member")
            require(set(archive.namelist()) == {*artifact_hashes, "artifact_hashes.json"},
                    f"Fold {fold} bundle member mismatch")
            for name, item in artifact_hashes.items():
                require(hashlib.sha256(archive.read(name)).hexdigest() == item["sha256"],
                        f"Fold {fold} archived artifact hash changed")

        metadata = json.loads((directory / "metadata.json").read_text())
        with np.load(directory / "patches.npz", allow_pickle=False) as arrays:
            patches = arrays["patches"]
            labels = arrays["labels"]
            full_keys = arrays["full_keys"].tolist()
            outer_folds = arrays["outer_folds"]
            require(patches.shape == (expected_count, 2, 100, 100), "Unexpected patch shape")
            require(patches.dtype == np.float32 and np.isfinite(patches).all(), "Invalid patch values")
            require(np.array_equal(outer_folds, np.full(expected_count, fold, dtype=np.uint8)),
                    "Unexpected fold array")
            require(hashlib.sha256(patches.tobytes(order="C")).hexdigest()
                    == metadata["patch_array_sha256"], "Patch-array hash mismatch")
            require(len(full_keys) == len(set(full_keys)) == expected_count, "Duplicate fold finding key")
            for index, full_key in enumerate(full_keys):
                require(full_key in expected, f"Unregistered patch: {full_key}")
                require(int(labels[index]) == expected[full_key]["label"], "Patch label mismatch")
                require(full_key not in observed, f"Cross-fold duplicate patch: {full_key}")
                observed[full_key] = fold

        for finding in metadata["findings"]:
            require(finding["full_key"] in expected, "Metadata finding is not accepted")
            for channel in ("T2_axial", "ADC"):
                maximum_snap = max(
                    maximum_snap,
                    finding["bounds"][channel]["maximum_boundary_snap_voxels"],
                )
                all_series.add(finding["series_uids"][channel])
        fold_results.append({
            "outer_fold": fold,
            "findings": expected_count,
            "ClinSig_TRUE": int(labels.sum()),
            "ClinSig_FALSE": int((labels == 0).sum()),
            "patch_shape": list(patches.shape),
            "bundle": bundle.name,
            "bundle_bytes": bundle.stat().st_size,
            "bundle_sha256": sha256_path(bundle),
            "patches_npz_sha256": artifact_hashes["patches.npz"]["sha256"],
            "patch_array_sha256": metadata["patch_array_sha256"],
        })

    require(set(observed) == set(expected), "Patch keys do not equal frozen accepted register")
    true_count = sum(row["ClinSig_TRUE"] for row in fold_results)
    false_count = sum(row["ClinSig_FALSE"] for row in fold_results)
    require((true_count, false_count) == (65, 241), "Unexpected all-fold labels")
    require(len(all_series) == 384, "Unexpected number of used source series")
    require(maximum_snap <= 1e-4, "Boundary snap exceeds frozen tolerance")
    report = {
        "audit_id": "PROSTATEx-paired-patches-all-folds-v0.3",
        "checked_date": "2026-09-27",
        "status": "COMPLETE",
        "eligibility_register_sha256": REGISTER_SHA256,
        "summary": {
            "outer_folds": 5,
            "patients": len({row["patient"] for row in expected.values()}),
            "findings": len(observed),
            "ClinSig_TRUE": true_count,
            "ClinSig_FALSE": false_count,
            "unique_source_series": len(all_series),
            "patch_shape_per_finding": [2, 100, 100],
            "dtype": "float32",
            "maximum_numerical_boundary_snap_voxels": maximum_snap,
        },
        "fold_results": fold_results,
        "checks": {
            "all_bundle_crc_checks_passed": True,
            "all_artifact_hashes_passed": True,
            "all_patch_arrays_finite": True,
            "all_full_keys_match_frozen_register": True,
            "all_labels_match_frozen_register": True,
            "fold_assignments_preserved": True,
            "padding_or_clinical_clipping_used": False,
        },
        "limitations": [
            "Stored values are unnormalized and ADC physical units remain unverified.",
            "Common physical-grid resampling does not establish anatomical registration or diagnostic image quality.",
            "No image-quality review, model fitting, hyperparameter selection, or clinical validation is included.",
        ],
        "next_step": "Perform a blinded technical visual-QC pass on frozen patches before any model-development stage.",
    }
    output = analysis_dir / "prostatex_all_folds_patch_audit.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    audit()