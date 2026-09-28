"""Prepare identity- and label-hidden visual QC materials for all frozen patches."""

import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


PATCH_AUDIT_SHA256 = None
REGISTER_SHA256 = "af602758925b35388a29322a17366dc3057729efcec824221456c3073ca06891"
CHANNELS = ("T2_axial", "ADC")

README = """# Blinded Technical Visual QC

Review `qc_preview.pdf` in `review_id` order and complete `review.csv`.
The preview and worksheet omit patient identity, finding identity, fold, zone, and ClinSig label.
Do not open the separately retained identity key until the visual statuses are frozen.

Allowed `visual_qc_status` values:
- `PENDING`
- `NO_OBVIOUS_TECHNICAL_ISSUE`
- `REVIEW_REQUIRED`
- `UNASSESSABLE`

Check only for obvious technical display/content problems, such as a blank or nearly constant
patch, severe interpolation failure, gross truncation, or an unusable channel. Independent
percentile windows are display-only. Differences between T2 and ADC intensity or appearance
are not by themselves failures. The center marker is the supplied finding position, not an
independently verified lesion center. Do not infer labels, diagnosis, registration quality,
or clinical suitability. No visual status automatically changes dataset eligibility.
"""


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


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def patch_metrics(patch):
    percentiles = np.percentile(patch, [0.5, 1, 25, 50, 75, 99, 99.5])
    return {
        "min": float(patch.min()),
        "max": float(patch.max()),
        "mean": float(patch.mean()),
        "std": float(patch.std()),
        "p00_5": float(percentiles[0]),
        "p01": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "p50": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p99": float(percentiles[5]),
        "p99_5": float(percentiles[6]),
        "zero_fraction": float(np.mean(patch == 0)),
        "finite": bool(np.isfinite(patch).all()),
    }


def prepare(analysis_dir="analysis", patch_root=None):
    analysis_dir = Path(analysis_dir)
    patch_root = Path(patch_root) if patch_root is not None else analysis_dir
    output = analysis_dir / "prostatex_blinded_visual_qc_v0_3"
    reviewer_bundle = analysis_dir / "prostatex_blinded_visual_qc_reviewer_v0_3.zip"
    require(not output.exists() and not reviewer_bundle.exists(), "QC output already exists")
    register_path = analysis_dir / "prostatex_eligibility_register_v0_3.json"
    require(canonical_json_sha256(register_path) == REGISTER_SHA256,
            "Eligibility register content changed")

    patch_audit = json.loads((analysis_dir / "prostatex_all_folds_patch_audit.json").read_text())
    audited_bundles = {row["outer_fold"]: row for row in patch_audit["fold_results"]}
    entries = []
    for fold in range(5):
        patch_directory = patch_root / f"prostatex_patches_fold_{fold}_v0_3"
        hashes = json.loads((patch_directory / "artifact_hashes.json").read_text())
        for name, item in hashes.items():
            path = patch_directory / name
            require(path.stat().st_size == item["bytes"], f"Fold {fold} artifact size changed")
            require(sha256_path(path) == item["sha256"], f"Fold {fold} artifact hash changed")
        metadata = json.loads((patch_directory / "metadata.json").read_text())
        with np.load(patch_directory / "patches.npz", allow_pickle=False) as arrays:
            patches = arrays["patches"].copy()
            full_keys = arrays["full_keys"].tolist()
        require(len(metadata["findings"]) == len(full_keys) == len(patches), "Patch metadata mismatch")
        for index, full_key in enumerate(full_keys):
            finding = metadata["findings"][index]
            require(finding["full_key"] == full_key, "Patch full key changed")
            entries.append({
                "full_key": full_key,
                "patient": finding["patient"],
                "fid": finding["fid"],
                "pos_mm": finding["pos_mm"],
                "outer_fold": fold,
                "source_array_index": index,
                "source_bundle": audited_bundles[fold]["bundle"],
                "source_bundle_sha256": audited_bundles[fold]["bundle_sha256"],
                "patch": patches[index],
            })

    require(len(entries) == 306 and len({row["full_key"] for row in entries}) == 306,
            "Unexpected all-fold patch count")
    entries.sort(key=lambda row: hashlib.sha256(
        (REGISTER_SHA256 + "|blinded-v0.3|" + row["full_key"]).encode("utf-8")
    ).hexdigest())
    output.mkdir(parents=True)

    review_rows = []
    identity_rows = []
    for index, entry in enumerate(entries, start=1):
        review_id = f"BQC{index:03d}"
        metrics = {channel: patch_metrics(entry["patch"][channel_index])
                   for channel_index, channel in enumerate(CHANNELS)}
        review_row = {
            "review_id": review_id,
            "visual_qc_status": "PENDING",
            "notes": "",
            "reviewer": "",
            "reviewed_utc": "",
        }
        for channel in CHANNELS:
            for name in ("min", "max", "mean", "std", "p01", "p50", "p99", "zero_fraction"):
                review_row[f"{channel}_{name}"] = metrics[channel][name]
        review_rows.append(review_row)
        identity_rows.append({
            "review_id": review_id,
            "full_key": entry["full_key"],
            "patient": entry["patient"],
            "fid": entry["fid"],
            "pos_mm": json.dumps(entry["pos_mm"]),
            "outer_fold": entry["outer_fold"],
            "source_array_index": entry["source_array_index"],
            "source_bundle": entry["source_bundle"],
            "source_bundle_sha256": entry["source_bundle_sha256"],
        })
        entry["review_id"] = review_id
        entry["metrics"] = metrics

    review_fields = list(review_rows[0])
    identity_fields = list(identity_rows[0])
    write_csv(output / "review.csv", review_fields, review_rows)
    write_csv(output / "identity_key.csv", identity_fields, identity_rows)
    (output / "README.md").write_text(README, encoding="utf-8")

    preview_path = output / "qc_preview.pdf"
    with PdfPages(preview_path) as pdf:
        for entry in entries:
            figure, axes = plt.subplots(1, 2, figsize=(10, 5))
            try:
                for channel_index, channel in enumerate(CHANNELS):
                    patch = entry["patch"][channel_index]
                    low = entry["metrics"][channel]["p00_5"]
                    high = entry["metrics"][channel]["p99_5"]
                    if high <= low:
                        low, high = float(patch.min()), float(patch.max())
                    if high <= low:
                        high = low + 1
                    axes[channel_index].imshow(
                        patch, cmap="gray", vmin=low, vmax=high,
                        extent=(-25, 25, 25, -25), interpolation="nearest"
                    )
                    axes[channel_index].scatter(0, 0, marker="+", color="gold")
                    axes[channel_index].set_title(channel)
                    axes[channel_index].set_xlabel("Offset from supplied point (mm)")
                figure.suptitle(
                    f"{entry['review_id']}\nIdentity and label hidden; independent display windows",
                    fontsize=13,
                )
                figure.tight_layout(rect=(0, 0, 1, 0.9))
                pdf.savefig(figure)
            finally:
                plt.close(figure)

    public_names = ("review.csv", "README.md", "qc_preview.pdf")
    manifest = {
        "qc_id": "PROSTATEx-blinded-technical-visual-QC-v0.3",
        "created_date": "2026-09-27",
        "review_count": len(entries),
        "initial_status": "PENDING",
        "identity_hidden_from_reviewer_bundle": True,
        "ClinSig_hidden_from_all_QC_materials": True,
        "randomization": "Ascending SHA-256 of eligibility-register hash, fixed QC salt, and full finding key",
        "eligibility_register_sha256": REGISTER_SHA256,
        "reviewer_artifacts": {
            name: {"bytes": (output / name).stat().st_size, "sha256": sha256_path(output / name)}
            for name in public_names
        },
        "identity_key": {
            "file": "identity_key.csv",
            "bytes": (output / "identity_key.csv").stat().st_size,
            "sha256": sha256_path(output / "identity_key.csv"),
            "instruction": "Retain separately and do not provide to the visual reviewer until statuses are frozen.",
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with ZipFile(reviewer_bundle, "x", compression=ZIP_DEFLATED, allowZip64=True) as archive:
        for name in (*public_names, "manifest.json"):
            archive.write(output / name, arcname=name)
    print("Prepared:", len(entries), "identity- and label-hidden QC pages")
    print("Reviewer bundle:", reviewer_bundle)
    print("Separate identity key:", output / "identity_key.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="analysis")
    parser.add_argument("--patch-root")
    arguments = parser.parse_args()
    prepare(arguments.analysis_dir, arguments.patch_root)