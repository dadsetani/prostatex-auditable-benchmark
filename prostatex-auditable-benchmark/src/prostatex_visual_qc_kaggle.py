"""Step 8: prepare a label-hidden visual QC worksheet; all reviews start pending."""

import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

QC_GUIDE = """# Step 8 — Human Visual QC

Review qc_preview.pdf and complete review.csv without changing identity columns.
Identity columns: review_id, array_index, patient, fid, pos_mm, source_bundle_sha256.
Each page corresponds to one full (patient, fid, pos_mm) key.

Allowed visual_qc_status values:
- PENDING: not reviewed.
- NO_OBVIOUS_TECHNICAL_ISSUE: no obvious display/content problem in this limited view.
- REVIEW_REQUIRED: a suspected issue needs investigation; describe it.
- UNASSESSABLE: this view or the reviewer's expertise is insufficient; explain why.

Record notes, reviewer, and reviewed_utc (ISO 8601 UTC) for each completed review.
Check for blank/unusable displays, conspicuous artifacts, and gross differences
between channels. Do not infer cancer labels or claim accurate registration.
If unsure, use REVIEW_REQUIRED or UNASSESSABLE rather than guessing.
No status automatically includes or excludes a finding from a dataset.

ClinSig labels are omitted from these views, but may have been seen previously;
this is not a formally blinded review. Windows use per-patch percentiles for
display only. Zero fractions are descriptive, not rejection thresholds; source
zeros are not proof of padding. The center marker denotes the supplied finding
position, not an independently verified lesion center. Reviewing these patches
alone does not establish anatomy, ADC units, clinical quality, or training readiness.

All rows are initialized as PENDING. Numerical reproducibility is not human QC.
ProstateX-0001 remains outside this worksheet. No patches are modified.
Keep the original bundle and save the completed CSV as a separate file.
"""


def prepare_visual_qc(input_bundle, output_dir):
    input_bundle, output_dir = Path(input_bundle), Path(output_dir)
    if output_dir.exists() or output_dir.with_suffix(".zip").exists():
        raise FileExistsError("Choose a new QC output directory")
    source_hash = hashlib.sha256(input_bundle.read_bytes()).hexdigest()
    with ZipFile(input_bundle) as archive:
        expected_files = {"patches.npz", "metadata.json", "geometry.json", "preview.pdf", "artifact_hashes.json"}
        if set(archive.namelist()) != expected_files or len(archive.namelist()) != 5:
            raise ValueError("Unexpected or duplicate bundle members")
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC failure")
        hashes = json.loads(archive.read("artifact_hashes.json"))
        if set(hashes) != expected_files - {"artifact_hashes.json"}:
            raise ValueError("Incomplete artifact hash manifest")
        for name, digest in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"Artifact hash mismatch: {name}")
        metadata = json.loads(archive.read("metadata.json"))
        with np.load(io.BytesIO(archive.read("patches.npz")), allow_pickle=False) as loaded:
            arrays = {name: loaded[name].copy() for name in loaded.files}
    patches = arrays["patches"]
    if patches.shape != (7, 2, 100, 100) or patches.dtype != np.float32 or not np.isfinite(patches).all():
        raise ValueError("Unexpected patch array")
    if hashlib.sha256(patches.tobytes(order="C")).hexdigest() != metadata["patch_array_sha256"]:
        raise ValueError("Patch array hash mismatch")
    if metadata["channels"] != ["T2_axial", "ADC"] or len(metadata["findings"]) != 7:
        raise ValueError("Unexpected channels or finding count")
    if metadata["edge_to_edge_fov_mm"] != [50, 50] or metadata["array_axes"] != ["finding", "channel", "row", "column"]:
        raise ValueError("Unexpected grid specification")
    rows, keys = [], set()
    for index, finding in enumerate(metadata["findings"]):
        key = (finding["patient"], finding["fid"], tuple(finding["pos_mm"]))
        if key in keys or finding["array_index"] != index:
            raise ValueError("Duplicate key or incorrect array index")
        keys.add(key)
        if arrays["patient_ids"][index] != key[0] or arrays["finding_ids"][index] != key[1]:
            raise ValueError("NPZ identity mismatch")
        np.testing.assert_array_equal(arrays["pos_mm"][index], finding["pos_mm"])
        if finding["ClinSig"] not in {"TRUE", "FALSE"} or arrays["labels"][index] != finding["label"] or finding["label"] != int(finding["ClinSig"] == "TRUE"):
            raise ValueError("NPZ label mismatch")
        row = {
            "review_id": f"QC{index + 1:02d}", "array_index": index,
            "patient": key[0], "fid": key[1], "pos_mm": json.dumps(key[2]),
            "source_bundle_sha256": source_hash,
            "visual_qc_status": "PENDING", "notes": "", "reviewer": "", "reviewed_utc": "",
        }
        for channel_index, channel in enumerate(metadata["channels"]):
            patch = patches[index, channel_index]
            row[f"{channel}_min"] = float(patch.min())
            row[f"{channel}_max"] = float(patch.max())
            row[f"{channel}_zero_fraction"] = float(np.mean(patch == 0))
        rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "review.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "README.md").write_text(QC_GUIDE, encoding="utf-8")
    with PdfPages(output_dir / "qc_preview.pdf") as pdf:
        for index, row in enumerate(rows):
            figure, axes = plt.subplots(1, 2, figsize=(10, 5))
            try:
                for channel_index, channel in enumerate(metadata["channels"]):
                    patch = patches[index, channel_index]
                    low, high = np.percentile(patch, [0, 99.5])
                    axes[channel_index].imshow(patch, cmap="gray", vmin=low, vmax=high,
                                              extent=(-25, 25, 25, -25), interpolation="nearest")
                    axes[channel_index].scatter(0, 0, marker="+", color="gold")
                    axes[channel_index].set_title(channel)
                    axes[channel_index].set_xlabel("Offset from supplied point (mm)")
                figure.suptitle(f"{row['review_id']} | {row['patient']} | fid={row['fid']}\n"
                               "Labels hidden; independent display windows; no registration")
                figure.tight_layout(rect=(0, 0, 1, .88))
                pdf.savefig(figure)
            finally:
                plt.close(figure)
    names = ("review.csv", "README.md", "qc_preview.pdf")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_bundle_sha256": source_hash,
        "patch_array_sha256": metadata["patch_array_sha256"],
        "review_count": len(rows), "initial_status": "PENDING",
        "artifact_sha256": {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest() for name in names},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    bundle = output_dir.with_suffix(".zip")
    with ZipFile(bundle, "x", compression=ZIP_DEFLATED) as archive:
        for name in (*names, "manifest.json"):
            archive.write(output_dir / name, arcname=name)
    print("Prepared 7 QC rows: all PENDING. No automatic acceptance or exclusion.")
    print("Saved:", bundle)
    return bundle


if __name__ == "__main__":
    if "patches_bundle" not in globals():
        raise RuntimeError("Set patches_bundle to the existing Step 7 ZIP path")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    qc_bundle = prepare_visual_qc(patches_bundle, Path(f"remaining_four_visual_qc_{stamp}"))
    from IPython.display import FileLink, display
    display(FileLink(str(qc_bundle)))