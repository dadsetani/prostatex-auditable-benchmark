"""Step 7: seven paired patches for four QC patients; no training."""

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import scipy

PATCH_CHANNELS = ("T2_axial", "ADC")


def stored_volume(samples):
    planes = []
    for sample in samples:
        if any(sample.get(tag) is not None for tag in (
            "rescale_slope", "rescale_intercept", "rescale_type"
        )):
            raise ValueError("Rescale tags require review; no automatic conversion")
        if sample["photometric"] != "MONOCHROME2":
            raise ValueError("Expected MONOCHROME2")
        if not 1 <= sample["bits_stored"] <= 16:
            raise ValueError("Invalid stored-bit range")
        plane = np.frombuffer(sample["pixels"], dtype="<u2").reshape(
            sample["rows"], sample["columns"]
        )
        if int(plane.max()) >= 2 ** sample["bits_stored"]:
            raise ValueError("Pixels exceed stored-bit range")
        planes.append(plane)
    return np.stack(planes)


def make_remaining_patches(bundle_path, labels_path, output_dir):
    bundle_path, labels_path = Path(bundle_path), Path(labels_path)
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.with_suffix(".zip").exists():
        raise FileExistsError("Choose a new output directory and bundle name")
    geometry = audit_remaining_geometry(bundle_path, labels_path)
    patches, records, previews = [], [], []
    with ZipFile(bundle_path) as outer:
        downloads = json.loads(outer.read("download_report.json"))["series"]
        for patient, reports in sorted(geometry["series_by_patient"].items()):
            volumes, samples_by_channel = {}, {}
            for channel in PATCH_CHANNELS:
                report = reports[channel]
                record = next(row for row in downloads if row["series_uid"] == report["series_uid"])
                fresh, samples = audit(
                    GeometryMemoryZip(outer.read(record["file"])),
                    report["series_uid"], labels_path
                )
                if fresh != report:
                    raise ValueError("Source audit changed during execution")
                samples_by_channel[channel] = samples
                volumes[channel] = stored_volume(samples)
            orientation = samples_by_channel["T2_axial"][0]["orientation"]
            findings = sorted(reports["T2_axial"]["findings"], key=lambda row: (
                int(row["fid"]), tuple(row["pos_mm"])
            ))
            for finding in findings:
                point = finding["pos_mm"]
                world = physical_grid(point, orientation, size=100, spacing_mm=0.5)
                paired, bounds, native_views, nearest = [], {}, [], {}
                for channel in PATCH_CHANNELS:
                    report = reports[channel]
                    matching = [row for row in report["findings"] if (
                        row["fid"] == finding["fid"] and row["pos_mm"] == point
                    )]
                    if len(matching) != 1 or matching[0]["ClinSig"] != finding["ClinSig"]:
                        raise ValueError("Full finding key or label mismatch")
                    source_finding = matching[0]
                    affine = np.asarray(report["dicom_affine_ijk_to_patient"])
                    patch, bounds[channel] = sample_grid(volumes[channel], affine, world)
                    paired.append(patch)
                    nearest[channel] = source_finding["nearest_slice_zero_based"]
                    native_views.append((
                        volumes[channel][nearest[channel]],
                        source_finding["dicom_continuous_ijk"][:2]
                    ))
                patches.append(np.stack(paired))
                records.append({
                    "array_index": len(records), "patient": patient,
                    "fid": finding["fid"], "pos_mm": point,
                    "ClinSig": finding["ClinSig"],
                    "label": int(finding["ClinSig"].upper() == "TRUE"),
                    "bounds": bounds, "nearest_slice_zero_based": nearest,
                    "column_direction": orientation[:3].tolist(),
                    "row_direction": orientation[3:].tolist(),
                    "first_pixel_center_mm": world[:, 0, 0].tolist(),
                })
                previews.append(native_views)
    patches = np.stack(patches).astype(np.float32)
    keys = {(row["patient"], row["fid"], tuple(row["pos_mm"])) for row in records}
    if patches.shape != (7, 2, 100, 100) or len(keys) != 7 or not np.isfinite(patches).all():
        raise ValueError("Unexpected shape, duplicate full key, or nonfinite pixels")
    metadata = {
        "scope": "Seven findings from four development-QC patients; not a training or evaluation dataset",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "shape": list(patches.shape), "dtype": str(patches.dtype),
        "array_axes": ["finding", "channel", "row", "column"],
        "channels": list(PATCH_CHANNELS), "finding_key": ["patient", "fid", "pos_mm"],
        "ordering": "patient, integer fid, lexicographic pos_mm",
        "label_encoding": {"FALSE": 0, "TRUE": 1},
        "output_spacing_mm": [0.5, 0.5], "edge_to_edge_fov_mm": [50, 50],
        "pixel_center_offsets_mm": [-24.75, 24.75],
        "grid": "Through recorded pos, parallel to each patient's T2 plane; same world grid for both channels",
        "interpolation": "Trilinear; scipy.ndimage.map_coordinates order=1",
        "boundary_policy": "Reject sampling points outside source pixel centers; no padding or clipping",
        "intensities": "Stored values; no normalization; ADC physical units unverified",
        "preview": "Independent channel windows from native-slice percentiles; display only",
        "patch_array_sha256": hashlib.sha256(patches.tobytes(order="C")).hexdigest(),
        "bundle_sha256": geometry["bundle_sha256"],
        "label_csv_sha256": geometry["label_csv_sha256"],
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "findings": records,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output_dir / "patches.npz", patches=patches,
        labels=np.array([row["label"] for row in records], dtype=np.uint8),
        patient_ids=np.array([row["patient"] for row in records]),
        finding_ids=np.array([row["fid"] for row in records]),
        pos_mm=np.array([row["pos_mm"] for row in records]),
    )
    for name, content in (("metadata.json", metadata), ("geometry.json", geometry)):
        (output_dir / name).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    with PdfPages(output_dir / "preview.pdf") as pdf:
        for index, record in enumerate(records):
            figure, axes = plt.subplots(2, 2, figsize=(9, 8))
            try:
                for channel_index, channel in enumerate(PATCH_CHANNELS):
                    native, center = previews[index][channel_index]
                    low, high = np.percentile(native, [0, 99.5])
                    axes[channel_index, 0].imshow(native, cmap="gray", vmin=low, vmax=high)
                    axes[channel_index, 0].scatter(*center, marker="+", color="gold")
                    axes[channel_index, 0].set_title(f"{channel}: nearest native slice")
                    axes[channel_index, 1].imshow(
                        patches[index, channel_index], cmap="gray", vmin=low, vmax=high,
                        extent=(-25, 25, 25, -25), interpolation="nearest"
                    )
                    axes[channel_index, 1].scatter(0, 0, marker="+", color="gold")
                    axes[channel_index, 1].set_title("Resampled patch; offsets in mm")
                figure.suptitle(
                    f"Patch {index}: {record['patient']} fid={record['fid']} "
                    f"ClinSig={record['ClinSig']}\nStored values; no anatomical registration"
                )
                figure.tight_layout(rect=(0, 0, 1, 0.92))
                pdf.savefig(figure)
            finally:
                plt.close(figure)
    filenames = ("patches.npz", "metadata.json", "geometry.json", "preview.pdf")
    hashes = {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest() for name in filenames}
    (output_dir / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    bundle = output_dir.with_suffix(".zip")
    with ZipFile(bundle, "x", compression=ZIP_DEFLATED) as archive:
        for name in (*filenames, "artifact_hashes.json"):
            archive.write(output_dir / name, arcname=name)
    print("Patch shape:", patches.shape, "Labels:", [row["label"] for row in records])
    print("Out-of-bounds points: 0; no padding, normalization, or training.")
    print("Saved:", bundle)
    return bundle


if __name__ == "__main__":
    required = ("audit_remaining_geometry", "audit", "GeometryMemoryZip",
                "physical_grid", "sample_grid", "remaining_bundle", "LABEL_ZIP")
    if any(name not in globals() for name in required):
        raise RuntimeError("Run the corrected pilot and Steps 5–6 first")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    patches_bundle = make_remaining_patches(
        remaining_bundle, LABEL_ZIP, Path(f"remaining_four_patches_{stamp}")
    )
    from IPython.display import FileLink, display
    display(FileLink(str(patches_bundle)))