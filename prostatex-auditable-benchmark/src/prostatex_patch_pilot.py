"""Extract two-channel physical patches for ProstateX-0002 only, not training data validation."""

import json
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.ndimage import map_coordinates

from prostatex_geometry_check import audit

CHANNELS = ("T2_axial", "ADC")
EXPECTED_UIDS = (
    "1.3.6.1.4.1.14519.5.2.1.7311.5101.479812804428819225709948044920",
    "1.3.6.1.4.1.14519.5.2.1.7311.5101.227755318954832018272622878067",
)


def physical_grid(point, orientation, size=100, spacing_mm=0.5):
    offsets = (np.arange(size) - (size - 1) / 2) * spacing_mm
    columns, rows = np.meshgrid(offsets, offsets, indexing="xy")
    world = (np.asarray(point)[:, None, None]
             + orientation[:3, None, None] * columns
             + orientation[3:, None, None] * rows)
    return world


def sample_grid(volume, affine, world):
    homogeneous = np.vstack((world.reshape(3, -1), np.ones((1, world.shape[1] * world.shape[2]))))
    indices = np.linalg.solve(affine, homogeneous)[:3]
    limits = np.array(volume.shape[::-1]) - 1
    outside = np.any((indices < 0) | (indices > limits[:, None]), axis=0)
    if outside.any():
        raise ValueError(f"Out of bounds: {outside.sum()}/{outside.size} sampling points; no padding applied")
    values = map_coordinates(volume.astype(np.float32), indices[::-1], order=1,
                             mode="constant", cval=np.nan, prefilter=False)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite interpolated values")
    return values.reshape(world.shape[1:]), {
        "min_source_ijk": indices.min(axis=1).tolist(),
        "max_source_ijk": indices.max(axis=1).tolist(),
        "outside_points": int(outside.sum()),
    }


def make_pilot(t2_zip, adc_zip, labels_zip, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose a new empty output directory: {output_dir}")
    reports, samples_by_channel, volumes = {}, {}, {}
    for name, image_zip, uid in zip(CHANNELS, (t2_zip, adc_zip), EXPECTED_UIDS):
        report, samples = audit(Path(image_zip), uid, Path(labels_zip))
        if report["patient"] != "ProstateX-0002":
            raise ValueError("This pilot is limited to ProstateX-0002")
        volume = []
        for sample in samples:
            if any(sample.get(tag) is not None for tag in ("rescale_slope", "rescale_intercept", "rescale_type")):
                raise ValueError("Unexpected rescale tags: this stored-value pilot requires review")
            if sample["photometric"] != "MONOCHROME2":
                raise ValueError("This pilot expects MONOCHROME2")
            pixels = np.frombuffer(sample["pixels"], dtype="<u2").reshape(sample["rows"], sample["columns"])
            if int(pixels.max()) >= 2 ** sample["bits_stored"]:
                raise ValueError("Values exceed the declared stored bit range")
            volume.append(pixels)
        reports[name], samples_by_channel[name] = report, samples
        volumes[name] = np.stack(volume)
    for field in ("patient", "study_uid", "frame_uid"):
        if reports["T2_axial"][field] != reports["ADC"][field]:
            raise ValueError(f"Cross-series mismatch: {field}")
    findings = {}
    for name in CHANNELS:
        rows = reports[name]["findings"]
        findings[name] = {(row["fid"], tuple(row["pos_mm"])): row for row in rows}
        if len(findings[name]) != len(rows):
            raise ValueError("Duplicate full finding keys")
    if set(findings["T2_axial"]) != set(findings["ADC"]):
        raise ValueError("Different findings between channels")
    keys = sorted(findings["T2_axial"], key=lambda key: (int(key[0]), key[1]))
    if len(keys) != 2:
        raise ValueError("Expected two findings in this specific pilot")
    orientation = samples_by_channel["T2_axial"][0]["orientation"]
    patches, labels, records = [], [], []
    for finding_key in keys:
        fid, point = finding_key
        world = physical_grid(point, orientation)
        channel_patches, bounds = [], {}
        t2_finding = findings["T2_axial"][finding_key]
        clinical_label = t2_finding["ClinSig"].lower()
        if clinical_label not in {"true", "false"}:
            raise ValueError("Unknown clinical label")
        for name in CHANNELS:
            if findings[name][finding_key]["ClinSig"].lower() != clinical_label:
                raise ValueError("Label mismatch")
            patch, bounds[name] = sample_grid(volumes[name],
                np.asarray(reports[name]["dicom_affine_ijk_to_patient"]), world)
            channel_patches.append(patch)
        patches.append(np.stack(channel_patches))
        labels.append(int(clinical_label == "true"))
        records.append({"patient": "ProstateX-0002", "fid": fid, "pos_mm": list(point),
                        "ClinSig": clinical_label, "bounds": bounds,
                        "first_output_pixel_center_mm": world[:, 0, 0].tolist(),
                        "nearest_source_slices": {name: findings[name][finding_key]["nearest_slice_zero_based"] for name in CHANNELS}})
    patches = np.stack(patches).astype(np.float32)
    metadata = {
        "scope": "Two findings from one patient; geometry pilot only; no training or registration",
        "array_axes": ["finding", "channel", "row", "column"], "channels": list(CHANNELS),
        "shape": list(patches.shape), "field_of_view_edge_to_edge_mm": [50, 50],
        "output_spacing_mm": [0.5, 0.5], "pixel_center_offsets_mm": [-24.75, 24.75],
        "grid_plane": "Through recorded pos, parallel to T2 image plane; same world grid for both channels",
        "column_direction": orientation[:3].tolist(), "row_direction": orientation[3:].tolist(),
        "interpolation": "Trilinear, scipy.ndimage.map_coordinates order=1",
        "intensities": "Stored values, no normalization; ADC physical units not verified",
        "boundary_policy": "Reject any point outside source pixel centers; no padding",
        "label_encoding": {"ClinSig_FALSE": 0, "ClinSig_TRUE": 1},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "findings": records, "source_audits": reports,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "patches.npz", patches=patches,
                        labels=np.asarray(labels, dtype=np.uint8),
                        patient_ids=np.array([record["patient"] for record in records]),
                        finding_ids=np.array([record["fid"] for record in records]),
                        pos_mm=np.array([record["pos_mm"] for record in records]))
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    figure, axes = plt.subplots(len(keys), 4, figsize=(13, 9), squeeze=False)
    for row_index, finding_key in enumerate(keys):
        for channel_index, name in enumerate(CHANNELS):
            finding = findings[name][finding_key]
            original = volumes[name][finding["nearest_slice_zero_based"]]
            low, high = np.percentile(original, [0, 99.5])
            original_axis, patch_axis = axes[row_index, channel_index * 2:channel_index * 2 + 2]
            original_axis.imshow(original, cmap="gray", vmin=low, vmax=high)
            original_axis.scatter(*finding["dicom_continuous_ijk"][:2], marker="+", color="gold")
            original_axis.set_title(f"{name}: nearest native slice")
            patch_axis.imshow(patches[row_index, channel_index], cmap="gray", vmin=low, vmax=high,
                              extent=(-25, 25, 25, -25), interpolation="nearest")
            patch_axis.scatter(0, 0, marker="+", color="gold")
            patch_axis.set_title(f"fid={finding_key[0]}, ClinSig={records[row_index]['ClinSig']}")
            patch_axis.set_xlabel("mm from recorded point")
    figure.suptitle("QC pilot: native slices and 50 x 50 mm resampled patches\n"
                   "Stored intensities; independent channel windows; no anatomical registration")
    figure.tight_layout(rect=(0, 0, 1, .9), h_pad=2.5)
    figure.savefig(output_dir / "preview.pdf")
    print("Patient: ProstateX-0002")
    print("Patch shape:", patches.shape, "[finding, channel, row, column]")
    print("Channels:", CHANNELS, "Labels:", labels)
    print("Out-of-bounds points: 0; no padding; no normalization; no training")
    print("Saved:", output_dir)
    return metadata, figure