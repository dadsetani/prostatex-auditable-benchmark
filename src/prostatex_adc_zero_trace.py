"""Trace ADC patch zeros to native stored values; no clinical decisions."""

import hashlib
import io
import itertools
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import numpy as np
import scipy
from scipy.ndimage import map_coordinates


def trace_corners(volume, indices):
    limits = np.asarray(volume.shape[::-1])[:, None] - 1
    if not np.isfinite(indices).all() or np.any((indices < 0) | (indices > limits)):
        raise ValueError("Invalid or out-of-bounds source coordinates")
    lower = np.floor(indices).astype(int)
    upper = np.ceil(indices).astype(int)
    fraction = indices - lower
    reconstructed = np.zeros(indices.shape[1])
    zero_weight = np.zeros(indices.shape[1])
    all_active_corners_zero = np.ones(indices.shape[1], dtype=bool)
    for choices in itertools.product((0, 1), repeat=3):
        coordinates = np.stack([
            upper[axis] if choice else lower[axis]
            for axis, choice in enumerate(choices)
        ])
        weight = np.prod([
            fraction[axis] if choice else 1 - fraction[axis]
            for axis, choice in enumerate(choices)
        ], axis=0)
        values = volume[tuple(coordinates[::-1])]
        reconstructed += weight * values
        zero_weight += weight * (values == 0)
        all_active_corners_zero &= (weight == 0) | (values == 0)
    return reconstructed, zero_weight, all_active_corners_zero


def trace_adc_zeros(patch_bundle, source_bundle, label_zip, output_dir):
    patch_bundle, source_bundle, label_zip = map(Path, (patch_bundle, source_bundle, label_zip))
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.with_suffix(".zip").exists():
        raise FileExistsError("Choose a new output directory")
    with ZipFile(patch_bundle) as archive:
        names = archive.namelist()
        expected = {"patches.npz", "metadata.json", "geometry.json", "preview.pdf", "artifact_hashes.json"}
        if len(names) != 5 or set(names) != expected or archive.testzip() is not None:
            raise ValueError("Invalid patch bundle")
        hashes = json.loads(archive.read("artifact_hashes.json"))
        if set(hashes) != expected - {"artifact_hashes.json"}:
            raise ValueError("Incomplete artifact hash manifest")
        for name, digest in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"Artifact hash mismatch: {name}")
        metadata = json.loads(archive.read("metadata.json"))
        geometry = json.loads(archive.read("geometry.json"))
        with np.load(io.BytesIO(archive.read("patches.npz")), allow_pickle=False) as loaded:
            arrays = {name: loaded[name].copy() for name in loaded.files}
    patches = arrays["patches"]
    if patches.shape != (7, 2, 100, 100) or patches.dtype != np.float32 or not np.isfinite(patches).all():
        raise ValueError("Unexpected patch array")
    if metadata["channels"] != ["T2_axial", "ADC"] or metadata["output_spacing_mm"] != [.5, .5]:
        raise ValueError("Unexpected channels or sampling grid")
    if metadata["bundle_sha256"] != hashlib.sha256(source_bundle.read_bytes()).hexdigest():
        raise ValueError("Source bundle hash mismatch")
    if metadata["patch_array_sha256"] != hashlib.sha256(patches.tobytes(order="C")).hexdigest():
        raise ValueError("Patch array hash mismatch")
    with ZipFile(label_zip) as archive:
        for basename, digest in metadata["label_csv_sha256"].items():
            names = [name for name in archive.namelist() if Path(name).name == basename]
            if len(names) != 1 or hashlib.sha256(archive.read(names[0])).hexdigest() != digest:
                raise ValueError("Label CSV hash mismatch")
    offsets = (np.arange(100) - 49.5) * .5
    columns, rows = np.meshgrid(offsets, offsets, indexing="xy")
    distance = np.hypot(columns, rows)
    results, views, cache, keys = [], [], {}, set()
    if len(metadata["findings"]) != 7:
        raise ValueError("Expected seven findings")
    with ZipFile(source_bundle) as outer:
        if outer.testzip() is not None or len(outer.namelist()) != len(set(outer.namelist())):
            raise ValueError("Invalid source archive")
        downloads = json.loads(outer.read("download_report.json"))["series"]
        for index, finding in enumerate(metadata["findings"]):
            patient, fid, point = finding["patient"], finding["fid"], finding["pos_mm"]
            key = (patient, fid, tuple(point))
            if key in keys or finding["array_index"] != index:
                raise ValueError("Duplicate key or array index mismatch")
            keys.add(key)
            if arrays["patient_ids"][index] != patient or arrays["finding_ids"][index] != fid:
                raise ValueError("NPZ identity mismatch")
            np.testing.assert_array_equal(arrays["pos_mm"][index], point)
            reference = geometry["series_by_patient"][patient]["ADC"]
            if patient not in cache:
                matching = [row for row in downloads if row["series_uid"] == reference["series_uid"]]
                if len(matching) != 1:
                    raise ValueError("Missing or ambiguous ADC source")
                raw = outer.read(matching[0]["file"])
                if hashlib.sha256(raw).hexdigest() != reference["image_zip_sha256"]:
                    raise ValueError("ADC archive hash mismatch")
                fresh, samples = audit(GeometryMemoryZip(raw), reference["series_uid"], label_zip)
                if {key: value for key, value in fresh.items() if key != "labels_zip_sha256"} != {key: value for key, value in reference.items() if key != "labels_zip_sha256"}:
                    raise ValueError("ADC source geometry mismatch")
                cache[patient] = (stored_volume(samples), samples)
            volume, samples = cache[patient]
            matched = [row for row in reference["findings"] if row["fid"] == fid and row["pos_mm"] == point]
            if len(matched) != 1 or int(matched[0]["ClinSig"] == "TRUE") != arrays["labels"][index]:
                raise ValueError("Finding/label mismatch")
            orientation = np.r_[finding["column_direction"], finding["row_direction"]]
            world = physical_grid(point, orientation, size=100, spacing_mm=.5)
            affine = np.asarray(reference["dicom_affine_ijk_to_patient"])
            recomputed, bounds = sample_grid(volume, affine, world)
            np.testing.assert_array_equal(recomputed, patches[index, 1])
            indices = np.linalg.solve(affine, np.vstack((world.reshape(3, -1), np.ones((1, 10000)))))[:3]
            reconstructed, zero_weight, source_zero = trace_corners(volume, indices)
            np.testing.assert_allclose(reconstructed, recomputed.ravel(), rtol=0, atol=1e-3)
            zero_mask = recomputed == 0
            if not np.array_equal(source_zero.reshape(100, 100), zero_mask):
                raise ValueError("Patch zero mask differs from active native-corner zero mask")
            center = np.linalg.solve(affine, np.r_[point, 1.])[:3]
            trace_corners(volume, center[:, None])
            center_value = float(map_coordinates(volume.astype(np.float32), center[::-1, None], order=1, prefilter=False)[0])
            lower, upper = int(np.floor(center[2])), int(np.ceil(center[2]))
            result = {
                "review_id": f"QC{index + 1:02d}", "patient": patient, "fid": fid, "pos_mm": point,
                "adc_zero_percent": float(zero_mask.mean() * 100),
                "central_10mm_zero_percent": float(zero_mask[40:60, 40:60].mean() * 100),
                "central_20mm_zero_percent": float(zero_mask[30:70, 30:70].mean() * 100),
                "nearest_zero_pixel_center_distance_mm": float(distance[zero_mask].min()) if zero_mask.any() else None,
                "center_interpolated_stored_value": center_value,
                "native_center_ijk": center.tolist(), "native_center_bracketing_slices": [lower, upper],
                "native_center_bracketing_filenames": [samples[lower]["filename"], samples[upper]["filename"]],
                "native_bracketing_slice_zero_percent": [float(np.mean(volume[plane] == 0) * 100) for plane in (lower, upper)],
                "all_patch_zeros_explained_by_active_native_zero_corners": True,
                "mixed_zero_nonzero_contribution_percent": float(np.mean((zero_weight > 0) & ~source_zero) * 100),
                "manual_interpolation_max_abs_difference": float(np.abs(reconstructed - recomputed.ravel()).max()),
                "patch_recomputed_exactly": True, "bounds": bounds,
                "human_qc_status": "UNCHANGED", "anatomical_alignment": "NOT_ASSESSED",
            }
            results.append(result)
            views.append((volume[lower], volume[upper], center, zero_mask, zero_weight.reshape(100, 100)))
    report = {
        "scope": "Technical zero-value provenance only; no clinical acceptance, rejection, or human review",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_bundle_sha256": metadata["bundle_sha256"],
        "patch_bundle_sha256": hashlib.sha256(patch_bundle.read_bytes()).hexdigest(),
        "patch_array_sha256": metadata["patch_array_sha256"], "label_csv_sha256": metadata["label_csv_sha256"],
        "definitions": {"zero": "exact stored/interpolated value == 0",
                        "central_regions": "10x10 and 20x20 mm squares centered on supplied point; descriptive only, not lesion masks",
                        "distance": "minimum in-plane distance to an output zero pixel center, not a native zero-region boundary",
                        "center_value": "trilinear estimate at supplied pos, not one of the even-sized patch's pixel centers",
                        "source_zero_weight": "sum of interpolation weights attached to zero-valued native voxels",
                        "manual_interpolation_tolerance": "rtol=0, atol=0.001 stored units; main SciPy recomputation must be exactly equal"},
        "limitations": ["Reason for native ADC zeros unknown", "No inference of healthy tissue, lesion extent, or anatomical registration", "No threshold-based filtering or normalization", "No training or completed human review"],
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "findings": results,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "zero_trace.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with PdfPages(output_dir / "zero_trace.pdf") as pdf:
        for index, result in enumerate(results):
            low_plane, high_plane, center, zero_mask, weights = views[index]
            figure, axes = plt.subplots(2, 3, figsize=(12, 8))
            try:
                for axis, image, title in zip(axes[0], (patches[index, 0], patches[index, 1], zero_mask), ("T2 patch", "ADC patch", "Exact-zero mask: white = zero")):
                    low, high = (0, 1) if image.dtype == bool else np.percentile(image, [0, 99.5])
                    axis.imshow(image, cmap="gray", vmin=low, vmax=high, extent=(-25, 25, 25, -25), interpolation="nearest")
                    axis.scatter(0, 0, marker="+", color="gold")
                    axis.set_title(title)
                for half_width, color in ((5, "red"), (10, "cyan")):
                    axes[0, 2].add_patch(Rectangle((-half_width, -half_width), 2 * half_width, 2 * half_width, fill=False, edgecolor=color))
                for axis, plane, number in zip(axes[1, :2], (low_plane, high_plane), result["native_center_bracketing_slices"]):
                    low, high = np.percentile(plane, [0, 99.5])
                    axis.imshow(plane, cmap="gray", vmin=low, vmax=high)
                    axis.scatter(*center[:2], marker="+", color="gold")
                    axis.set_title(f"Native ADC slice {number} (zero-based)")
                axes[1, 2].imshow(weights, cmap="gray", vmin=0, vmax=1, extent=(-25, 25, 25, -25))
                axes[1, 2].set_title("Weight from native zeros (black=0, white=1)")
                figure.suptitle(f"{result['review_id']} | {result['patient']} | fid={result['fid']}\n"
                               "Technical inspection only; center-bracketing planes; display windows independent")
                figure.tight_layout(rect=(0, 0, 1, .91))
                pdf.savefig(figure)
            finally:
                plt.close(figure)
    hashes = {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest() for name in ("zero_trace.json", "zero_trace.pdf")}
    (output_dir / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    bundle = output_dir.with_suffix(".zip")
    with ZipFile(bundle, "x", compression=ZIP_DEFLATED) as archive:
        for name in (*hashes, "artifact_hashes.json"):
            archive.write(output_dir / name, arcname=name)
    for row in results:
        print(row["review_id"], "zero %:", row["adc_zero_percent"], "central 10mm %:", row["central_10mm_zero_percent"],
              "central 20mm %:", row["central_20mm_zero_percent"], "nearest zero mm:", round(row["nearest_zero_pixel_center_distance_mm"], 2))
    print("Saved:", bundle, "No human QC status changed.")
    return report, bundle


if __name__ == "__main__":
    required = ("audit", "GeometryMemoryZip", "stored_volume", "physical_grid", "sample_grid", "patches_bundle", "remaining_bundle", "LABEL_ZIP")
    if any(name not in globals() for name in required):
        raise RuntimeError("Run the corrected pilot and Steps 5–7 first")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    zero_report, zero_bundle = trace_adc_zeros(patches_bundle, remaining_bundle, LABEL_ZIP, Path(f"adc_zero_trace_{stamp}"))
    from IPython.display import FileLink, display
    display(FileLink(str(zero_bundle)))