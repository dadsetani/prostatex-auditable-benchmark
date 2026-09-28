"""Extract one frozen PROSTATEx outer fold into paired T2/ADC patches."""

import argparse
import csv
import hashlib
import io
import json
import platform
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.ndimage import map_coordinates


REGISTER_SHA256 = "af602758925b35388a29322a17366dc3057729efcec824221456c3073ca06891"
PROTOCOL_SHA256 = "45a423324fab7f4e2971743ce2f5fd5961b7976e9745bb7567366bf89a889b08"
BOUND_TOLERANCE_VOXELS = 1e-4
PATCH_SIZE = 100
PATCH_SPACING_MM = 0.5
CHANNELS = ("T2_axial", "ADC")
EXPECTED_FINDINGS = {0: 64, 1: 57, 2: 62, 3: 62, 4: 61}
FOLD_MANIFESTS = {
    0: "fold0_manifest.json",
    1: "fold1_final_manifest.json",
    2: "fold2_manifest.json",
    3: "fold3_manifest.json",
    4: "fold4_manifest.json",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_json_sha256(path):
    raw = (json.dumps(read_json(path), indent=2, allow_nan=False) + "\n").encode("utf-8")
    return sha256_bytes(raw)


def verified_fold_bundle(upload_dir, manifest_name, temporary_dir):
    manifest_path = Path(upload_dir) / manifest_name
    manifest = read_json(manifest_path)
    output = Path(temporary_dir) / manifest["original_filename"]
    with output.open("wb") as destination:
        for item in manifest["parts"]:
            part = Path(upload_dir) / item["name"]
            require(part.stat().st_size == item["bytes"], f"Unexpected part size: {part}")
            require(sha256_path(part) == item["sha256"], f"Unexpected part hash: {part}")
            with part.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    destination.write(block)
    require(output.stat().st_size == manifest["original_bytes"], "Unexpected merged bundle size")
    require(sha256_path(output) == manifest["original_sha256"], "Unexpected merged bundle hash")
    return output, manifest


def verified_existing_bundle(bundle_root, source):
    bundle_root = Path(bundle_root)
    matches = list(bundle_root.rglob(source["bundle_filename"]))
    require(len(matches) == 1,
            f"Expected one merged bundle named {source['bundle_filename']}; found {len(matches)}")
    bundle = matches[0]
    require(bundle.stat().st_size == source["bundle_bytes"], "Unexpected merged bundle size")
    require(sha256_path(bundle) == source["bundle_sha256"], "Unexpected merged bundle hash")
    return bundle, {
        "original_filename": source["bundle_filename"],
        "original_bytes": source["bundle_bytes"],
        "original_sha256": source["bundle_sha256"],
    }


def open_nested_zip(raw):
    archive = ZipFile(io.BytesIO(raw))
    names = archive.namelist()
    require(len(names) == len(set(names)), "Duplicate nested ZIP member")
    require(archive.testzip() is None, "Nested ZIP CRC failure")
    return archive


def decode_native_explicit_le_plane(dicom, expected):
    require(len(dicom) >= 132 and dicom[128:132] == b"DICM", "Missing DICOM preamble")
    require(expected["transfer_syntax"] == "1.2.840.10008.1.2.1",
            "Only audited Explicit-VR Little-Endian input is supported")
    require(expected["bits_allocated"] == 16 and expected["pixel_representation"] == 0,
            "Only audited unsigned 16-bit stored pixels are supported")
    expected_bytes = expected["rows"] * expected["columns"] * 2
    tag = b"\xe0\x7f\x10\x00"
    candidates = []
    start = 132
    while True:
        offset = dicom.find(tag, start)
        if offset < 0:
            break
        if offset + 12 <= len(dicom) and dicom[offset + 4:offset + 6] in {b"OB", b"OW"}:
            length = int.from_bytes(dicom[offset + 8:offset + 12], "little")
            if length == expected_bytes and offset + 12 + length <= len(dicom):
                candidates.append(offset)
        start = offset + 1
    require(len(candidates) == 1, "Expected one native PixelData element")
    offset = candidates[0] + 12
    plane = np.frombuffer(dicom[offset:offset + expected_bytes], dtype="<u2").reshape(
        expected["rows"], expected["columns"]
    )
    require(int(plane.max()) < 2 ** expected["bits_stored"], "Pixels exceed stored-bit range")
    return plane


def decode_volume(raw_zip, series):
    geometry = series["geometry"]
    require(series.get("audit_status", "PASSED") == "PASSED", "Series audit did not pass")
    require(not geometry["semantic_flags"], "Intensity-semantics flag requires review")
    sorted_instances = geometry["sorted_instances"]
    planes = []
    with open_nested_zip(raw_zip) as archive:
        require(len(sorted_instances) == geometry["shape_zyx"][0], "Unexpected sorted slice count")
        for expected in sorted_instances:
            dicom = archive.read(expected["filename"])
            require(sha256_bytes(dicom) == expected["dicom_sha256"], "DICOM hash changed")
            plane = decode_native_explicit_le_plane(dicom, expected)
            require(list(plane.shape) == geometry["shape_zyx"][1:], "Decoded plane shape changed")
            planes.append(plane)
    volume = np.stack(planes).astype(np.float32, copy=False)
    require(list(volume.shape) == geometry["shape_zyx"], "Decoded volume shape changed")
    require(np.isfinite(volume).all(), "Nonfinite stored pixels")
    return volume


def physical_grid(point, t2_geometry):
    offsets = (np.arange(PATCH_SIZE, dtype=np.float64) - (PATCH_SIZE - 1) / 2) * PATCH_SPACING_MM
    column_offsets, row_offsets = np.meshgrid(offsets, offsets, indexing="xy")
    column = np.asarray(t2_geometry["column_direction"], dtype=np.float64)
    row = np.asarray(t2_geometry["row_direction"], dtype=np.float64)
    world = (np.asarray(point, dtype=np.float64)[:, None, None]
             + column[:, None, None] * column_offsets
             + row[:, None, None] * row_offsets)
    return world


def sample_patch(volume, geometry, world):
    homogeneous = np.vstack((world.reshape(3, -1), np.ones(PATCH_SIZE * PATCH_SIZE)))
    source_ijk = np.linalg.solve(
        np.asarray(geometry["affine_ijk_to_patient"], dtype=np.float64), homogeneous
    )[:3]
    limits = np.asarray(geometry["shape_zyx"][::-1], dtype=np.float64) - 1
    outside = np.any(
        (source_ijk < -BOUND_TOLERANCE_VOXELS)
        | (source_ijk > limits[:, None] + BOUND_TOLERANCE_VOXELS), axis=0
    )
    require(not outside.any(), "Accepted finding failed independent complete-grid bounds check")
    snapped = np.clip(source_ijk, 0, limits[:, None])
    maximum_snap = float(np.max(np.abs(snapped - source_ijk)))
    require(maximum_snap <= BOUND_TOLERANCE_VOXELS, "Numerical boundary snap exceeded tolerance")
    values = map_coordinates(
        volume,
        [snapped[2], snapped[1], snapped[0]],
        order=1,
        mode="nearest",
        prefilter=False,
    ).reshape(PATCH_SIZE, PATCH_SIZE)
    require(np.isfinite(values).all(), "Nonfinite resampled patch")
    return values.astype(np.float32, copy=False), {
        "min_source_ijk": source_ijk.min(axis=1).tolist(),
        "max_source_ijk": source_ijk.max(axis=1).tolist(),
        "limits_ijk": limits.tolist(),
        "maximum_boundary_snap_voxels": maximum_snap,
    }


def write_series_csv(path, rows):
    fields = [
        "patient", "study_uid", "channel", "series_uid", "description",
        "archive_name", "archive_sha256", "archive_bytes",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def extract_fold(outer_fold, upload_dir="prism-uploads", analysis_dir="analysis",
                 output_root="analysis", bundle_root=None):
    import scipy

    outer_fold = int(outer_fold)
    require(outer_fold in FOLD_MANIFESTS, "Fold must be 0 through 4")
    upload_dir = Path(upload_dir)
    analysis_dir = Path(analysis_dir)
    output_root = Path(output_root)
    register_path = analysis_dir / "prostatex_eligibility_register_v0_3.json"
    require(canonical_json_sha256(register_path) == REGISTER_SHA256,
            "Unexpected eligibility-register content hash")
    register = read_json(register_path)
    require(register["protocol_sha256"] == PROTOCOL_SHA256, "Unexpected protocol hash in register")
    findings = [row for row in register["rows"]
                if row["eligibility"] == "ACCEPTED" and int(row["outer_fold"]) == outer_fold]
    require(len(findings) == EXPECTED_FINDINGS[outer_fold], "Unexpected accepted fold size")
    findings.sort(key=lambda row: (row["patient"], int(row["fid"]), tuple(row["pos_mm"])))

    output = output_root / f"prostatex_patches_fold_{outer_fold}_v0_3"
    bundle_output = output.with_suffix(".zip")
    require(not output.exists() and not bundle_output.exists(), "Patch output already exists")
    output.mkdir(parents=True)

    source = next(row for row in register["bundle_sources"] if row["outer_fold"] == outer_fold)
    with tempfile.TemporaryDirectory(prefix=f"prostatex-patches-{outer_fold}-") as temporary_dir:
        if bundle_root is None:
            bundle, bundle_manifest = verified_fold_bundle(
                upload_dir, FOLD_MANIFESTS[outer_fold], temporary_dir
            )
        else:
            bundle, bundle_manifest = verified_existing_bundle(bundle_root, source)
        with ZipFile(bundle) as outer:
            require(outer.testzip() is None, "Outer ZIP CRC failure")
            names = outer.namelist()
            require(len(names) == len(set(names)), "Duplicate outer ZIP member")
            report = json.loads(outer.read("report.json"))
            require(int(report["outer_fold"]) == outer_fold, "Report fold changed")
            report_findings = {row["full_key"]: row for row in report["findings"]}
            series_by_uid = {row["series_uid"]: row for row in report["series"]}

            required_series_uids = sorted({
                uid for row in findings for uid in (row["t2_series_uid"], row["adc_series_uid"])
            })
            require(all(required_series_uids), "Accepted row lacks a selected series")
            volumes = {}
            used_series = []
            for uid in required_series_uids:
                require(uid in series_by_uid, f"Selected series absent from report: {uid}")
                series = series_by_uid[uid]
                raw_zip = outer.read(series["archive_name"])
                require(len(raw_zip) == series["archive_bytes"], "Nested series size changed")
                require(sha256_bytes(raw_zip) == series["archive_sha256"], "Nested series hash changed")
                volumes[uid] = decode_volume(raw_zip, series)
                used_series.append(series)

            patches = []
            records = []
            for array_index, finding in enumerate(findings):
                source = report_findings[finding["full_key"]]
                require(source["patient"] == finding["patient"], "Finding patient changed")
                require(source["ClinSig"].upper() == finding["ClinSig"], "Finding label changed")
                require(source["pos_mm"] == finding["pos_mm"], "Finding position changed")
                t2_series = series_by_uid[finding["t2_series_uid"]]
                adc_series = series_by_uid[finding["adc_series_uid"]]
                require(t2_series["study_uid"] == finding["assigned_study_uid"], "T2 study changed")
                require(adc_series["study_uid"] == finding["assigned_study_uid"], "ADC study changed")
                require(t2_series["geometry"]["identities"]["FrameOfReferenceUID"]
                        == adc_series["geometry"]["identities"]["FrameOfReferenceUID"],
                        "Cross-channel frame changed")
                world = physical_grid(finding["pos_mm"], t2_series["geometry"])
                paired = []
                bounds = {}
                for channel, series in (("T2_axial", t2_series), ("ADC", adc_series)):
                    patch, bounds[channel] = sample_patch(
                        volumes[series["series_uid"]], series["geometry"], world
                    )
                    paired.append(patch)
                patches.append(np.stack(paired))
                records.append({
                    "array_index": array_index,
                    "full_key": finding["full_key"],
                    "patient": finding["patient"],
                    "fid": finding["fid"],
                    "pos_mm": finding["pos_mm"],
                    "zone": finding["zone"],
                    "ClinSig": finding["ClinSig"],
                    "label": finding["label"],
                    "outer_fold": outer_fold,
                    "assigned_study_uid": finding["assigned_study_uid"],
                    "series_uids": {
                        "T2_axial": finding["t2_series_uid"],
                        "ADC": finding["adc_series_uid"],
                    },
                    "bounds": bounds,
                    "first_output_pixel_center_mm": world[:, 0, 0].tolist(),
                })

    patch_array = np.stack(patches).astype(np.float32, copy=False)
    expected_shape = (EXPECTED_FINDINGS[outer_fold], 2, PATCH_SIZE, PATCH_SIZE)
    require(patch_array.shape == expected_shape, "Unexpected patch array shape")
    require(np.isfinite(patch_array).all(), "Nonfinite patch array")
    labels = np.asarray([row["label"] for row in records], dtype=np.uint8)
    require(set(labels.tolist()).issubset({0, 1}), "Unexpected label encoding")

    npz_path = output / "patches.npz"
    np.savez_compressed(
        npz_path,
        patches=patch_array,
        labels=labels,
        patient_ids=np.asarray([row["patient"] for row in records]),
        finding_ids=np.asarray([row["fid"] for row in records]),
        positions_mm=np.asarray([row["pos_mm"] for row in records], dtype=np.float64),
        full_keys=np.asarray([row["full_key"] for row in records]),
        outer_folds=np.full(len(records), outer_fold, dtype=np.uint8),
    )
    metadata = {
        "dataset_id": f"PROSTATEx-paired-patches-fold-{outer_fold}-v0.3",
        "created_date": "2026-09-27",
        "scope": "Frozen technically eligible paired patches; no normalization, image-quality review, modeling, or clinical validation",
        "protocol_sha256": PROTOCOL_SHA256,
        "eligibility_register_sha256": REGISTER_SHA256,
        "source_bundle": {
            "filename": bundle_manifest["original_filename"],
            "bytes": bundle_manifest["original_bytes"],
            "sha256": bundle_manifest["original_sha256"],
        },
        "shape": list(patch_array.shape),
        "dtype": str(patch_array.dtype),
        "array_axes": ["finding", "channel", "row", "column"],
        "channels": list(CHANNELS),
        "ordering": "patient, integer fid, lexicographic pos_mm",
        "label_encoding": {"FALSE": 0, "TRUE": 1},
        "output_spacing_mm": [PATCH_SPACING_MM, PATCH_SPACING_MM],
        "edge_to_edge_field_of_view_mm": [50.0, 50.0],
        "pixel_center_offsets_mm": [-24.75, 24.75],
        "grid": "Through recorded position, parallel to selected T2 plane; identical world grid sampled in both channels",
        "interpolation": "Trilinear scipy.ndimage.map_coordinates order=1",
        "boundary_policy": "Reject beyond 1e-4 voxel; snap only numerical overshoot within tolerance; no clinical padding or clipping",
        "intensities": "Stored pixel values converted to float32; no normalization, LUT, rescale, or ADC-unit conversion",
        "patch_array_sha256": sha256_bytes(patch_array.tobytes(order="C")),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "dicom_decoder": "Hash-verified native Explicit-VR Little-Endian PixelData reader",
        },
        "findings": records,
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    series_path = output / "source_series.csv"
    write_series_csv(series_path, sorted(used_series, key=lambda row: (
        row["patient"], row["study_uid"], CHANNELS.index(row["channel"])
    )))
    artifact_paths = (npz_path, metadata_path, series_path)
    artifact_hashes = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_path(path)}
        for path in artifact_paths
    }
    hashes_path = output / "artifact_hashes.json"
    hashes_path.write_text(json.dumps(artifact_hashes, indent=2) + "\n", encoding="utf-8")
    with ZipFile(bundle_output, "x", compression=ZIP_DEFLATED, allowZip64=True) as archive:
        for path in (*artifact_paths, hashes_path):
            archive.write(path, arcname=path.name)
    print("Fold:", outer_fold)
    print("Patch shape:", patch_array.shape)
    print("Labels:", {"TRUE": int(labels.sum()), "FALSE": int((labels == 0).sum())})
    print("Saved:", bundle_output)
    return bundle_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fold", type=int, choices=range(5))
    parser.add_argument("--upload-dir", default="prism-uploads")
    parser.add_argument("--analysis-dir", default="analysis")
    parser.add_argument("--output-root", default="analysis")
    parser.add_argument("--bundle-root")
    arguments = parser.parse_args()
    extract_fold(arguments.fold, arguments.upload_dir, arguments.analysis_dir,
                 arguments.output_root, arguments.bundle_root)