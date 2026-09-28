"""QC01: separate decoder and manual resampling; no download or training."""

import hashlib
import io
import itertools
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np

PATIENT = "ProstateX-0000"
POINT = np.array([25.7457, 31.8707, -38.511])
CHANNELS = ("T2_axial", "ADC")
PREFIX = "1.3.6.1.4.1.14519.5.2.1.7311.5101."
UIDS = dict(zip(CHANNELS, (PREFIX + "160028252338004527274326500702",
                           PREFIX + "339319789559896104041345048780")))
SOURCE_HASH = "1ad3836fd31e7a9d3c629d2811efd1f7f59d658ab9a8de6631a6a6a719aa7f0d"
PATCH_HASH = "980eda4a8eb3faed34c6e6b49f380e76dfca4287bf4bd02c60291d5f33474a5d"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def archive_bytes(raw):
    archive = ZipFile(io.BytesIO(raw))
    entries = archive.infolist()
    require(len(entries) == len({entry.filename for entry in entries}), "Duplicate ZIP members")
    require(sum(entry.file_size for entry in entries) <= 256 * 1024**2, "ZIP size limit")
    require(archive.testzip() is None, "ZIP CRC failure")
    return archive


def geometry_from_planes(orientations, spacings, positions):
    orientations = np.asarray(orientations, dtype=float)
    spacings = np.asarray(spacings, dtype=float)
    positions = np.asarray(positions, dtype=float)
    count = len(positions)
    require(count >= 2 and orientations.shape == (count, 6)
            and spacings.shape == (count, 2) and positions.shape == (count, 3), "Geometry shape")
    require(all(np.isfinite(values).all() for values in (orientations, spacings, positions)),
            "Nonfinite geometry")
    require(np.all(spacings > 0), "Nonpositive spacing")
    require(np.max(np.abs(orientations - orientations[0])) <= 1e-6, "Varying orientation")
    require(np.max(np.abs(spacings - spacings[0])) <= 1e-6, "Varying spacing")
    basis = orientations[0].reshape(2, 3).T
    require(np.max(np.abs(basis.T @ basis - np.eye(2))) <= 1e-6, "Nonorthonormal directions")
    normal = np.cross(basis[:, 0], basis[:, 1])
    normal /= np.linalg.norm(normal)
    order = np.argsort(positions @ normal)
    steps = np.diff(positions[order], axis=0)
    require(np.all(steps @ normal > 0), "Duplicate/reversed physical planes")
    step = steps.mean(axis=0)
    deviation = float(np.max(np.abs(steps - step)))
    require(deviation <= 1e-4, "Irregular slice-step vectors")
    affine = np.eye(4)
    affine[:3, :2] = basis * spacings[0, ::-1]
    affine[:3, 2] = step
    affine[:3, 3] = positions[order[0]]
    require(abs(np.linalg.det(affine[:3, :3])) > 1e-12, "Singular affine")
    return order, affine, basis, deviation


def decode_series(raw, channel, pydicom):
    planes, positions, orientations, spacings, details = [], [], [], [], []
    identities = {name: set() for name in ("PatientID", "StudyInstanceUID", "SeriesInstanceUID",
                                         "FrameOfReferenceUID")}
    sops, semantic_flags = set(), []
    watched = {"RescaleSlope", "RescaleIntercept", "RescaleType", "ModalityLUTSequence",
               "RealWorldValueMappingSequence", "PixelValueTransformationSequence",
               "PixelPaddingValue", "PixelPaddingRangeLimit"}
    with archive_bytes(raw) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".dcm")]
        require(len(names) == 19, "Expected 19 DICOMs per QC01 series")
        for name in names:
            dicom = archive.read(name)
            dataset = pydicom.dcmread(io.BytesIO(dicom), force=False)
            require(str(dataset.file_meta.TransferSyntaxUID) == "1.2.840.10008.1.2.1",
                    "This check supports native Explicit VR Little Endian only")
            require(str(dataset.SOPClassUID) == "1.2.840.10008.5.1.4.1.1.4"
                    and str(dataset.Modality) == "MR", "Expected classic MR")
            require(str(dataset.file_meta.MediaStorageSOPClassUID) == str(dataset.SOPClassUID),
                    "File-meta SOP class mismatch")
            sop = str(dataset.SOPInstanceUID)
            require(sop and sop not in sops and str(dataset.file_meta.MediaStorageSOPInstanceUID) == sop,
                    "Duplicate or file-meta SOP instance mismatch")
            sops.add(sop)
            for field in identities:
                value = str(getattr(dataset, field, ""))
                require(bool(value), "Missing identity: " + field)
                identities[field].add(value)
            require(int(dataset.BitsAllocated) == 16 and int(dataset.PixelRepresentation) == 0
                    and int(dataset.SamplesPerPixel) == 1 and int(getattr(dataset, "NumberOfFrames", 1)) == 1
                    and str(dataset.PhotometricInterpretation) == "MONOCHROME2", "Unsupported pixel format")
            bits = int(dataset.BitsStored)
            require(1 <= bits <= 16 and int(dataset.HighBit) == bits - 1, "Stored-bit layout")
            shape = (int(dataset.Rows), int(dataset.Columns))
            require(min(shape) > 0 and len(dataset.PixelData) == 2 * shape[0] * shape[1], "Pixel byte length")
            manual = np.frombuffer(dataset.PixelData, dtype="<u2").reshape(shape)
            require(int(manual.max()) < 2**bits, "Nonzero unused high bits require review")
            decoded = np.asarray(dataset.pixel_array)
            require(decoded.shape == shape and np.array_equal(decoded, manual), "Decoder disagreement")
            require(not planes or decoded.shape == planes[0].shape, "Varying dimensions")
            flags = [{"keyword": element.keyword, "tag": str(element.tag),
                      "value": "SEQUENCE_PRESENT" if element.VR == "SQ" else str(element.value)}
                     for element in dataset.iterall() if element.keyword in watched]
            semantic_flags.extend({"filename": name, **flag} for flag in flags)
            details.append({"filename": name, "sop_uid": sop, "dicom_sha256": sha(dicom),
                            "stored_pixels_sha256": sha(decoded.astype("<u2").tobytes()),
                            "zero_count": int(np.count_nonzero(decoded == 0)), "bits_stored": bits,
                            "raw_decode_equal": True})
            planes.append(decoded.copy())
            positions.append(dataset.ImagePositionPatient)
            orientations.append(dataset.ImageOrientationPatient)
            spacings.append(dataset.PixelSpacing)
    require(all(len(values) == 1 for values in identities.values()), "Inconsistent identities")
    require(identities["PatientID"] == {PATIENT} and identities["SeriesInstanceUID"] == {UIDS[channel]},
            "Unexpected patient or series")
    order, affine, basis, deviation = geometry_from_planes(orientations, spacings, positions)
    volume = np.stack([planes[index] for index in order])
    return volume, affine, basis, {
        "identities": {name: next(iter(values)) for name, values in identities.items()},
        "affine_ijk_to_patient": affine.tolist(), "shape_zyx": list(volume.shape),
        "max_step_component_deviation_mm": deviation, "semantic_flags": semantic_flags,
        "sorted_instances": [details[index] for index in order],
        "supported_format": "Classic MR; explicit little endian; unsigned native 16-bit MONOCHROME2",
    }


def interpolate(volume, coordinates):
    require(coordinates.ndim == 2 and coordinates.shape[0] == 3
            and np.isfinite(coordinates).all(), "Invalid sampling coordinates")
    limits = np.asarray(volume.shape[::-1])[:, None] - 1
    require(np.all((coordinates >= 0) & (coordinates <= limits)), "Out-of-bounds sampling")
    lower = np.floor(coordinates).astype(int)
    upper = np.ceil(coordinates).astype(int)
    fraction = coordinates - lower
    values = np.zeros(coordinates.shape[1], dtype=float)
    all_active_zero = np.ones(coordinates.shape[1], dtype=bool)
    for corner in itertools.product((0, 1), repeat=3):
        indices = [upper[axis] if corner[axis] else lower[axis] for axis in range(3)]
        weight = np.prod([fraction[axis] if corner[axis] else 1 - fraction[axis]
                          for axis in range(3)], axis=0)
        native = volume[indices[2], indices[1], indices[0]]
        values += weight * native
        all_active_zero &= (weight == 0) | (native == 0)
    return values, all_active_zero


def run_qc01(source_path, patch_path):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = Path("qc01_pydicom_" + stamp)
    output.mkdir(exist_ok=False)
    code_path = Path(__file__)
    report = {"status": "INCOMPLETE", "created_utc": datetime.now(timezone.utc).isoformat(),
              "code_sha256": sha(code_path.read_bytes()),
              "scope": "Separate pydicom decoder and manual resampler for QC01 only; not blinded external review",
              "software": {"python": platform.python_version(), "numpy": np.__version__},
              "limits": ["No clinical or anatomical validation", "No ADC physical-unit claim",
                         "No official label verification", "No changes to human QC or source files"],
              "tolerances": {"pixel_atol": 0.001, "pixel_rtol": 0, "coordinate_atol_voxels": 1e-5,
                             "orientation_spacing_atol": 1e-6, "slice_step_atol_mm": 1e-4},
              "pixel_tolerance_basis": "Fixed before comparison; unsigned16 magnitude <=65535 gives half a float32 ULP <=0.001953125. The stricter 0.001 threshold may flag harmless rounding; mismatches require review, not tolerance tuning."}
    try:
        import pydicom
        report["software"]["pydicom"] = pydicom.__version__
        source_raw, patch_raw = Path(source_path).read_bytes(), Path(patch_path).read_bytes()
        report["source_sha256"], report["patch_bundle_sha256"] = sha(source_raw), sha(patch_raw)
        require(sha(source_raw) == SOURCE_HASH and sha(patch_raw) == PATCH_HASH, "Unexpected source bundle hash")
        with archive_bytes(patch_raw) as archive:
            required = {"metadata.json", "geometry.json", "patches.npz", "preview.pdf"}
            require(set(archive.namelist()) == required | {"artifact_hashes.json"}, "Patch archive members")
            hashes = json.loads(archive.read("artifact_hashes.json"))
            require(set(hashes) == required, "Incomplete patch manifest")
            for name, expected_hash in hashes.items():
                require(sha(archive.read(name)) == expected_hash, "Artifact hash: " + name)
            metadata = json.loads(archive.read("metadata.json"))
            require(metadata["channels"] == list(CHANNELS)
                    and metadata["array_axes"] == ["finding", "channel", "row", "column"]
                    and metadata["output_spacing_mm"] == [0.5, 0.5], "Reference channel/grid specification")
            require(metadata["bundle_sha256"] == SOURCE_HASH, "Reference source mismatch")
            with np.load(io.BytesIO(archive.read("patches.npz")), allow_pickle=False) as arrays:
                patches = arrays["patches"]
                require(patches.shape == (7, 2, 100, 100) and patches.dtype == np.float32
                        and np.isfinite(patches).all(), "Reference patch shape/dtype/finiteness")
                require(sha(patches.tobytes()) == metadata["patch_array_sha256"], "Reference array hash")
                candidates = [index for index, row in enumerate(metadata["findings"])
                              if row["patient"] == PATIENT and row["fid"] == "1"
                              and np.array_equal(row["pos_mm"], POINT)]
                require(len(candidates) == 1, "Ambiguous full finding key")
                index = candidates[0]
                finding = metadata["findings"][index]
                require(finding["array_index"] == index and arrays["patient_ids"][index] == PATIENT
                        and arrays["finding_ids"][index] == "1"
                        and np.array_equal(arrays["pos_mm"][index], POINT), "Reference identity mismatch")
                expected = patches[index].copy()
        series, report["series"] = {}, {}
        with archive_bytes(source_raw) as archive:
            records = json.loads(archive.read("download_report.json"))["series"]
            for channel in CHANNELS:
                matches = [row for row in records if row["patient"] == PATIENT
                           and row["series_uid"] == UIDS[channel] and row["channel"] == channel]
                require(len(matches) == 1, "Ambiguous source series")
                row = matches[0]
                raw = archive.read(row["file"])
                require(sha(raw) == row["zip_sha256"], "Inner archive hash mismatch")
                volume, affine, basis, details = decode_series(raw, channel, pydicom)
                series[channel] = (volume, affine, basis)
                report["series"][channel] = details
        for field in ("StudyInstanceUID", "FrameOfReferenceUID"):
            require(len({item["identities"][field] for item in report["series"].values()}) == 1,
                    "Cross-series " + field + " mismatch")
        offsets = (np.arange(100) - 49.5) * 0.5
        column_offsets, row_offsets = np.meshgrid(offsets, offsets, indexing="xy")
        basis = series["T2_axial"][2]
        world = POINT[:, None, None] + basis[:, 0, None, None] * column_offsets + basis[:, 1, None, None] * row_offsets
        homogeneous = np.vstack((world.reshape(3, -1), np.ones(10000)))
        comparisons, reconstructed = {}, []
        for index, channel in enumerate(CHANNELS):
            volume, affine, basis = series[channel]
            coordinates = np.linalg.solve(affine, homogeneous)[:3]
            values, source_zero = interpolate(volume, coordinates)
            patch = values.reshape(100, 100).astype(np.float32)
            difference = np.abs(values.reshape(100, 100) - expected[index])
            bound = finding["bounds"][channel]
            coordinate_error = max(float(np.max(np.abs(coordinates.min(axis=1) - bound["min_source_ijk"]))),
                                   float(np.max(np.abs(coordinates.max(axis=1) - bound["max_source_ijk"]))))
            zero_mismatches = int(np.count_nonzero((patch == 0) != (expected[index] == 0)))
            unsupported = int(np.count_nonzero((patch.ravel() == 0) != source_zero))
            passed = (float(difference.max()) <= 0.001 and coordinate_error <= 1e-5
                      and zero_mismatches == 0 and unsupported == 0 and bound["outside_points"] == 0)
            comparisons[channel] = {"passed": passed, "max_abs_difference_float64_vs_reference": float(difference.max()),
                "mean_abs_difference": float(difference.mean()), "float32_differing_pixels": int(np.count_nonzero(patch != expected[index])),
                "max_claimed_bound_difference_voxels": coordinate_error, "outside_points": 0,
                "zero_mask_differing_pixels": zero_mismatches, "zero_trace_mismatches": unsupported,
                "native_volume_zero_percent": float(np.mean(volume == 0) * 100),
                "patch_zero_percent": float(np.mean(patch == 0) * 100)}
            reconstructed.append(patch)
        report["comparison"] = comparisons
        flagged = any(item["semantic_flags"] for item in report["series"].values())
        numerical_match = all(item["passed"] for item in comparisons.values())
        report["numeric_match"] = numerical_match
        report["status"] = ("DISCREPANCY_FOUND" if not numerical_match else
                            "REVIEW_REQUIRED" if flagged else "TECHNICAL_MATCH")
        report["semantic_review_required"] = flagged
        report["target"] = {"patient": PATIENT, "fid": "1", "pos_mm": POINT.tolist()}
        np.savez_compressed(output / "reconstructed.npz", patches=np.stack(reconstructed)[None])
        report["reconstructed_npz_sha256"] = sha((output / "reconstructed.npz").read_bytes())
    except Exception as error:
        report["error"] = type(error).__name__ + ": " + str(error)
    report_path = output / "report.json"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Status:", report["status"])
    if "error" in report:
        print("Error:", report["error"])
    for channel, result in report.get("comparison", {}).items():
        print(channel, json.dumps(result))
    print("Report:", report_path)
    return report_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("source_bundle")
    parser.add_argument("patch_bundle")
    arguments = parser.parse_args()
    run_qc01(arguments.source_bundle, arguments.patch_bundle)