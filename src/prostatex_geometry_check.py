"""Audit one uploaded PROSTATEx series; not a general-purpose DICOM reader.

Only Explicit VR Little Endian, native monochrome 16-bit single-frame images
are accepted. No dependencies beyond NumPy are required. No files are extracted.
"""

import csv
import hashlib
import io
import json
from pathlib import Path
import struct
from zipfile import ZipFile

import numpy as np

SERIES_UID = "1.3.6.1.4.1.14519.5.2.1.7311.5101.255838192398023889769418952380"
IMAGE_ZIP = Path("prism-uploads") / f"{SERIES_UID}.zip"
LABEL_ZIP = Path("prism-uploads/ProstateX-TrainingLesionInformationv2.zip")
LONG_VRS = {b"OB", b"OD", b"OF", b"OL", b"OV", b"OW", b"SQ", b"UC", b"UR", b"UT", b"UN", b"SV", b"UV"}
FIELDS = {
    (0x0002, 0x0010): "transfer_syntax",
    (0x0008, 0x0016): "sop_class",
    (0x0008, 0x0018): "sop_uid",
    (0x0008, 0x0060): "modality",
    (0x0008, 0x103E): "description",
    (0x0010, 0x0020): "patient",
    (0x0018, 0x0050): "thickness",
    (0x0020, 0x000D): "study_uid",
    (0x0020, 0x000E): "series_uid",
    (0x0020, 0x0011): "series_number",
    (0x0020, 0x0013): "instance_number",
    (0x0020, 0x0032): "position",
    (0x0020, 0x0037): "orientation",
    (0x0020, 0x0052): "frame_uid",
    (0x0028, 0x0002): "samples_per_pixel",
    (0x0028, 0x0004): "photometric",
    (0x0028, 0x0008): "frames",
    (0x0028, 0x0010): "rows",
    (0x0028, 0x0011): "columns",
    (0x0028, 0x0030): "spacing",
    (0x0028, 0x0100): "bits_allocated",
    (0x0028, 0x0101): "bits_stored",
    (0x0028, 0x0102): "high_bit",
    (0x0028, 0x0103): "pixel_representation",
    (0x0028, 0x1052): "rescale_intercept",
    (0x0028, 0x1053): "rescale_slope",
    (0x0028, 0x1054): "rescale_type",
    (0x7FE0, 0x0010): "pixels",
}


def element_header(data, offset):
    if offset + 8 > len(data):
        raise ValueError("Truncated element header")
    tag = struct.unpack_from("<HH", data, offset)
    if tag[0] == 0xFFFE:
        return tag, None, struct.unpack_from("<I", data, offset + 4)[0], offset + 8
    vr = data[offset + 4:offset + 6]
    if not all(65 <= value <= 90 for value in vr):
        raise ValueError("Non-explicit-VR element")
    if vr in LONG_VRS:
        if offset + 12 > len(data) or data[offset + 6:offset + 8] != b"\0\0":
            raise ValueError("Invalid long-VR header")
        return tag, vr, struct.unpack_from("<I", data, offset + 8)[0], offset + 12
    return tag, vr, struct.unpack_from("<H", data, offset + 6)[0], offset + 8


def skip_undefined(data, offset, terminator, depth=0):
    if depth > 32:
        raise ValueError("Excessive nesting")
    while offset < len(data):
        tag, vr, length, start = element_header(data, offset)
        if tag == terminator:
            if length != 0:
                raise ValueError("Nonempty delimiter")
            return start
        if tag in {(0xFFFE, 0xE00D), (0xFFFE, 0xE0DD)}:
            raise ValueError("Unexpected delimiter")
        if length == 0xFFFFFFFF:
            if tag == (0xFFFE, 0xE000):
                nested_terminator = (0xFFFE, 0xE00D)
            elif vr == b"SQ":
                nested_terminator = (0xFFFE, 0xE0DD)
            else:
                raise ValueError("Unsupported undefined-length element")
            offset = skip_undefined(data, start, nested_terminator, depth + 1)
        else:
            offset = start + length
            if offset > len(data):
                raise ValueError("Truncated nested value")
    raise ValueError("Missing delimiter")


def read_sample(data):
    if data[128:132] != b"DICM":
        raise ValueError("Missing Part 10 marker")
    values = {}
    offset = 132
    while offset < len(data):
        tag, vr, length, start = element_header(data, offset)
        if tag[0] != 2 and values.get("transfer_syntax") != "1.2.840.10008.1.2.1":
            raise ValueError("Only Explicit VR Little Endian is supported")
        if length == 0xFFFFFFFF:
            if vr != b"SQ":
                raise ValueError("Only undefined-length sequences can be skipped")
            offset = skip_undefined(data, start, (0xFFFE, 0xE0DD))
            continue
        offset = start + length
        if offset > len(data):
            raise ValueError("Truncated element value")
        if tag not in FIELDS:
            continue
        name = FIELDS[tag]
        if name in values:
            raise ValueError("Duplicate selected top-level tag")
        raw = data[start:offset]
        if name == "pixels":
            values[name] = raw
        elif vr == b"US":
            if len(raw) != 2:
                raise ValueError("Expected scalar US")
            values[name] = struct.unpack("<H", raw)[0]
        else:
            values[name] = raw.decode("ascii").strip(" \0")
    if values["sop_class"] != "1.2.840.10008.5.1.4.1.1.4":
        raise ValueError("Expected classic MR Image Storage")
    if int(values.get("frames", 1)) != 1 or values["samples_per_pixel"] != 1:
        raise ValueError("Expected single-frame monochrome data")
    if values["bits_allocated"] != 16 or values["pixel_representation"] != 0:
        raise ValueError("Only native unsigned 16-bit data accepted")
    if values["high_bit"] != values["bits_stored"] - 1:
        raise ValueError("Unsupported stored-bit alignment")
    if values["photometric"] not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError("Unexpected photometric interpretation")
    if len(values["pixels"]) != 2 * values["rows"] * values["columns"]:
        raise ValueError("Unexpected pixel byte count")
    for name in ("position", "orientation", "spacing"):
        values[name] = np.asarray(values[name].split("\\"), dtype=float)
    return values


def audit(image_zip=IMAGE_ZIP, expected_uid=SERIES_UID, label_zip=LABEL_ZIP):
    samples = []
    with ZipFile(image_zip) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC failure")
        for entry in archive.infolist():
            if entry.filename.lower().endswith(".dcm"):
                sample = read_sample(archive.read(entry))
                sample["filename"] = entry.filename
                samples.append(sample)
    if len(samples) < 2:
        raise ValueError("At least two slices are required")
    reference = samples[0]
    if reference["series_uid"] != expected_uid:
        raise ValueError("Unexpected series")
    for field in ("patient", "study_uid", "series_uid", "frame_uid", "description", "series_number", "rows", "columns"):
        if len({sample[field] for sample in samples}) != 1:
            raise ValueError(f"Inconsistent {field}")
    if len({sample["sop_uid"] for sample in samples}) != len(samples):
        raise ValueError("Repeated SOP instance")
    column_direction = reference["orientation"][:3]
    row_direction = reference["orientation"][3:]
    basis = np.column_stack([column_direction, row_direction])
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-6)
    normal = np.cross(column_direction, row_direction)
    normal /= np.linalg.norm(normal)
    for sample in samples:
        np.testing.assert_allclose(sample["orientation"], reference["orientation"], atol=1e-6)
        np.testing.assert_allclose(sample["spacing"], reference["spacing"], atol=1e-6)
    samples.sort(key=lambda sample: float(sample["position"] @ normal))
    origins = np.asarray([sample["position"] for sample in samples])
    steps = np.diff(origins, axis=0)
    slice_step = steps.mean(axis=0)
    np.testing.assert_allclose(steps, np.broadcast_to(slice_step, steps.shape), atol=1e-4)
    slice_spacing = float(slice_step @ normal)
    if slice_spacing <= 0:
        raise ValueError("Invalid slice spacing")
    affine = np.eye(4)
    affine[:3, 0] = column_direction * reference["spacing"][1]
    affine[:3, 1] = row_direction * reference["spacing"][0]
    affine[:3, 2] = slice_step
    affine[:3, 3] = origins[0]
    with ZipFile(label_zip) as archive:
        tables = {}
        for basename in ("ProstateX-Images-Train.csv", "ProstateX-Findings-Train.csv"):
            name = next(name for name in archive.namelist() if name.endswith("/" + basename))
            tables[basename] = list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8-sig"))))
    matches = [row for row in tables["ProstateX-Images-Train.csv"]
               if row["ProxID"] == reference["patient"] and row["DCMSerDescr"] == reference["description"]
               and int(row["DCMSerNum"]) == int(reference["series_number"])]
    if not matches:
        raise ValueError("No matching findings for this series")
    results = []
    for row in matches:
        world = np.asarray(row["WorldMatrix"].split(","), dtype=float).reshape(4, 4)
        point = np.r_[np.asarray(row["pos"].split(), dtype=float), 1.0]
        dicom_ijk = np.linalg.solve(affine, point)[:3]
        world_ijk = np.linalg.solve(world, point)[:3]
        csv_ijk = np.asarray(row["ijk"].split(), dtype=float)
        labels = [finding for finding in tables["ProstateX-Findings-Train.csv"]
                  if all(finding[key] == row[key] for key in ("ProxID", "fid", "pos"))]
        if len(labels) != 1:
            raise ValueError("Ambiguous label")
        distances = np.abs((point[:3] - origins) @ normal)
        nearest = int(np.argmin(distances))
        if distances[nearest] > slice_spacing / 2 + 1e-4:
            raise ValueError("Finding outside slice coverage")
        if not (0 <= dicom_ijk[0] < reference["columns"] and 0 <= dicom_ijk[1] < reference["rows"]):
            raise ValueError("Finding outside image")
        np.testing.assert_allclose(affine @ np.r_[dicom_ijk, 1.0], point, atol=1e-9)
        results.append({
            "fid": row["fid"], "ClinSig": labels[0]["ClinSig"],
            "pos_mm": point[:3].tolist(), "dicom_continuous_ijk": dicom_ijk.tolist(),
            "worldmatrix_continuous_ijk": world_ijk.tolist(), "csv_ijk": csv_ijk.tolist(),
            "worldmatrix_minus_dicom_ijk": (world_ijk - dicom_ijk).tolist(),
            "csv_ijk_equals_rounded_worldmatrix": bool(np.array_equal(csv_ijk, np.rint(world_ijk))),
            "worldmatrix_in_dicom_coordinates": np.linalg.solve(affine, world).tolist(),
            "nearest_slice_zero_based": nearest, "nearest_filename": samples[nearest]["filename"],
            "instance_number": samples[nearest]["instance_number"],
            "distance_to_plane_mm": float(distances[nearest]),
        })
    report = {
        "scope": "One uploaded classic MR series; no clinical or whole-dataset validation",
        "image_zip_sha256": hashlib.sha256(image_zip.read_bytes()).hexdigest(),
        "labels_zip_sha256": hashlib.sha256(label_zip.read_bytes()).hexdigest(),
        "patient": reference["patient"], "series_uid": expected_uid, "description": reference["description"],
        "study_uid": reference["study_uid"], "frame_uid": reference["frame_uid"],
        "rescale_tags_by_slice": [
            {"filename": sample["filename"], **{key: sample.get(key) for key in
             ("rescale_intercept", "rescale_slope", "rescale_type")}} for sample in samples
        ],
        "series_number": reference["series_number"], "dicom_files": len(samples),
        "rows": reference["rows"], "columns": reference["columns"],
        "pixel_spacing_mm": reference["spacing"].tolist(), "slice_spacing_mm": slice_spacing,
        "max_step_deviation_mm": float(np.abs(steps - slice_step).max()),
        "dicom_affine_ijk_to_patient": affine.tolist(), "findings": results,
    }
    return report, samples


if __name__ == "__main__":
    report, _ = audit()
    destination = Path("analysis/prostatex_geometry_result.json")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))