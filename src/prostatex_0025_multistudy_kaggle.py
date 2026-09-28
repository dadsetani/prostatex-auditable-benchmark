"""Resolve the ProstateX-0025 study assignment by DICOM geometry only."""

import csv
import hashlib
import io
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
from zipfile import ZIP_STORED, ZipFile

import numpy as np


PATIENT = "ProstateX-0025"
METADATA_BUNDLE_SHA256 = "bfb848cc895336686bf84153db304f4a750a7122d43530af059dbc31828032ce"
PROTOCOL_SHA256 = "d18f0465abe1ff4e39e8e4a4ffd786f97d80282be6c3f938ea90600a59ef4501"
FINDINGS_CSV_SHA256 = "311c21c44691eb73868ca77f36f814300114117ca3aece209b5a446cf5fcf70f"
IMAGE_API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"
MAX_DOWNLOAD_BYTES = 512 * 1024**2
MAX_UNCOMPRESSED_BYTES = 1024 * 1024**3
NATIVE_TRANSFER_SYNTAXES = {"1.2.840.10008.1.2", "1.2.840.10008.1.2.1"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def open_zip_bytes(raw):
    archive = ZipFile(io.BytesIO(raw))
    entries = archive.infolist()
    require(len(entries) == len({entry.filename for entry in entries}), "Duplicate ZIP members")
    require(sum(entry.file_size for entry in entries) <= MAX_UNCOMPRESSED_BYTES,
            "ZIP exceeds uncompressed-size limit")
    require(archive.testzip() is None, "ZIP CRC failure")
    return archive


def read_findings(label_zip):
    with ZipFile(label_zip) as archive:
        require(archive.testzip() is None, "Label ZIP CRC failure")
        matches = [name for name in archive.namelist()
                   if Path(name).name == "ProstateX-Findings-Train.csv"]
        require(len(matches) == 1, "Expected one findings CSV")
        raw = archive.read(matches[0])
    require(sha256(raw) == FINDINGS_CSV_SHA256, "Unexpected findings CSV hash")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    selected = []
    for row in rows:
        if row["ProxID"] != PATIENT:
            continue
        point = tuple(float(value) for value in row["pos"].split())
        require(len(point) == 3 and np.isfinite(point).all(), "Invalid finding position")
        selected.append({"patient": PATIENT, "fid": row["fid"], "pos_mm": list(point),
                         "zone": row["zone"], "ClinSig": row["ClinSig"].upper()})
    require(len(selected) == 5, "Expected five ProstateX-0025 findings")
    keys = {(row["patient"], row["fid"], tuple(row["pos_mm"])) for row in selected}
    require(len(keys) == 5, "Duplicate full finding key")
    return selected


def read_study_pairs(metadata_bundle):
    raw = Path(metadata_bundle).read_bytes()
    require(sha256(raw) == METADATA_BUNDLE_SHA256, "Unexpected metadata bundle hash")
    with open_zip_bytes(raw) as archive:
        rows = list(csv.DictReader(io.StringIO(archive.read("series_candidates.csv").decode())))
    candidates = [row for row in rows if row["patient"] == PATIENT
                  and row["status"] == "ELIGIBLE_METADATA"]
    require(len(candidates) == 4, "Expected four eligible series")
    pairs = {}
    for row in candidates:
        require(row["channel"] in {"T2_axial", "ADC"}, "Unexpected channel")
        study = row["study_uid"]
        require(row["channel"] not in pairs.setdefault(study, {}), "Duplicate study channel")
        pairs[study][row["channel"]] = {
            "series_uid": row["series_uid"], "series_number": int(row["series_number"]),
            "description": row["description"], "expected_instances": int(row["image_count"]),
        }
    require(len(pairs) == 2 and all(set(pair) == {"T2_axial", "ADC"} for pair in pairs.values()),
            "Expected two complete study pairs")
    return pairs


def download_series(series_uid):
    url = IMAGE_API + "?" + urlencode({"SeriesInstanceUID": series_uid})
    last_error = None
    for attempt in range(1, 4):
        try:
            request = Request(url, headers={"Cache-Control": "no-cache", "User-Agent": "PROSTATEx-QC/0.2"})
            with urlopen(request, timeout=180) as response:
                parsed = urlsplit(response.geturl())
                require(parsed.scheme == "https" and parsed.hostname == "services.cancerimagingarchive.net",
                        "Unexpected download redirect")
                require(response.status == 200, "Unexpected HTTP status")
                raw = response.read(MAX_DOWNLOAD_BYTES + 1)
                require(len(raw) <= MAX_DOWNLOAD_BYTES, "Series download exceeds size limit")
                return raw, {"requested_url": url, "final_url": response.geturl(),
                             "http_status": response.status, "server_date": response.headers.get("Date"),
                             "attempt": attempt}
        except Exception as error:
            last_error = error
            if attempt < 3:
                time.sleep(5 * attempt)
    raise last_error


def derive_geometry(raw_zip, expected, pydicom):
    positions, orientations, spacings, details = [], [], [], []
    identity_fields = ("PatientID", "StudyInstanceUID", "SeriesInstanceUID", "FrameOfReferenceUID")
    identities = {field: set() for field in identity_fields}
    sops = set()
    semantic_keywords = {"RescaleSlope", "RescaleIntercept", "RescaleType", "ModalityLUTSequence",
                         "RealWorldValueMappingSequence", "PixelValueTransformationSequence",
                         "PixelPaddingValue", "PixelPaddingRangeLimit"}
    semantic_flags = []
    with open_zip_bytes(raw_zip) as archive:
        names = [entry.filename for entry in archive.infolist()
                 if not entry.is_dir() and entry.filename.lower().endswith(".dcm")]
        require(len(names) == expected["expected_instances"], "Unexpected DICOM count")
        for name in names:
            raw = archive.read(name)
            dataset = pydicom.dcmread(io.BytesIO(raw), force=False)
            transfer_syntax = str(dataset.file_meta.TransferSyntaxUID)
            require(transfer_syntax in NATIVE_TRANSFER_SYNTAXES, "Unsupported transfer syntax")
            require(str(dataset.Modality) == "MR"
                    and str(dataset.SOPClassUID) == "1.2.840.10008.5.1.4.1.1.4",
                    "Expected classic single-frame MR")
            require(str(dataset.file_meta.MediaStorageSOPClassUID) == str(dataset.SOPClassUID),
                    "Media-storage SOP class mismatch")
            sop = str(dataset.SOPInstanceUID)
            require(sop and sop not in sops
                    and str(dataset.file_meta.MediaStorageSOPInstanceUID) == sop,
                    "Duplicate or file-meta SOP instance mismatch")
            sops.add(sop)
            for field in identity_fields:
                value = str(getattr(dataset, field, ""))
                require(value, "Missing identity field: " + field)
                identities[field].add(value)
            require(int(getattr(dataset, "NumberOfFrames", 1)) == 1
                    and int(dataset.SamplesPerPixel) == 1
                    and str(dataset.PhotometricInterpretation) in {"MONOCHROME1", "MONOCHROME2"},
                    "Unsupported pixel organization")
            require(int(dataset.Rows) > 0 and int(dataset.Columns) > 0,
                    "Invalid image dimensions")
            require(int(dataset.BitsAllocated) in {8, 16}
                    and int(dataset.PixelRepresentation) in {0, 1}, "Unsupported stored pixels")
            require("PixelData" in dataset, "Missing PixelData")
            positions.append([float(value) for value in dataset.ImagePositionPatient])
            orientations.append([float(value) for value in dataset.ImageOrientationPatient])
            spacings.append([float(value) for value in dataset.PixelSpacing])
            for element in dataset.iterall():
                if element.keyword in semantic_keywords:
                    semantic_flags.append({"filename": name, "keyword": element.keyword,
                                           "tag": str(element.tag),
                                           "value": "SEQUENCE_PRESENT" if element.VR == "SQ"
                                           else str(element.value)})
            details.append({"filename": name, "sop_uid": sop, "dicom_sha256": sha256(raw),
                            "rows": int(dataset.Rows), "columns": int(dataset.Columns),
                            "transfer_syntax": transfer_syntax})
    require(all(len(values) == 1 for values in identities.values()), "Inconsistent DICOM identities")
    identities = {field: next(iter(values)) for field, values in identities.items()}
    require(identities["PatientID"] == PATIENT, "Unexpected DICOM patient")
    require(identities["SeriesInstanceUID"] == expected["series_uid"], "Unexpected DICOM series")
    orientation = np.asarray(orientations, dtype=float)
    spacing = np.asarray(spacings, dtype=float)
    position = np.asarray(positions, dtype=float)
    require(np.isfinite(orientation).all() and np.isfinite(spacing).all()
            and np.isfinite(position).all(), "Nonfinite geometry")
    require(np.max(np.abs(orientation - orientation[0])) <= 1e-6, "Varying orientation")
    require(np.max(np.abs(spacing - spacing[0])) <= 1e-6 and np.all(spacing > 0),
            "Varying or invalid pixel spacing")
    basis = orientation[0].reshape(2, 3).T
    require(np.max(np.abs(basis.T @ basis - np.eye(2))) <= 1e-6,
            "Nonorthonormal in-plane directions")
    normal = np.cross(basis[:, 0], basis[:, 1])
    require(np.linalg.norm(normal) > 0.999999, "Invalid slice normal")
    normal /= np.linalg.norm(normal)
    order = np.argsort(position @ normal)
    steps = np.diff(position[order], axis=0)
    require(np.all(steps @ normal > 0), "Duplicate or reversed physical planes")
    step = steps.mean(axis=0)
    deviation = float(np.max(np.abs(steps - step)))
    require(deviation <= 1e-4, "Irregular slice-step vectors")
    rows = details[0]["rows"]
    columns = details[0]["columns"]
    require(all(item["rows"] == rows and item["columns"] == columns for item in details),
            "Varying image dimensions")
    affine = np.eye(4)
    affine[:3, 0] = basis[:, 0] * spacing[0, 1]
    affine[:3, 1] = basis[:, 1] * spacing[0, 0]
    affine[:3, 2] = step
    affine[:3, 3] = position[order[0]]
    require(abs(np.linalg.det(affine[:3, :3])) > 1e-12, "Singular geometry")
    return {
        "identities": identities, "shape_zyx": [len(order), rows, columns],
        "affine_ijk_to_patient": affine.tolist(), "column_direction": basis[:, 0].tolist(),
        "row_direction": basis[:, 1].tolist(), "pixel_spacing_row_column_mm": spacing[0].tolist(),
        "slice_step_vector_mm": step.tolist(), "max_step_component_deviation_mm": deviation,
        "semantic_flags": semantic_flags, "sorted_instances": [details[index] for index in order],
    }


def grid_bounds(point, t2_geometry, target_geometry):
    offsets = (np.arange(100) - 49.5) * 0.5
    column_offsets, row_offsets = np.meshgrid(offsets, offsets, indexing="xy")
    column = np.asarray(t2_geometry["column_direction"])
    row = np.asarray(t2_geometry["row_direction"])
    world = (np.asarray(point)[:, None, None] + column[:, None, None] * column_offsets
             + row[:, None, None] * row_offsets)
    homogeneous = np.vstack((world.reshape(3, -1), np.ones(10000)))
    affine = np.asarray(target_geometry["affine_ijk_to_patient"])
    coordinates = np.linalg.solve(affine, homogeneous)[:3]
    limits = np.asarray(target_geometry["shape_zyx"][::-1], dtype=float) - 1
    outside = np.any((coordinates < 0) | (coordinates > limits[:, None]), axis=0)
    return {"inside": not bool(outside.any()), "outside_points": int(outside.sum()),
            "min_source_ijk": coordinates.min(axis=1).tolist(),
            "max_source_ijk": coordinates.max(axis=1).tolist(),
            "limits_ijk": limits.tolist()}


def resolve_0025(metadata_bundle, label_zip, protocol_path):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = Path("prostatex_0025_multistudy_" + stamp)
    output.mkdir(exist_ok=False)
    report = {
        "scope": "Geometry-only assignment of five ProstateX-0025 findings to two study pairs; no patch extraction, display review, training, or clinical validation",
        "status": "FAILED", "created_utc": datetime.now(timezone.utc).isoformat(),
        "patient": PATIENT, "software": {"python": platform.python_version(), "numpy": np.__version__},
        "inputs": {}, "studies": {}, "findings": [],
    }
    try:
        import pydicom
        report["software"]["pydicom"] = pydicom.__version__
        protocol_raw = Path(protocol_path).read_bytes()
        require(sha256(protocol_raw) == PROTOCOL_SHA256, "Unexpected protocol hash")
        protocol = json.loads(protocol_raw)
        require(protocol["protocol_id"] == "PROSTATEx-selection-v0.2", "Unexpected protocol")
        report["inputs"] = {
            "metadata_bundle_sha256": sha256(Path(metadata_bundle).read_bytes()),
            "label_zip_sha256": sha256(Path(label_zip).read_bytes()),
            "protocol_sha256": sha256(protocol_raw),
        }
        pairs = read_study_pairs(metadata_bundle)
        findings = read_findings(label_zip)
        geometries = {}
        for study_index, (study_uid, pair) in enumerate(sorted(pairs.items()), start=1):
            study_report = {"study_uid": study_uid, "channels": {}}
            geometries[study_uid] = {}
            for channel in ("T2_axial", "ADC"):
                expected = pair[channel]
                print(f"Downloading {PATIENT} study {study_index}/2 {channel}...", flush=True)
                raw, retrieval = download_series(expected["series_uid"])
                filename = f"study_{study_index}_{channel}.zip"
                (output / filename).write_bytes(raw)
                geometry = derive_geometry(raw, expected, pydicom)
                require(geometry["identities"]["StudyInstanceUID"] == study_uid,
                        "Study UID differs from metadata inventory")
                geometries[study_uid][channel] = geometry
                study_report["channels"][channel] = {
                    **expected, "archive_file": filename, "archive_sha256": sha256(raw),
                    "retrieval": retrieval, "geometry": geometry,
                }
            frames = {geometries[study_uid][channel]["identities"]["FrameOfReferenceUID"]
                      for channel in ("T2_axial", "ADC")}
            require(len(frames) == 1, "Cross-channel FrameOfReferenceUID mismatch")
            study_report["shared_frame_of_reference_uid"] = next(iter(frames))
            report["studies"][study_uid] = study_report
        assignment_rows = []
        for finding in findings:
            valid_studies = []
            assessments = {}
            for study_uid in sorted(geometries):
                t2 = geometries[study_uid]["T2_axial"]
                channel_bounds = {channel: grid_bounds(finding["pos_mm"], t2,
                                                        geometries[study_uid][channel])
                                  for channel in ("T2_axial", "ADC")}
                paired_inside = all(item["inside"] for item in channel_bounds.values())
                assessments[study_uid] = {"paired_grid_inside": paired_inside,
                                          "channels": channel_bounds}
                if paired_inside:
                    valid_studies.append(study_uid)
            status = "UNIQUE_STUDY_ASSIGNED" if len(valid_studies) == 1 else "REVIEW_REQUIRED"
            assigned = valid_studies[0] if len(valid_studies) == 1 else None
            result = {**finding, "full_key": f"{PATIENT}|{finding['fid']}|{' '.join(map(str, finding['pos_mm']))}",
                      "status": status, "valid_study_uids": valid_studies,
                      "assigned_study_uid": assigned, "assessments": assessments}
            report["findings"].append(result)
            assignment_rows.append({"patient": PATIENT, "fid": finding["fid"],
                                    "pos_mm": " ".join(map(str, finding["pos_mm"])),
                                    "ClinSig": finding["ClinSig"], "status": status,
                                    "assigned_study_uid": assigned or "",
                                    "valid_study_count": len(valid_studies)})
        report["status"] = ("COMPLETE_UNIQUE_ASSIGNMENTS"
                            if all(row["status"] == "UNIQUE_STUDY_ASSIGNED"
                                   for row in report["findings"])
                            else "REVIEW_REQUIRED")
        with (output / "assignments.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(assignment_rows[0]))
            writer.writeheader()
            writer.writerows(assignment_rows)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                        encoding="utf-8")
    artifact_names = sorted(path.name for path in output.iterdir() if path.is_file())
    manifest = {name: {"sha256": sha256((output / name).read_bytes()),
                       "bytes": (output / name).stat().st_size} for name in artifact_names}
    (output / "artifact_hashes.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                  encoding="utf-8")
    bundle = output.with_suffix(".zip")
    with ZipFile(bundle, "x", compression=ZIP_STORED) as archive:
        for name in artifact_names + ["artifact_hashes.json"]:
            archive.write(output / name, arcname=name)
    print("Status:", report["status"])
    if "error" in report:
        print("Error:", report["error"])
    for row in report["findings"]:
        print(row["full_key"], row["status"], row["assigned_study_uid"])
    print("Saved:", bundle)
    return bundle


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_bundle")
    parser.add_argument("label_zip")
    parser.add_argument("protocol")
    arguments = parser.parse_args()
    resolve_0025(arguments.metadata_bundle, arguments.label_zip, arguments.protocol)