"""Retrieve and audit one frozen PROSTATEx outer fold at a time."""

import csv
import hashlib
import io
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
from zipfile import ZIP_STORED, ZipFile

import numpy as np


PROTOCOL_SHA256 = "45a423324fab7f4e2971743ce2f5fd5961b7976e9745bb7567366bf89a889b08"
PLAN_SHA256 = "ae8fc814cc3fecfc9d6cf1059c82840b0d9acd0d2f0f016bf58d0cdfca96604b"
LABEL_ZIP_SHA256 = "fe862406213c058d7d714439f86a8df15efea7507aa7da3732a10fd9347c7c5e"
FINDINGS_CSV_SHA256 = "311c21c44691eb73868ca77f36f814300114117ca3aece209b5a446cf5fcf70f"
IMAGE_API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"
MAX_DOWNLOAD_BYTES = 512 * 1024**2
MAX_UNCOMPRESSED_BYTES = 1024 * 1024**3
BOUND_TOLERANCE_VOXELS = 1e-4
NATIVE_TRANSFER_SYNTAXES = {"1.2.840.10008.1.2", "1.2.840.10008.1.2.1"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path, expected_hash):
    raw = Path(path).read_bytes()
    require(sha256(raw) == expected_hash, "Unexpected file hash: " + str(path))
    return json.loads(raw)


def read_plan(path):
    raw = Path(path).read_bytes()
    require(sha256(raw) == PLAN_SHA256, "Unexpected download-plan hash")
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def open_zip_bytes(raw):
    archive = ZipFile(io.BytesIO(raw))
    entries = archive.infolist()
    require(len(entries) == len({entry.filename for entry in entries}), "Duplicate ZIP members")
    require(sum(entry.file_size for entry in entries) <= MAX_UNCOMPRESSED_BYTES,
            "ZIP exceeds uncompressed-size limit")
    require(archive.testzip() is None, "ZIP CRC failure")
    return archive


def read_findings(label_zip, patients):
    label_zip = Path(label_zip)
    require(sha256(label_zip.read_bytes()) == LABEL_ZIP_SHA256, "Unexpected label ZIP hash")
    with ZipFile(label_zip) as archive:
        require(archive.testzip() is None, "Label ZIP CRC failure")
        names = [name for name in archive.namelist()
                 if Path(name).name == "ProstateX-Findings-Train.csv"]
        require(len(names) == 1, "Expected one findings CSV")
        raw = archive.read(names[0])
    require(sha256(raw) == FINDINGS_CSV_SHA256, "Unexpected findings CSV hash")
    selected = {}
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
        if row["ProxID"] not in patients:
            continue
        point = [float(value) for value in row["pos"].split()]
        require(len(point) == 3 and np.isfinite(point).all(), "Invalid finding position")
        selected.setdefault(row["ProxID"], []).append({
            "patient": row["ProxID"], "fid": row["fid"], "pos_text": row["pos"],
            "pos_mm": point, "zone": row["zone"], "ClinSig": row["ClinSig"].upper(),
            "full_key": f"{row['ProxID']}|{row['fid']}|{row['pos']}",
        })
    require(set(selected) == set(patients), "Missing fold findings")
    keys = [row["full_key"] for rows in selected.values() for row in rows]
    require(len(keys) == len(set(keys)), "Duplicate full finding key")
    return selected


def download_series(series_uid):
    url = IMAGE_API + "?" + urlencode({"SeriesInstanceUID": series_uid})
    last_error = None
    for attempt in range(1, 4):
        try:
            request = Request(url, headers={"Cache-Control": "no-cache", "User-Agent": "PROSTATEx-QC/0.3"})
            with urlopen(request, timeout=180) as response:
                parsed = urlsplit(response.geturl())
                require(parsed.scheme == "https" and parsed.hostname == "services.cancerimagingarchive.net",
                        "Unexpected download redirect")
                require(response.status == 200, "Unexpected HTTP status")
                raw = response.read(MAX_DOWNLOAD_BYTES + 1)
                require(len(raw) <= MAX_DOWNLOAD_BYTES, "Series download exceeds size limit")
                with open_zip_bytes(raw):
                    pass
                return raw, {"requested_url": url, "final_url": response.geturl(),
                             "http_status": response.status, "server_date": response.headers.get("Date"),
                             "attempt": attempt, "downloaded_utc": datetime.now(timezone.utc).isoformat()}
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
        require(len(names) == int(expected["image_count"]),
                f"Unexpected DICOM count for {expected['series_uid']}")
        for name in names:
            dicom = archive.read(name)
            dataset = pydicom.dcmread(io.BytesIO(dicom), force=False)
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
            details.append({"filename": name, "sop_uid": sop, "dicom_sha256": sha256(dicom),
                            "rows": int(dataset.Rows), "columns": int(dataset.Columns),
                            "bits_allocated": int(dataset.BitsAllocated),
                            "bits_stored": int(dataset.BitsStored),
                            "pixel_representation": int(dataset.PixelRepresentation),
                            "photometric_interpretation": str(dataset.PhotometricInterpretation),
                            "transfer_syntax": transfer_syntax})
    require(all(len(values) == 1 for values in identities.values()), "Inconsistent DICOM identities")
    identities = {field: next(iter(values)) for field, values in identities.items()}
    require(identities["PatientID"] == expected["patient"], "Unexpected DICOM patient")
    require(identities["StudyInstanceUID"] == expected["study_uid"], "Unexpected DICOM study")
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
    rows, columns = details[0]["rows"], details[0]["columns"]
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
    coordinates = np.linalg.solve(np.asarray(target_geometry["affine_ijk_to_patient"]), homogeneous)[:3]
    limits = np.asarray(target_geometry["shape_zyx"][::-1], dtype=float) - 1
    outside = np.any((coordinates < -BOUND_TOLERANCE_VOXELS)
                     | (coordinates > limits[:, None] + BOUND_TOLERANCE_VOXELS), axis=0)
    return {"inside": not bool(outside.any()), "outside_points": int(outside.sum()),
            "boundary_tolerance_voxels": BOUND_TOLERANCE_VOXELS,
            "min_source_ijk": coordinates.min(axis=1).tolist(),
            "max_source_ijk": coordinates.max(axis=1).tolist(), "limits_ijk": limits.tolist()}


def seed_qc0025_cache(cache_dir, plan_by_uid):
    candidates = sorted(Path("/kaggle/working").glob("prostatex_0025_multistudy_*.zip"))
    if not candidates:
        return []
    seeded = []
    source = candidates[-1]
    with ZipFile(source) as archive:
        if archive.testzip() is not None or "report.json" not in archive.namelist():
            return []
        report = json.loads(archive.read("report.json"))
        if report.get("status") != "COMPLETE_UNIQUE_ASSIGNMENTS":
            return []
        for study in report["studies"].values():
            for channel in study["channels"].values():
                uid = channel["series_uid"]
                if uid not in plan_by_uid:
                    continue
                raw = archive.read(channel["archive_file"])
                if sha256(raw) != channel["archive_sha256"]:
                    continue
                target = cache_dir / (uid + ".zip")
                target.write_bytes(raw)
                seeded.append(uid)
    return seeded


def retrieve_fold(fold_index, plan_path, label_zip, protocol_path):
    fold_index = int(fold_index)
    require(0 <= fold_index <= 4, "Fold must be 0 through 4")
    protocol = read_json(protocol_path, PROTOCOL_SHA256)
    require(protocol["protocol_id"] == "PROSTATEx-selection-v0.3", "Unexpected protocol")
    all_plan = read_plan(plan_path)
    plan = [row for row in all_plan if int(row["outer_fold"]) == fold_index]
    require(plan and len(plan) == len({row["series_uid"] for row in plan}), "Invalid fold plan")
    patients = sorted({row["patient"] for row in plan})
    findings = read_findings(label_zip, patients)
    assignments = {row["full_key"]: row["assigned_study_uid"]
                   for row in protocol["multi_study_resolution"]["finding_assignments"]}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = Path(f"prostatex_fold_{fold_index}_retrieval_{stamp}")
    output.mkdir(exist_ok=False)
    cache_dir = Path(f"prostatex_fold_{fold_index}_cache")
    cache_dir.mkdir(exist_ok=True)
    report = {
        "scope": "One frozen PROSTATEx outer-fold retrieval with DICOM and complete-grid geometry audit; no patch extraction, display review, training, or clinical validation",
        "status": "FAILED", "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"], "outer_fold": fold_index,
        "planned_patients": len(patients), "planned_series": len(plan),
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "input_sha256": {"protocol": PROTOCOL_SHA256, "plan": PLAN_SHA256,
                         "label_zip": LABEL_ZIP_SHA256},
        "cache_directory": str(cache_dir), "series": [], "patients": {}, "findings": [],
    }
    try:
        import pydicom
        report["software"]["pydicom"] = pydicom.__version__
        plan_by_uid = {row["series_uid"]: row for row in plan}
        report["seeded_qc0025_series"] = seed_qc0025_cache(cache_dir, plan_by_uid)
        geometries = {}
        for series_index, expected in enumerate(plan, start=1):
            uid = expected["series_uid"]
            cache_path = cache_dir / (uid + ".zip")
            source = "CACHE"
            retrieval = None
            raw = cache_path.read_bytes() if cache_path.exists() else None
            if raw is not None:
                try:
                    with open_zip_bytes(raw):
                        pass
                except Exception:
                    cache_path.unlink()
                    raw = None
            if raw is None:
                source = "TCIA_FRESH_DOWNLOAD"
                print(f"Fold {fold_index}: downloading {series_index}/{len(plan)} {expected['patient']} {expected['channel']}",
                      flush=True)
                raw, retrieval = download_series(uid)
                cache_path.write_bytes(raw)
            geometry = None
            geometry_error = None
            try:
                geometry = derive_geometry(raw, expected, pydicom)
            except Exception as error:
                geometry_error = f"{type(error).__name__}: {error}"
            archive_name = f"series/{uid}.zip"
            geometries[uid] = geometry
            report["series"].append({**expected, "archive_name": archive_name,
                                     "archive_sha256": sha256(raw), "archive_bytes": len(raw),
                                     "source": source, "retrieval": retrieval,
                                     "audit_status": "PASSED" if geometry is not None else "FAILED_HARD_CHECK",
                                     "audit_error": geometry_error, "geometry": geometry})
        for patient in patients:
            patient_rows = [row for row in plan if row["patient"] == patient]
            studies = {}
            for row in patient_rows:
                studies.setdefault(row["study_uid"], {})[row["channel"]] = row
            require(all(set(pair) == {"T2_axial", "ADC"} for pair in studies.values()),
                    "Incomplete study pair: " + patient)
            failed_series = [row for row in patient_rows
                             if geometries[row["series_uid"]] is None]
            patient_report = {"studies": {}, "finding_count": len(findings[patient]),
                              "status": "EXCLUDED_HARD_CHECK" if failed_series else "PASSED",
                              "failed_series": [{"channel": row["channel"],
                                                 "series_uid": row["series_uid"],
                                                 "error": next(item["audit_error"] for item in report["series"]
                                                               if item["series_uid"] == row["series_uid"])}
                                                for row in failed_series],
                              "failed_checks": []}
            if failed_series:
                report["patients"][patient] = patient_report
                for finding in findings[patient]:
                    report["findings"].append({**finding, "assigned_study_uid": None,
                                               "status": "EXCLUDED_SOURCE_GEOMETRY",
                                               "bounds": None})
                continue
            for study_uid, pair in studies.items():
                t2_geometry = geometries[pair["T2_axial"]["series_uid"]]
                adc_geometry = geometries[pair["ADC"]["series_uid"]]
                frames = {t2_geometry["identities"]["FrameOfReferenceUID"],
                          adc_geometry["identities"]["FrameOfReferenceUID"]}
                if len(frames) != 1:
                    patient_report["status"] = "EXCLUDED_HARD_CHECK"
                    patient_report["failed_checks"].append({
                        "check": "CROSS_CHANNEL_FRAME_OF_REFERENCE_UID",
                        "study_uid": study_uid,
                        "T2_axial_series_uid": pair["T2_axial"]["series_uid"],
                        "T2_axial_frame_of_reference_uid":
                            t2_geometry["identities"]["FrameOfReferenceUID"],
                        "ADC_series_uid": pair["ADC"]["series_uid"],
                        "ADC_frame_of_reference_uid":
                            adc_geometry["identities"]["FrameOfReferenceUID"],
                    })
                    continue
                patient_report["studies"][study_uid] = {
                    "shared_frame_of_reference_uid": next(iter(frames)),
                    "T2_axial_series_uid": pair["T2_axial"]["series_uid"],
                    "ADC_series_uid": pair["ADC"]["series_uid"],
                }
            report["patients"][patient] = patient_report
            if patient_report["failed_checks"]:
                for finding in findings[patient]:
                    report["findings"].append({**finding, "assigned_study_uid": None,
                                               "status": "EXCLUDED_CROSS_CHANNEL_FRAME",
                                               "bounds": None})
                continue
            for finding in findings[patient]:
                if patient == "ProstateX-0025":
                    require(finding["full_key"] in assignments, "Missing QC0025 assignment")
                    study_uid = assignments[finding["full_key"]]
                else:
                    require(len(studies) == 1, "Unexpected multiple primary studies: " + patient)
                    study_uid = next(iter(studies))
                require(study_uid in studies, "Assigned study absent from fold plan")
                pair = studies[study_uid]
                t2_geometry = geometries[pair["T2_axial"]["series_uid"]]
                bounds = {channel: grid_bounds(finding["pos_mm"], t2_geometry,
                                                geometries[pair[channel]["series_uid"]])
                          for channel in ("T2_axial", "ADC")}
                status = "PASSED" if all(item["inside"] for item in bounds.values()) else "OUT_OF_BOUNDS"
                report["findings"].append({**finding, "assigned_study_uid": study_uid,
                                           "status": status, "bounds": bounds})
        semantic_flags = sum(len(row["geometry"]["semantic_flags"]) for row in report["series"]
                             if row["geometry"] is not None)
        out_of_bounds_findings = sum(row["status"] == "OUT_OF_BOUNDS" for row in report["findings"])
        excluded_findings = sum(row["status"].startswith("EXCLUDED_")
                                for row in report["findings"])
        excluded_patients = sum(row["status"] == "EXCLUDED_HARD_CHECK"
                                for row in report["patients"].values())
        report["summary"] = {
            "retrieved_series": len(report["series"]), "audited_patients": len(report["patients"]),
            "audited_findings": len(report["findings"]),
            "out_of_bounds_findings": out_of_bounds_findings,
            "excluded_patients": excluded_patients, "excluded_findings": excluded_findings,
            "semantic_flags": semantic_flags,
        }
        report["status"] = ("REVIEW_REQUIRED" if semantic_flags or out_of_bounds_findings else
                            "COMPLETE_WITH_EXCLUSIONS" if excluded_patients else "COMPLETE")
        with (output / "series_index.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["patient", "outer_fold", "study_uid", "channel", "series_number", "series_uid",
                      "description", "image_count", "archive_name", "archive_sha256", "archive_bytes",
                      "source", "audit_status", "audit_error"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in report["series"]:
                writer.writerow({key: row[key] for key in fields})
        with (output / "finding_audit.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["patient", "fid", "pos_text", "ClinSig", "assigned_study_uid", "status",
                      "T2_outside_points", "ADC_outside_points"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in report["findings"]:
                writer.writerow({"patient": row["patient"], "fid": row["fid"],
                                 "pos_text": row["pos_text"], "ClinSig": row["ClinSig"],
                                 "assigned_study_uid": row["assigned_study_uid"] or "",
                                 "status": row["status"],
                                 "T2_outside_points": (row["bounds"]["T2_axial"]["outside_points"]
                                                       if row["bounds"] is not None else ""),
                                 "ADC_outside_points": (row["bounds"]["ADC"]["outside_points"]
                                                        if row["bounds"] is not None else "")})
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                        encoding="utf-8")
    local_files = sorted(path for path in output.iterdir() if path.is_file())
    artifacts = {path.name: {"sha256": sha256(path.read_bytes()), "bytes": path.stat().st_size}
                 for path in local_files}
    for row in report.get("series", []):
        cache_path = cache_dir / (row["series_uid"] + ".zip")
        if cache_path.exists():
            artifacts[row["archive_name"]] = {"sha256": sha256(cache_path.read_bytes()),
                                               "bytes": cache_path.stat().st_size}
    (output / "artifact_hashes.json").write_text(json.dumps(artifacts, indent=2) + "\n",
                                                  encoding="utf-8")
    bundle = output.with_suffix(".zip")
    with ZipFile(bundle, "x", compression=ZIP_STORED, allowZip64=True) as archive:
        for path in local_files:
            archive.write(path, arcname=path.name)
        archive.write(output / "artifact_hashes.json", arcname="artifact_hashes.json")
        for row in report.get("series", []):
            cache_path = cache_dir / (row["series_uid"] + ".zip")
            if cache_path.exists():
                archive.write(cache_path, arcname=row["archive_name"])
    print("Status:", report["status"])
    if "error" in report:
        print("Error:", report["error"])
    else:
        print("Summary:", json.dumps(report["summary"]))
    print("Saved:", bundle)
    return bundle


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("fold", type=int)
    parser.add_argument("plan_csv")
    parser.add_argument("label_zip")
    parser.add_argument("protocol_json")
    arguments = parser.parse_args()
    retrieve_fold(arguments.fold, arguments.plan_csv, arguments.label_zip, arguments.protocol_json)