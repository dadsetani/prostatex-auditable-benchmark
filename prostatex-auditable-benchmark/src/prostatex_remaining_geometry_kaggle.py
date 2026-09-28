"""Step 6: eight-series geometry and finding-key audit; no patch generation."""

import csv
import hashlib
import io
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np


class GeometryMemoryZip(io.BytesIO):
    def read_bytes(self):
        return self.getvalue()


def finding_map(patient, findings):
    mapped = {}
    for finding in findings:
        point = tuple(float(value) for value in finding["pos_mm"])
        if len(point) != 3 or not np.isfinite(point).all():
            raise ValueError("Invalid finding position")
        key = (patient, str(finding["fid"]), point)
        label = finding["ClinSig"].strip().upper()
        if key in mapped or label not in {"TRUE", "FALSE"}:
            raise ValueError("Duplicate full finding key or invalid label")
        mapped[key] = label
    return mapped


def strict_geometry(samples):
    reference = samples[0]
    orientation = reference["orientation"]
    if orientation.shape != (6,) or not np.isfinite(orientation).all():
        raise ValueError("Invalid orientation")
    basis = np.column_stack((orientation[:3], orientation[3:]))
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), rtol=0, atol=1e-6)
    for sample in samples:
        for field, shape in (("position", (3,)), ("spacing", (2,)), ("orientation", (6,))):
            if sample[field].shape != shape or not np.isfinite(sample[field]).all():
                raise ValueError(f"Invalid {field}")
        if np.any(sample["spacing"] <= 0) or min(sample["rows"], sample["columns"]) <= 0:
            raise ValueError("Nonpositive image dimensions or spacing")
        for field in ("orientation", "spacing"):
            np.testing.assert_allclose(sample[field], reference[field], rtol=0, atol=1e-6)
    steps = np.diff([sample["position"] for sample in samples], axis=0)
    mean_step = steps.mean(axis=0)
    np.testing.assert_allclose(steps, np.broadcast_to(mean_step, steps.shape), rtol=0, atol=1e-4)
    normal = np.cross(orientation[:3], orientation[3:])
    if np.any(steps @ normal <= 0):
        raise ValueError("Repeated or reversed slice planes")


def audit_remaining_geometry(bundle_path, labels_path):
    bundle_path, labels_path = Path(bundle_path), Path(labels_path)
    with ZipFile(labels_path) as labels_archive:
        tables, csv_hashes = {}, {}
        for basename in ("ProstateX-Findings-Train.csv", "ProstateX-Images-Train.csv"):
            names = [name for name in labels_archive.namelist() if Path(name).name == basename]
            if len(names) != 1:
                raise ValueError(f"Expected one {basename}")
            raw = labels_archive.read(names[0])
            tables[basename] = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
            csv_hashes[basename] = hashlib.sha256(raw).hexdigest()
    by_patient = {}
    with ZipFile(bundle_path) as outer:
        if outer.testzip() is not None or len(outer.namelist()) != len(set(outer.namelist())):
            raise ValueError("Invalid outer archive")
        download = json.loads(outer.read("download_report.json"))
        snapshot = outer.read("inventory.json")
        if download["status"] != "complete" or hashlib.sha256(snapshot).hexdigest() != download["inventory_sha256"]:
            raise ValueError("Invalid download report or inventory hash")
        selected = select_remaining(json.loads(snapshot))
        records = download["series"]
        if len(records) != 8 or len({row["series_uid"] for row in records}) != 8:
            raise ValueError("Expected eight distinct series")
        for candidate in selected:
            matches = [row for row in records if row["series_uid"] == candidate["series_uid"]]
            if len(matches) != 1 or any(matches[0][key] != value for key, value in candidate.items()):
                raise ValueError("Download/inventory mismatch")
            record = matches[0]
            raw = outer.read(record["file"])
            if hashlib.sha256(raw).hexdigest() != record["zip_sha256"] or len(raw) != record["zip_bytes"]:
                raise ValueError("Series archive hash/size mismatch")
            report, samples = audit(GeometryMemoryZip(raw), record["series_uid"], labels_path)
            strict_geometry(samples)
            for field in ("patient", "study_uid", "description"):
                if report[field] != record[field]:
                    raise ValueError(f"Identity mismatch: {field}")
            if int(report["series_number"]) != int(record["series_number"]) or len(samples) != int(record["image_count"]):
                raise ValueError("Series number/count mismatch")
            by_patient.setdefault(report["patient"], {})[record["channel"]] = report
    summaries = []
    for patient, channels in sorted(by_patient.items()):
        t2, adc = channels["T2_axial"], channels["ADC"]
        for field in ("study_uid", "frame_uid"):
            if t2[field] != adc[field]:
                raise ValueError(f"Cross-series {field} mismatch: {patient}")
        expected_rows = [
            {"fid": row["fid"], "pos_mm": row["pos"].split(), "ClinSig": row["ClinSig"]}
            for row in tables["ProstateX-Findings-Train.csv"] if row["ProxID"] == patient
        ]
        expected = finding_map(patient, expected_rows)
        if finding_map(patient, t2["findings"]) != expected or finding_map(patient, adc["findings"]) != expected:
            raise ValueError(f"Finding coverage or label mismatch: {patient}")
        positive = sum(label == "TRUE" for label in expected.values())
        summary = {
            "patient": patient, "findings": len(expected), "ClinSig_TRUE": positive,
            "ClinSig_FALSE": len(expected) - positive,
            "full_finding_keys_and_labels_match": True,
            "repeated_fids_retained": sorted({key[1] for key in expected if sum(other[1] == key[1] for other in expected) > 1}),
        }
        summaries.append(summary)
        print(patient, "findings:", len(expected), "TRUE:", positive, "FALSE:", len(expected) - positive)
        for channel, report in channels.items():
            print(" ", channel, "slices:", report["dicom_files"], "spacing mm:", round(report["slice_spacing_mm"], 6))
    result = {
        "scope": "Four-patient geometry and full finding-key checks with the existing limited reader; not independent clinical validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        "labels_zip_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "label_csv_sha256": csv_hashes,
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "strict_tolerances": {"rtol": 0, "orientation_and_spacing_atol": 1e-6, "slice_step_atol_mm": 1e-4},
        "summaries": summaries, "series_by_patient": by_patient,
        "not_performed": ["patch generation or patch bounds checks", "registration", "ADC unit validation", "normalization", "training"],
    }
    print("Geometry and finding checks passed; no patches or training.")
    return result


if __name__ == "__main__":
    required = ("audit", "select_remaining", "remaining_bundle", "LABEL_ZIP")
    if any(name not in globals() for name in required):
        raise RuntimeError("Run the corrected pilot and Step 5 cells first")
    geometry_result = audit_remaining_geometry(remaining_bundle, LABEL_ZIP)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    geometry_path = Path(f"remaining_four_geometry_{stamp}.json")
    geometry_path.write_text(json.dumps(geometry_result, indent=2, allow_nan=False) + "\n")
    from IPython.display import FileLink, display
    display(FileLink(str(geometry_path)))