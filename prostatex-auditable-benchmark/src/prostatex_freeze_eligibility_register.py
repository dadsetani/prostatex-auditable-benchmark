"""Freeze the post-retrieval PROSTATEx finding eligibility register."""

import csv
import hashlib
import json
import tempfile
from pathlib import Path
from zipfile import ZipFile


PROTOCOL_SHA256 = "45a423324fab7f4e2971743ce2f5fd5961b7976e9745bb7567366bf89a889b08"
EXPECTED = {
    "patients": 197,
    "findings": 319,
    "accepted_patients": 191,
    "accepted_findings": 306,
    "accepted_true": 65,
    "accepted_false": 241,
}
FOLD_INPUTS = {
    0: "fold0_manifest.json",
    1: "fold1_final_manifest.json",
    2: "fold2_manifest.json",
    3: "fold3_manifest.json",
    4: "fold4_manifest.json",
}
EXCLUDED_PATIENTS = {
    "ProstateX-0038": "ADC geometry failed: irregular slice-step vectors caused by a missing physical plane",
    "ProstateX-0199": "T2/ADC FrameOfReferenceUID mismatch",
    "ProstateX-0200": "T2/ADC FrameOfReferenceUID mismatch",
    "ProstateX-0201": "T2/ADC FrameOfReferenceUID mismatch",
    "ProstateX-0202": "T2/ADC FrameOfReferenceUID mismatch",
    "ProstateX-0203": "T2/ADC FrameOfReferenceUID mismatch",
}
FINDING_EXCLUSIONS = {
    "ProstateX-0189|3|8.35175 26.1661 -23.7111": (
        "Complete T2-parallel grid lies approximately 1.00 mm before the first ADC pixel-center plane"
    )
}
NUMERICAL_ACCEPTANCES = {
    "ProstateX-0154|3|-42.3586 50.6483 73.0089": (
        "Accepted after numerical-boundary review: ADC overshoot 1.566e-5 voxel is below 1e-4-voxel tolerance"
    )
}
CSV_FIELDS = [
    "full_key", "patient", "fid", "pos_text", "pos_x_mm", "pos_y_mm", "pos_z_mm",
    "zone", "ClinSig", "label", "outer_fold", "eligibility", "decision_reason",
    "reported_finding_status", "assigned_study_uid", "t2_series_uid", "adc_series_uid",
    "t2_archive_name", "adc_archive_name",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row[field] for field in CSV_FIELDS} for row in rows)


def freeze_register(upload_dir="prism-uploads", analysis_dir="analysis"):
    upload_dir = Path(upload_dir)
    analysis_dir = Path(analysis_dir)
    protocol_path = analysis_dir / "prostatex_selection_protocol_v0_3.json"
    require(sha256_path(protocol_path) == PROTOCOL_SHA256, "Unexpected protocol hash")
    folds = read_json(analysis_dir / "prostatex_patient_folds_v0_3.json")["folds"]
    expected_fold_by_patient = {
        patient: int(fold) for fold, patients in folds.items() for patient in patients
    }
    require(len(expected_fold_by_patient) == EXPECTED["patients"], "Unexpected patient fold register")

    rows = []
    bundle_sources = []
    with tempfile.TemporaryDirectory(prefix="prostatex-register-") as temporary_dir:
        for outer_fold, manifest_name in FOLD_INPUTS.items():
            bundle, manifest = verified_fold_bundle(upload_dir, manifest_name, temporary_dir)
            bundle_sources.append({
                "outer_fold": outer_fold,
                "manifest": f"{upload_dir.name}/{manifest_name}",
                "bundle_filename": manifest["original_filename"],
                "bundle_bytes": manifest["original_bytes"],
                "bundle_sha256": manifest["original_sha256"],
            })
            with ZipFile(bundle) as archive:
                require(archive.testzip() is None, f"Fold {outer_fold} ZIP CRC failure")
                names = archive.namelist()
                require(len(names) == len(set(names)), f"Fold {outer_fold} duplicate ZIP members")
                report = json.loads(archive.read("report.json"))
            require(report["protocol_id"] == "PROSTATEx-selection-v0.3", "Protocol mismatch")
            require(int(report["outer_fold"]) == outer_fold, "Outer-fold mismatch")

            series_by_study_channel = {}
            for series in report["series"]:
                key = (series["patient"], series["study_uid"], series["channel"])
                require(key not in series_by_study_channel, f"Duplicate selected channel: {key}")
                series_by_study_channel[key] = series

            for finding in report["findings"]:
                patient = finding["patient"]
                full_key = finding["full_key"]
                require(expected_fold_by_patient[patient] == outer_fold, "Frozen fold assignment changed")
                if patient in EXCLUDED_PATIENTS:
                    eligibility = "EXCLUDED_PATIENT"
                    reason = EXCLUDED_PATIENTS[patient]
                elif full_key in FINDING_EXCLUSIONS:
                    eligibility = "EXCLUDED_FINDING"
                    reason = FINDING_EXCLUSIONS[full_key]
                elif full_key in NUMERICAL_ACCEPTANCES:
                    eligibility = "ACCEPTED"
                    reason = NUMERICAL_ACCEPTANCES[full_key]
                else:
                    require(finding["status"] == "PASSED", f"Unresolved finding status: {full_key}")
                    eligibility = "ACCEPTED"
                    reason = "Passed frozen post-download identity, geometry, frame, and complete-grid checks"

                study_uid = finding.get("assigned_study_uid") or ""
                selected = {}
                if eligibility != "EXCLUDED_PATIENT":
                    require(study_uid, f"Accepted finding lacks assigned study: {full_key}")
                    for channel in ("T2_axial", "ADC"):
                        key = (patient, study_uid, channel)
                        require(key in series_by_study_channel, f"Missing selected series: {key}")
                        selected[channel] = series_by_study_channel[key]
                position = finding["pos_mm"]
                rows.append({
                    "full_key": full_key,
                    "patient": patient,
                    "fid": finding["fid"],
                    "pos_text": finding["pos_text"],
                    "pos_mm": position,
                    "pos_x_mm": position[0],
                    "pos_y_mm": position[1],
                    "pos_z_mm": position[2],
                    "zone": finding["zone"],
                    "ClinSig": finding["ClinSig"].upper(),
                    "label": int(finding["ClinSig"].upper() == "TRUE"),
                    "outer_fold": outer_fold,
                    "eligibility": eligibility,
                    "decision_reason": reason,
                    "reported_finding_status": finding["status"],
                    "assigned_study_uid": study_uid,
                    "t2_series_uid": selected.get("T2_axial", {}).get("series_uid", ""),
                    "adc_series_uid": selected.get("ADC", {}).get("series_uid", ""),
                    "t2_archive_name": selected.get("T2_axial", {}).get("archive_name", ""),
                    "adc_archive_name": selected.get("ADC", {}).get("archive_name", ""),
                    "reported_bounds": finding.get("bounds"),
                })

    rows.sort(key=lambda row: (row["patient"], int(row["fid"]), tuple(row["pos_mm"])))
    require(len(rows) == EXPECTED["findings"], "Unexpected finding count")
    require(len({row["full_key"] for row in rows}) == len(rows), "Duplicate full finding key")
    accepted = [row for row in rows if row["eligibility"] == "ACCEPTED"]
    accepted_patients = {row["patient"] for row in accepted}
    require(len(accepted) == EXPECTED["accepted_findings"], "Unexpected accepted finding count")
    require(len(accepted_patients) == EXPECTED["accepted_patients"], "Unexpected accepted patient count")
    require(sum(row["label"] for row in accepted) == EXPECTED["accepted_true"], "Unexpected TRUE count")
    require(sum(not row["label"] for row in accepted) == EXPECTED["accepted_false"], "Unexpected FALSE count")
    require({row["patient"] for row in rows if row["eligibility"] == "EXCLUDED_PATIENT"}
            == set(EXCLUDED_PATIENTS), "Excluded patient set changed")

    register = {
        "register_id": "PROSTATEx-post-download-eligibility-v0.3",
        "frozen_date": "2026-09-27",
        "protocol_id": "PROSTATEx-selection-v0.3",
        "protocol_sha256": PROTOCOL_SHA256,
        "scope": "Finding-level technical eligibility after five-fold retrieval; no patch extraction, image-quality review, modeling, or clinical validation",
        "finding_key": ["patient", "fid", "pos_mm"],
        "fold_assignment_preserved": True,
        "replacement_patients_added": 0,
        "summary": {
            "represented_patients": len({row["patient"] for row in rows}),
            "represented_findings": len(rows),
            "accepted_patients": len(accepted_patients),
            "accepted_findings": len(accepted),
            "accepted_ClinSig_TRUE": sum(row["label"] for row in accepted),
            "accepted_ClinSig_FALSE": sum(not row["label"] for row in accepted),
            "excluded_patients": len(EXCLUDED_PATIENTS),
            "excluded_findings": len(rows) - len(accepted),
        },
        "bundle_sources": bundle_sources,
        "exclusion_policy": {
            "excluded_patients": EXCLUDED_PATIENTS,
            "finding_level_exclusions": FINDING_EXCLUSIONS,
            "numerical_boundary_acceptances": NUMERICAL_ACCEPTANCES,
        },
        "rows": rows,
    }

    json_path = analysis_dir / "prostatex_eligibility_register_v0_3.json"
    csv_path = analysis_dir / "prostatex_eligibility_register_v0_3.csv"
    accepted_path = analysis_dir / "prostatex_accepted_findings_v0_3.csv"
    json_path.write_text(json.dumps(register, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)
    write_csv(accepted_path, accepted)
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_path(path)}
        for path in (json_path, csv_path, accepted_path)
    }
    manifest_path = analysis_dir / "prostatex_eligibility_register_v0_3_manifest.json"
    manifest_path.write_text(json.dumps({
        "register_id": register["register_id"],
        "frozen_date": register["frozen_date"],
        "artifacts": artifacts,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(register["summary"], indent=2))
    print("Saved:", json_path, csv_path, accepted_path, manifest_path)


if __name__ == "__main__":
    freeze_register()