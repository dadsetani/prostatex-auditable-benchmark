"""Freeze the post-QC0025 protocol and resolved cohort download plan."""

import csv
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path


PROTOCOL_V2 = Path("analysis/prostatex_selection_protocol_v0_2.json")
FOLDS_V2 = Path("analysis/prostatex_patient_folds_v0_2.csv")
PLAN_V2 = Path("analysis/prostatex_download_plan_v0_2.csv")
QC0025_AUDIT = Path("analysis/prostatex_0025_multistudy_audit.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    protocol = json.loads(PROTOCOL_V2.read_text(encoding="utf-8"))
    audit = json.loads(QC0025_AUDIT.read_text(encoding="utf-8"))
    if audit["reported_status"] != "COMPLETE_UNIQUE_ASSIGNMENTS":
        raise ValueError("QC0025 assignments are not complete")

    assignments = audit["finding_assignments"]
    if len(assignments) != 5 or len({row["full_key"] for row in assignments}) != 5:
        raise ValueError("Unexpected QC0025 assignment set")

    folds = read_csv(FOLDS_V2)
    matched = [row for row in folds if row["patient"] == "ProstateX-0025"]
    if len(matched) != 1 or matched[0]["outer_fold"] != "2":
        raise ValueError("Unexpected QC0025 fold")
    matched[0]["special_handling"] = "MULTI_STUDY_FINDING_ASSIGNMENT_RESOLVED"
    folds_path = Path("analysis/prostatex_patient_folds_v0_3.csv")
    write_csv(folds_path, folds)

    plan = read_csv(PLAN_V2)
    qc_rows = [row for row in plan if row["patient"] == "ProstateX-0025"]
    if len(qc_rows) != 4 or len({row["study_uid"] for row in qc_rows}) != 2:
        raise ValueError("Unexpected QC0025 plan rows")
    assigned_studies = {row["assigned_study_uid"] for row in assignments}
    if assigned_studies != {row["study_uid"] for row in qc_rows}:
        raise ValueError("Assigned studies differ from download plan")
    for row in qc_rows:
        row["download_phase"] = "FOLD_2_PRIMARY_AFTER_EXCEPTION_RESOLVED"
        row["pair_role"] = "FINDING_ASSIGNED_PRIMARY_PAIR"
    plan_path = Path("analysis/prostatex_download_plan_v0_3.csv")
    write_csv(plan_path, plan)

    protocol.update({
        "protocol_id": "PROSTATEx-selection-v0.3",
        "status": "FROZEN_FOR_BATCHED_IMAGE_RETRIEVAL",
        "amended_date": date.today().isoformat(),
        "supersedes": {"protocol_id": "PROSTATEx-selection-v0.2",
                       "sha256": digest(PROTOCOL_V2)},
        "amendment_timing": "Written after resolving the ProstateX-0025 multi-study exception and before bulk cohort image retrieval, full patch extraction, or model training.",
        "multi_study_resolution": {
            "patient": "ProstateX-0025",
            "evidence_bundle_sha256": audit["uploaded_bundle_sha256"],
            "status": audit["reported_status"],
            "rule": "Use the study assigned by complete paired-grid pixel-center bounds for each full finding key; do not apply a cross-study UID tie-break.",
            "finding_assignments": assignments,
            "patient_fold": 2,
            "interpretation_limit": "Geometry-only source assignment; not anatomical registration, image-quality, label, or clinical validation."
        },
        "batched_retrieval": {
            "unit": "One frozen outer fold per output bundle",
            "folds": 5,
            "planned_unique_series": len({row["series_uid"] for row in plan}),
            "planned_rows": len(plan),
            "resume_rule": "Within one Kaggle session, reuse only files whose recorded series UID, ZIP CRC, DICOM count, and archive SHA-256 pass validation.",
            "failure_rule": "Preserve and return a FAILED or REVIEW_REQUIRED report; do not silently skip, substitute, or move a patient between folds.",
            "post_download_rule": "Audit identities, encoding, physical geometry, paired FrameOfReferenceUID, intensity semantics, and complete per-finding T2-parallel grid bounds before patch extraction."
        },
    })
    protocol.pop("multi_study_rule", None)
    protocol["patient_folds"]["pending_multi_study_patient"] = []
    protocol["patient_folds"]["resolved_multi_study_patient"] = ["ProstateX-0025"]
    protocol["ready_before_download"] = {
        "patients": 197,
        "findings": 319,
        "ClinSig_TRUE": 72,
        "ClinSig_FALSE": 247,
        "planned_unique_series": len({row["series_uid"] for row in plan}),
        "excluded_development_patients": 6,
        "excluded_no_canonical_T2": ["ProstateX-0191"]
    }
    protocol_path = Path("analysis/prostatex_selection_protocol_v0_3.json")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    fold_counts = {}
    for fold in range(5):
        rows = [row for row in plan if int(row["outer_fold"]) == fold]
        fold_counts[str(fold)] = {
            "patients": len({row["patient"] for row in rows}),
            "series": len(rows),
            "reported_DICOM_instances": sum(int(row["image_count"]) for row in rows),
        }
    plan_summary = {
        "protocol_id": protocol["protocol_id"],
        "status": "READY_FOR_BATCHED_IMAGE_RETRIEVAL",
        "patients": len({row["patient"] for row in plan}),
        "findings": 319,
        "unique_series": len({row["series_uid"] for row in plan}),
        "series_rows": len(plan),
        "folds": fold_counts,
        "resolved_exception": {"patient": "ProstateX-0025", "series": 4,
                               "studies": 2, "finding_assignments": 5},
        "excluded": {"development_QC_patients": [f"ProstateX-{index:04d}" for index in range(6)],
                     "no_canonical_T2": ["ProstateX-0191"]},
    }
    plan_json_path = Path("analysis/prostatex_download_plan_v0_3.json")
    plan_json_path.write_text(json.dumps(plan_summary, indent=2) + "\n", encoding="utf-8")

    folds_json_path = Path("analysis/prostatex_patient_folds_v0_3.json")
    folds_json_path.write_text(json.dumps({
        "protocol_id": protocol["protocol_id"],
        "patients": len(folds),
        "folds": {str(fold): [row["patient"] for row in folds if int(row["outer_fold"]) == fold]
                  for fold in range(5)},
        "resolved_special_handling": {"ProstateX-0025": "Finding-specific study assignment complete"},
    }, indent=2) + "\n", encoding="utf-8")

    markdown_path = Path("analysis/prostatex_selection_protocol_v0_3.md")
    markdown_path.write_text(
        "# PROSTATEx data-selection protocol v0.3\n\n"
        "Status: **frozen for batched image retrieval** on 2026-09-26.\n\n"
        "The ProstateX-0025 exception is resolved by full-grid geometry. One finding maps to "
        "study `1.3.6.1.4.1.14519.5.2.1.7311.5101.101137934734515856187339619047`; "
        "the other four map to study `1.3.6.1.4.1.14519.5.2.1.7311.5101.260163101010991858039197694584`. "
        "Both study pairs remain primary sources for this patient, which stays in outer fold 2.\n\n"
        "The primary cohort contains 197 patients and 319 findings after excluding six development-QC "
        "patients and ProstateX-0191 from the paired primary analysis. The retrieval plan contains 396 "
        "series and is executed one frozen outer fold at a time. Post-download identity, encoding, geometry, "
        "intensity-semantic, frame, and complete-grid bounds checks are mandatory before patch extraction.\n",
        encoding="utf-8")

    artifacts = [protocol_path, markdown_path, folds_path, folds_json_path, plan_path, plan_json_path]
    manifest_path = Path("analysis/prostatex_selection_protocol_v0_3_manifest.json")
    manifest_path.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__)),
        "generator_sha256": digest(Path(__file__)),
        "source_sha256": {str(path): digest(path) for path in
                          (PROTOCOL_V2, FOLDS_V2, PLAN_V2, QC0025_AUDIT)},
        "artifacts": {str(path): {"sha256": digest(path), "bytes": path.stat().st_size}
                      for path in artifacts},
    }, indent=2) + "\n", encoding="utf-8")
    print("Protocol:", protocol_path)
    print("Plan rows/series:", len(plan), len({row["series_uid"] for row in plan}))
    print("Fold counts:", fold_counts)


if __name__ == "__main__":
    main()