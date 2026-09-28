"""Audit the full metadata bundle, amend selection rules, and freeze patient folds."""

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile


BUNDLE = Path("prism-uploads/prostatex_full_metadata_20260917_121401_877544.zip")
LABEL_ZIP = Path("prism-uploads/ProstateX-TrainingLesionInformationv2.zip")
MANIFEST = Path("prism-uploads/PROSTATEx-train.tcia")
PROTOCOL_V1 = Path("analysis/prostatex_selection_protocol_v0_1.json")
LOCAL_PREINVENTORY = Path("analysis/prostatex_full_preinventory.csv")
FOLD_SALT = "PROSTATEx-selection-v0.2|outer-folds|2026-09-17"
PILOT_PATIENTS = {f"ProstateX-{index:04d}" for index in range(6)}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def csv_rows(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_and_audit_bundle():
    raw = BUNDLE.read_bytes()
    with ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        expected = {"api_getSeries_PROSTATEx_MR.json", "artifact_hashes.json",
                    "inventory_report.json", "pair_alternatives.csv",
                    "patient_pair_selection.csv", "preinventory.csv", "series_candidates.csv"}
        if len(names) != len(set(names)) or set(names) != expected or archive.testzip() is not None:
            raise ValueError("Unexpected members, duplicate names, or CRC failure")
        if sum(entry.file_size for entry in archive.infolist()) > 32 * 1024**2:
            raise ValueError("Bundle exceeds metadata size limit")
        hashes = json.loads(archive.read("artifact_hashes.json"))
        if set(hashes) != expected - {"artifact_hashes.json"}:
            raise ValueError("Incomplete artifact hash manifest")
        for name, item in hashes.items():
            content = archive.read(name)
            if len(content) != item["bytes"] or sha(content) != item["sha256"]:
                raise ValueError("Artifact mismatch: " + name)
        report = json.loads(archive.read("inventory_report.json"))
        patients = csv_rows(archive.read("patient_pair_selection.csv"))
        alternatives = csv_rows(archive.read("pair_alternatives.csv"))
        candidates = csv_rows(archive.read("series_candidates.csv"))
        preinventory = csv_rows(archive.read("preinventory.csv"))
        api = json.loads(archive.read("api_getSeries_PROSTATEx_MR.json"))
    if report["status"] != "COMPLETE" or report["protocol_id"] != "PROSTATEx-selection-v0.1":
        raise ValueError("Inventory did not complete under protocol v0.1")
    if len(patients) != len({row["patient"] for row in patients}) or len(patients) != 204:
        raise ValueError("Unexpected patient selection rows")
    if len(preinventory) != 204 or len(candidates) != 464 or len(alternatives) != 55:
        raise ValueError("Unexpected metadata table sizes")
    if preinventory != csv_rows(LOCAL_PREINVENTORY.read_bytes()):
        raise ValueError("Local/Kaggle preinventory content mismatch")
    if len(api) != 18321 or len({row["SeriesInstanceUID"] for row in api}) != len(api):
        raise ValueError("Unexpected API snapshot or duplicate series UIDs")
    if Counter(row["status"] for row in candidates) != {"ELIGIBLE_METADATA": 464}:
        raise ValueError("Unexpected candidate status")
    protocol_hash = sha(PROTOCOL_V1.read_bytes())
    manifest_hash = sha(MANIFEST.read_bytes())
    if report["input_sha256"]["protocol"] != protocol_hash:
        raise ValueError("Protocol hash mismatch")
    if report["input_sha256"]["manifest"] != manifest_hash:
        raise ValueError("Manifest hash mismatch")
    with ZipFile(LABEL_ZIP) as archive:
        for basename in ("ProstateX-Findings-Train.csv", "ProstateX-Images-Train.csv"):
            names = [name for name in archive.namelist() if Path(name).name == basename]
            if len(names) != 1 or sha(archive.read(names[0])) != report["input_sha256"][basename]:
                raise ValueError("Local/Kaggle CSV content mismatch: " + basename)
    eligible = {(row["patient"], row["channel"], row["series_uid"])
                for row in candidates if row["status"] == "ELIGIBLE_METADATA"}
    for row in patients:
        if row["primary_t2_series_uid"]:
            if ((row["patient"], "T2_axial", row["primary_t2_series_uid"]) not in eligible
                    or (row["patient"], "ADC", row["primary_adc_series_uid"]) not in eligible):
                raise ValueError("Primary pair is not eligible metadata")
    for row in alternatives:
        if ((row["patient"], "T2_axial", row["t2_series_uid"]) not in eligible
                or (row["patient"], "ADC", row["adc_series_uid"]) not in eligible):
            raise ValueError("Alternate pair is not eligible metadata")
    return raw, report, patients, alternatives, candidates, preinventory, api


def identify_anomalies(patients, alternatives, candidates):
    studies = defaultdict(set)
    for row in patients:
        if row["study_uid"]:
            studies[row["patient"]].add(row["study_uid"])
    for row in alternatives:
        studies[row["patient"]].add(row["study_uid"])
    multi_study = sorted(patient for patient, values in studies.items() if len(values) > 1)
    api_ambiguous = sorted({row["patient"] for row in candidates
                            if int(row["api_match_count"]) > 1
                            or int(row["manifest_match_count"]) > 1})
    no_primary = [row["patient"] for row in patients
                  if not row["primary_t2_series_uid"] or not row["primary_adc_series_uid"]]
    if multi_study != ["ProstateX-0025"] or api_ambiguous != ["ProstateX-0025"]:
        raise ValueError("Unexpected multi-study/API ambiguity")
    if no_primary != ["ProstateX-0191"]:
        raise ValueError("Unexpected patient without primary pair")
    row_25 = next(row for row in patients if row["patient"] == "ProstateX-0025")
    row_191 = next(row for row in patients if row["patient"] == "ProstateX-0191")
    if (int(row_25["finding_count"]), int(row_25["ClinSig_TRUE"]), int(row_25["ClinSig_FALSE"])) != (5, 0, 5):
        raise ValueError("Unexpected ProstateX-0025 findings")
    if (int(row_191["finding_count"]), int(row_191["ClinSig_TRUE"]), int(row_191["ClinSig_FALSE"])) != (1, 0, 1):
        raise ValueError("Unexpected ProstateX-0191 findings")
    return row_25, row_191


def assign_folds(patients):
    selected = [row for row in patients if row["modeling_role"] == "COHORT_CANDIDATE"
                and row["primary_t2_series_uid"]]
    if len(selected) != 197:
        raise ValueError("Expected 197 metadata-eligible nonpilot patients")

    def stable_hash(patient):
        return hashlib.sha256((FOLD_SALT + "|" + patient).encode()).hexdigest()

    positive = sorted((row for row in selected if int(row["ClinSig_TRUE"]) > 0),
                      key=lambda row: (-int(row["ClinSig_TRUE"]), -int(row["finding_count"]),
                                       stable_hash(row["patient"])))
    negative = sorted((row for row in selected if int(row["ClinSig_TRUE"]) == 0),
                      key=lambda row: (-int(row["finding_count"]), stable_hash(row["patient"])))
    folds = [{"rows": [], "patients": 0, "positive_patients": 0,
              "ClinSig_TRUE": 0, "ClinSig_FALSE": 0, "findings": 0} for _ in range(5)]

    def add(row, index):
        fold = folds[index]
        fold["rows"].append(row)
        fold["patients"] += 1
        fold["positive_patients"] += int(int(row["ClinSig_TRUE"]) > 0)
        fold["ClinSig_TRUE"] += int(row["ClinSig_TRUE"])
        fold["ClinSig_FALSE"] += int(row["ClinSig_FALSE"])
        fold["findings"] += int(row["finding_count"])

    for row in positive:
        eligible = [index for index, fold in enumerate(folds) if fold["patients"] < 40]
        index = min(eligible, key=lambda value: (folds[value]["positive_patients"],
                    folds[value]["ClinSig_TRUE"], folds[value]["findings"],
                    folds[value]["patients"], value))
        add(row, index)
    for row in negative:
        eligible = [index for index, fold in enumerate(folds) if fold["patients"] < 40]
        index = min(eligible, key=lambda value: (folds[value]["patients"],
                    folds[value]["ClinSig_FALSE"], folds[value]["findings"], value))
        add(row, index)

    rows = []
    for index, fold in enumerate(folds):
        for source in sorted(fold["rows"], key=lambda row: row["patient"]):
            special = ("MULTI_STUDY_FINDING_ASSIGNMENT_PENDING"
                       if source["patient"] == "ProstateX-0025" else "NONE")
            rows.append({"patient": source["patient"], "outer_fold": index,
                "finding_count": int(source["finding_count"]),
                "ClinSig_TRUE": int(source["ClinSig_TRUE"]),
                "ClinSig_FALSE": int(source["ClinSig_FALSE"]),
                "has_ClinSig_TRUE": int(int(source["ClinSig_TRUE"]) > 0),
                "metadata_selection_status": source["selection_status"],
                "special_handling": special})
    summary = [{key: value for key, value in fold.items() if key != "rows"}
               | {"outer_fold": index} for index, fold in enumerate(folds)]
    if sorted(row["patient"] for row in rows) != sorted(row["patient"] for row in selected):
        raise ValueError("Fold membership mismatch")
    if max(row["patients"] for row in summary) - min(row["patients"] for row in summary) > 1:
        raise ValueError("Unbalanced patient counts")
    if max(row["positive_patients"] for row in summary) - min(row["positive_patients"] for row in summary) > 1:
        raise ValueError("Unbalanced positive-patient counts")
    return rows, summary


def build_download_plan(patients, alternatives, candidates, folds):
    candidate_by_uid = {row["series_uid"]: row for row in candidates if row["series_uid"]}
    if len(candidate_by_uid) != len([row for row in candidates if row["series_uid"]]):
        raise ValueError("Duplicate candidate series UID")
    fold_by_patient = {row["patient"]: row["outer_fold"] for row in folds}
    output = []
    initial_uids = set()
    for patient_row in patients:
        patient = patient_row["patient"]
        if patient in PILOT_PATIENTS or patient == "ProstateX-0191":
            continue
        if not patient_row["primary_t2_series_uid"] or patient not in fold_by_patient:
            raise ValueError("Missing primary pair or fold for download candidate")
        pairs = [{"rank": 1, "study_uid": patient_row["study_uid"],
                  "T2_axial": patient_row["primary_t2_series_uid"],
                  "ADC": patient_row["primary_adc_series_uid"]}]
        if patient == "ProstateX-0025":
            for row in alternatives:
                if row["patient"] == patient:
                    pairs.append({"rank": int(row["rank"]), "study_uid": row["study_uid"],
                                  "T2_axial": row["t2_series_uid"], "ADC": row["adc_series_uid"]})
        for pair in pairs:
            role = ("MULTI_STUDY_ASSIGNMENT_CANDIDATE" if patient == "ProstateX-0025"
                    else "PRIMARY_METADATA_PAIR")
            phase = ("EXCEPTION_RESOLUTION_FIRST" if patient == "ProstateX-0025"
                     else f"FOLD_{fold_by_patient[patient]}_PRIMARY_AFTER_EXCEPTION")
            for channel in ("T2_axial", "ADC"):
                uid = pair[channel]
                details = candidate_by_uid[uid]
                if details["patient"] != patient or details["channel"] != channel:
                    raise ValueError("Download-plan candidate identity mismatch")
                initial_uids.add(uid)
                output.append({"patient": patient, "outer_fold": fold_by_patient[patient],
                    "download_phase": phase, "pair_rank": pair["rank"], "pair_role": role,
                    "study_uid": pair["study_uid"], "channel": channel,
                    "series_number": int(details["series_number"]), "series_uid": uid,
                    "description": details["description"], "image_count": int(details["image_count"])})
    if len(output) != len(initial_uids) or len(output) != 396:
        raise ValueError("Unexpected initial download-plan series count")
    deferred_pairs = [row for row in alternatives
                      if row["patient"] not in PILOT_PATIENTS and row["patient"] != "ProstateX-0025"]
    deferred_uids = {uid for row in deferred_pairs
                     for uid in (row["t2_series_uid"], row["adc_series_uid"])} - initial_uids
    summary = {
        "protocol_id": "PROSTATEx-selection-v0.2",
        "scope": "Initial image-retrieval plan only; no download performed",
        "patients": len({row["patient"] for row in output}),
        "unique_series": len(output),
        "T2_axial_series": sum(row["channel"] == "T2_axial" for row in output),
        "ADC_series": sum(row["channel"] == "ADC" for row in output),
        "reported_DICOM_instances": sum(row["image_count"] for row in output),
        "exception_first": {"patient": "ProstateX-0025", "series": 4, "study_pairs": 2},
        "fold_series_counts": {str(index): sum(row["outer_fold"] == index for row in output)
                               for index in range(5)},
        "deferred_same_study_sensitivity_pair_rows": len(deferred_pairs),
        "deferred_additional_unique_series": len(deferred_uids),
        "excluded": {"development_QC_patients": sorted(PILOT_PATIENTS),
                     "no_canonical_T2": ["ProstateX-0191"]},
    }
    return output, summary


def main():
    bundle_raw, report, patients, alternatives, candidates, preinventory, api = load_and_audit_bundle()
    row_25, row_191 = identify_anomalies(patients, alternatives, candidates)
    folds, fold_summary = assign_folds(patients)
    download_rows, download_summary = build_download_plan(patients, alternatives, candidates, folds)
    protocol_v1 = json.loads(PROTOCOL_V1.read_text())
    protocol_v2 = {
        **protocol_v1,
        "protocol_id": "PROSTATEx-selection-v0.2",
        "status": "FROZEN_FOR_DOWNLOAD_PLANNING",
        "amended_date": "2026-09-17",
        "supersedes": {"protocol_id": protocol_v1["protocol_id"],
                       "sha256": sha(PROTOCOL_V1.read_bytes())},
        "amendment_timing": "Written after the complete metadata inventory and before bulk image retrieval, fold use, full patch extraction, or model training.",
        "metadata_inventory": {"bundle_sha256": sha(bundle_raw),
            "api_snapshot_sha256": report["api_snapshot_sha256"],
            "api_retrieval_completed_utc": report["completed_utc"]},
        "multi_study_rule": {
            "patient": "ProstateX-0025",
            "reason": "Two same-number/same-description T2/ADC pairs map to two different StudyInstanceUID values; the findings CSV does not directly identify a study.",
            "action": "Do not use the v0.1 UID tie-break across studies. Retrieve and audit both study pairs. Assign each full finding key to a study only if exactly one pair supports the complete T2 and ADC sampling grid inside pixel-center bounds. Keep all studies and findings from the patient in one fold.",
            "ambiguity_outcome": "If zero or multiple study pairs remain valid for a finding, exclude that finding from the paired primary analysis and log it; do not choose using labels, visual appearance, or model performance.",
        },
        "noncanonical_T2_rule": {
            "patient": "ProstateX-0191",
            "reason": "No exact t2_tse_tra candidate exists under the frozen canonical rule.",
            "action": "Exclude from the paired primary cohort and primary modality comparisons. Retain ADC eligibility only for a separately labeled supplementary analysis. Do not introduce t2_tse_tra_Grappa3 as an ad hoc exception.",
        },
        "patient_folds": {
            "eligible_patients": 197, "excluded_development_patients": sorted(PILOT_PATIENTS),
            "paired_primary_exclusion": ["ProstateX-0191"],
            "pending_multi_study_patient": ["ProstateX-0025"],
            "fold_count": 5, "salt": FOLD_SALT,
            "assignment": "Deterministic greedy allocation: positive patients first by descending TRUE/finding counts and salted hash, minimizing positive-patient, TRUE, finding, and patient counts; negative-only patients next by descending finding count and salted hash, minimizing patient, FALSE, and finding counts.",
            "evaluation_cycle": "For run k, fold k is test, fold (k+1) mod 5 is validation, and the other three folds are training. Augmentation and all learned preprocessing fit training patients only; GA/model selection uses validation only; test labels remain untouched until final scoring.",
            "failure_policy": "Post-download exclusions remain in their assigned fold and are not replaced or moved.",
        },
        "ready_before_download": {"patients_without_special_handling": 196,
            "findings_without_special_handling": 314, "ClinSig_TRUE": 72, "ClinSig_FALSE": 242,
            "additional_pending_patient": "ProstateX-0025 with five FALSE findings"},
    }
    protocol_path = Path("analysis/prostatex_selection_protocol_v0_2.json")
    protocol_path.write_text(json.dumps(protocol_v2, indent=2) + "\n")
    folds_path = Path("analysis/prostatex_patient_folds_v0_2.csv")
    write_csv(folds_path, folds)
    fold_report = {"protocol_id": protocol_v2["protocol_id"], "salt": FOLD_SALT,
                   "evaluation_cycle": protocol_v2["patient_folds"]["evaluation_cycle"],
                   "folds": fold_summary,
                   "ProstateX-0025_fold": next(row["outer_fold"] for row in folds
                                                if row["patient"] == "ProstateX-0025")}
    fold_path = Path("analysis/prostatex_patient_folds_v0_2.json")
    fold_path.write_text(json.dumps(fold_report, indent=2) + "\n")
    download_path = Path("analysis/prostatex_download_plan_v0_2.csv")
    write_csv(download_path, download_rows)
    download_report_path = Path("analysis/prostatex_download_plan_v0_2.json")
    download_report_path.write_text(json.dumps(download_summary, indent=2) + "\n")
    audit = {
        "scope": "Local integrity and structural audit of the uploaded full metadata inventory; no image retrieval or clinical validation",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_path": str(BUNDLE), "bundle_sha256": sha(bundle_raw), "archive_crc_pass": True,
        "artifact_hashes_and_sizes_match": True, "api_http_status": report["retrieval"]["http_status"],
        "api_server_date": report["retrieval"]["server_date"],
        "api_snapshot_sha256": report["api_snapshot_sha256"],
        "api_series_count": len(api), "api_patient_count": len({row["PatientID"] for row in api}),
        "api_series_uids_unique": True, "labeled_patients": 204, "findings": 330,
        "candidate_rows": len(candidates), "all_candidate_rows_eligible_metadata": True,
        "primary_pair_rows": sum(bool(row["primary_t2_series_uid"]) for row in patients),
        "alternate_pair_rows": len(alternatives),
        "patients_with_alternates": len({row["patient"] for row in alternatives}),
        "local_manifest_hash_matches": True, "local_protocol_v0_1_hash_matches": True,
        "label_zip_container_hash_matches": report["input_sha256"]["label_zip"] == sha(LABEL_ZIP.read_bytes()),
        "label_csv_content_hashes_match": True,
        "preinventory_rows_content_equal_after_line_ending_normalization": True,
        "exceptions": {
            "ProstateX-0025": {"status": "MULTI_STUDY_FINDING_ASSIGNMENT_REQUIRED",
                "findings": int(row_25["finding_count"]), "ClinSig_TRUE": int(row_25["ClinSig_TRUE"]),
                "ClinSig_FALSE": int(row_25["ClinSig_FALSE"]), "eligible_study_pairs": 2},
            "ProstateX-0191": {"status": "PAIRED_PRIMARY_EXCLUDED_NO_CANONICAL_T2",
                "findings": int(row_191["finding_count"]), "ClinSig_TRUE": int(row_191["ClinSig_TRUE"]),
                "ClinSig_FALSE": int(row_191["ClinSig_FALSE"]), "eligible_ADC_series": int(row_191["eligible_adc_series"])},
        },
        "protocol_amendment": str(protocol_path), "fold_manifest": str(folds_path),
        "download_plan": str(download_path),
        "limitations": ["Metadata matching does not validate DICOM geometry or clinical quality",
                        "ProstateX-0025 finding-to-study association remains pending image geometry",
                        "ProstateX-0191 is outside the paired primary cohort",
                        "Official label-container provenance remains unverified despite matching CSV hashes"],
    }
    audit_path = Path("analysis/prostatex_full_metadata_inventory_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    markdown = f"""# PROSTATEx selection protocol v0.2 amendment

The full metadata inventory completed on 2026-09-17 and passed archive, hash,
identity-table, and structural checks. It contains 18,321 unique MR series records
for 346 collection patients. Matching was restricted to the 204 labeled patients.

The v0.1 table found primary metadata pairs for 203 labeled patients. Six pilot
patients remain excluded from all reported modeling. `ProstateX-0191` has no exact
canonical `t2_tse_tra` candidate and is excluded from the paired primary cohort.

One unanticipated case requires an amendment. `ProstateX-0025` has two eligible
T2/ADC pairs with identical descriptions and series numbers but different studies.
The five findings cannot safely be attached to a study using UID ordering. Both
study pairs must be audited geometrically; each finding is assigned only when
exactly one study supports the complete paired sampling grid. Ambiguous findings
are excluded without replacement. All studies from this patient stay in one fold.

Five deterministic patient folds are now frozen for 197 nonpilot patients with a
canonical metadata pair, including the pending `ProstateX-0025` patient. Each run
uses fold k as test, the next fold as validation, and the remaining three as train.
No patient is moved after a later geometry exclusion.

Fold sizes are {', '.join(str(row['patients']) for row in fold_summary)} patients;
positive-patient counts are {', '.join(str(row['positive_patients']) for row in fold_summary)};
TRUE finding counts are {', '.join(str(row['ClinSig_TRUE']) for row in fold_summary)}.

Before bulk retrieval, 196 patients and 314 findings require no special metadata
handling. `ProstateX-0025` adds one pending patient with five FALSE findings. The
initial plan contains {download_summary['unique_series']} unique series and begins
with the four series from its two study pairs. Same-study sensitivity alternatives
are deferred until the primary automated geometry checks are complete.
No images were downloaded and no model was trained in this amendment step.
"""
    markdown_path = Path("analysis/prostatex_selection_protocol_v0_2.md")
    markdown_path.write_text(markdown)
    artifacts = [protocol_path, folds_path, fold_path, download_path,
                 download_report_path, audit_path, markdown_path]
    output_manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__)), "generator_sha256": sha(Path(__file__).read_bytes()),
        "source_bundle_sha256": sha(bundle_raw),
        "artifacts": {str(path): {"sha256": sha(path.read_bytes()), "bytes": path.stat().st_size}
                      for path in artifacts}}
    manifest_path = Path("analysis/prostatex_selection_protocol_v0_2_manifest.json")
    manifest_path.write_text(json.dumps(output_manifest, indent=2) + "\n")
    print("Metadata bundle audit: PASS")
    print("Protocol amendment:", protocol_path)
    print("Fold patient/positive/TRUE counts:",
          [(row["patients"], row["positive_patients"], row["ClinSig_TRUE"]) for row in fold_summary])
    print("Initial download-plan series / reported instances:",
          download_summary["unique_series"], "/", download_summary["reported_DICOM_instances"])
    print("Exceptions: ProstateX-0025 multi-study pending; ProstateX-0191 paired-primary excluded")


if __name__ == "__main__":
    main()