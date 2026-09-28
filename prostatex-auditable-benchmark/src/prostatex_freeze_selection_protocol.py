"""Freeze the PROSTATEx data-selection protocol after pilot QC, before modeling."""

import csv
import hashlib
import io
import json
from datetime import date, datetime, timezone
from pathlib import Path
from zipfile import ZipFile


PILOT_PATIENTS = [f"ProstateX-{index:04d}" for index in range(6)]
LABEL_ZIP = Path("prism-uploads/ProstateX-TrainingLesionInformationv2.zip")
INVENTORY = Path("prism-uploads/inventory.json")
PILOT_METADATA = Path("analysis/prostatex_patch_pilot_output/metadata.json")
REMAINING_GEOMETRY = Path("analysis/prostatex_remaining_four_geometry.json")
REPEAT_COMPARISON = Path("analysis/prostatex_0001_comparison.json")
QC01_REPORT = Path("prism-uploads/qc01_pydicom_20260916_205759_755261report.json")
VISUAL_QC = Path("prism-uploads/remaining_four_visual_qc_20260916_113906_812410.zip")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label_rows():
    with ZipFile(LABEL_ZIP) as archive:
        names = [name for name in archive.namelist()
                 if Path(name).name == "ProstateX-Findings-Train.csv"]
        if len(names) != 1:
            raise ValueError("Expected one findings CSV")
        raw = archive.read(names[0])
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    selected = [row for row in rows if row["ProxID"] in PILOT_PATIENTS]
    if len(selected) != 10:
        raise ValueError("Expected ten pilot findings")
    keys = {(row["ProxID"], row["fid"], tuple(map(float, row["pos"].split())))
            for row in selected}
    if len(keys) != len(selected):
        raise ValueError("Duplicate full finding key")
    return selected, hashlib.sha256(raw).hexdigest()


def build_series_register():
    inventory = json.loads(INVENTORY.read_text())
    pilot = json.loads(PILOT_METADATA.read_text())
    repeat = json.loads(REPEAT_COMPARISON.read_text())
    rows = []
    for row in inventory["candidates"]:
        if row["patient"] == "ProstateX-0001" and row["channel"] == "T2_axial":
            decision = ("PROPOSED_PRIMARY_METADATA_TIE_BREAK" if int(row["series_number"]) == 6
                        else "RETAINED_SENSITIVITY_ALTERNATE")
            ambiguity = "Repeated acquisition; same geometry/protocol, distinct pixels and SOP UIDs"
        else:
            decision = "UNIQUE_PILOT_CANDIDATE"
            ambiguity = ""
        rows.append({
            "patient": row["patient"], "channel": row["channel"],
            "series_number": int(row["series_number"]), "series_uid": row["series_uid"],
            "study_uid": row["study_uid"], "description": row["description"],
            "image_count": int(row["image_count"]), "candidate_decision": decision,
            "ambiguity": ambiguity, "modeling_role": "DEVELOPMENT_QC_EXCLUDED",
        })
    for channel, report in pilot["source_audits"].items():
        rows.append({
            "patient": "ProstateX-0002", "channel": channel,
            "series_number": int(report["series_number"]), "series_uid": report["series_uid"],
            "study_uid": report["study_uid"], "description": report["description"],
            "image_count": int(report["dicom_files"]), "candidate_decision": "UNIQUE_PILOT_CANDIDATE",
            "ambiguity": "", "modeling_role": "DEVELOPMENT_QC_EXCLUDED",
        })
    rows.sort(key=lambda row: (row["patient"], row["channel"], row["series_number"], row["series_uid"]))
    if len(rows) != 13 or len({row["series_uid"] for row in rows}) != 13:
        raise ValueError("Unexpected pilot series register")
    if repeat["selection"]["status"] != "not_finalized":
        raise ValueError("Unexpected prior repeat-selection status")
    if not repeat["t2_comparison"]["geometry_matches_within_tolerance"]:
        raise ValueError("Expected matching repeated-T2 geometry")
    if repeat["t2_comparison"]["array_equal"]:
        raise ValueError("Repeated T2 acquisitions unexpectedly identical")
    return rows


def build_finding_register(labels, series_rows):
    primary = {}
    alternates = {}
    for row in series_rows:
        key = (row["patient"], row["channel"])
        if row["candidate_decision"] in {"UNIQUE_PILOT_CANDIDATE", "PROPOSED_PRIMARY_METADATA_TIE_BREAK"}:
            if key in primary:
                raise ValueError("Multiple primary pilot series")
            primary[key] = row
        elif row["candidate_decision"] == "RETAINED_SENSITIVITY_ALTERNATE":
            alternates.setdefault(key, []).append(row)
    pilot = json.loads(PILOT_METADATA.read_text())
    pilot_keys = {(row["patient"], row["fid"], tuple(row["pos_mm"])) for row in pilot["findings"]}
    remaining = json.loads(REMAINING_GEOMETRY.read_text())
    remaining_keys = set()
    for patient, reports in remaining["series_by_patient"].items():
        remaining_keys.update((patient, row["fid"], tuple(row["pos_mm"]))
                              for row in reports["T2_axial"]["findings"])
    qc01 = json.loads(QC01_REPORT.read_text())
    if qc01["status"] != "TECHNICAL_MATCH":
        raise ValueError("QC01 technical match not established")
    rows = []
    for label in labels:
        patient, fid = label["ProxID"], label["fid"]
        point = tuple(map(float, label["pos"].split()))
        key = (patient, fid, point)
        t2, adc = primary[(patient, "T2_axial")], primary[(patient, "ADC")]
        if patient == "ProstateX-0001":
            geometry = "PASSED_FOR_BOTH_T2_ALTERNATIVES_AND_ADC"
            patch = "NOT_GENERATED_SELECTION_POLICY_WAS_PENDING"
            technical = "REPEATED_T2_NOT_EXACT_DUPLICATES"
            human_qc = "NOT_PREPARED"
        elif key in pilot_keys:
            geometry, patch = "PASSED", "GENERATED_NO_OUT_OF_BOUNDS"
            technical, human_qc = "SAME_LINEAGE_REPRODUCTION_ONLY", "NOT_PREPARED"
        elif key in remaining_keys:
            geometry, patch = "PASSED", "GENERATED_NO_OUT_OF_BOUNDS"
            technical = ("SEPARATE_PYDICOM_TECHNICAL_MATCH" if patient == "ProstateX-0000"
                         else "SAME_LINEAGE_REPRODUCTION_ONLY")
            human_qc = "PENDING_NONCLINICAL_WORKSHEET"
        else:
            raise ValueError("Finding absent from geometry evidence")
        rows.append({
            "patient": patient, "fid": fid,
            "pos_x_mm": point[0], "pos_y_mm": point[1], "pos_z_mm": point[2],
            "zone": label["zone"], "ClinSig": label["ClinSig"].upper(),
            "full_key": f"{patient}|{fid}|{point[0]} {point[1]} {point[2]}",
            "t2_primary_series_number": t2["series_number"], "t2_primary_series_uid": t2["series_uid"],
            "t2_alternate_series_uids": ";".join(row["series_uid"] for row in alternates.get((patient, "T2_axial"), [])),
            "adc_primary_series_number": adc["series_number"], "adc_primary_series_uid": adc["series_uid"],
            "geometry_status": geometry, "patch_status": patch,
            "technical_reconstruction_status": technical, "human_visual_qc_status": human_qc,
            "modeling_role": "DEVELOPMENT_QC_EXCLUDED_FROM_ALL_REPORTED_MODELING",
        })
    rows.sort(key=lambda row: (row["patient"], int(row["fid"]), row["pos_x_mm"], row["pos_y_mm"], row["pos_z_mm"]))
    if sum(row["ClinSig"] == "TRUE" for row in rows) != 4:
        raise ValueError("Unexpected pilot label distribution")
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    labels, findings_hash = label_rows()
    series_rows = build_series_register()
    finding_rows = build_finding_register(labels, series_rows)
    series_path = Path("analysis/prostatex_pilot_series_decisions.csv")
    findings_path = Path("analysis/prostatex_pilot_finding_register.csv")
    write_csv(series_path, series_rows)
    write_csv(findings_path, finding_rows)
    protocol = {
        "protocol_id": "PROSTATEx-selection-v0.1",
        "status": "FROZEN_FOR_FULL_METADATA_INVENTORY",
        "frozen_date": date.today().isoformat(),
        "timing": "Frozen after six-patient development QC and before full-cohort metadata inventory, image download, split generation, or model training; not a prospective preregistration.",
        "task": "Finding-level binary classification of recorded ClinSig at a supplied physical position; not whole-image detection, segmentation, or clinical diagnosis.",
        "finding_identity": ["ProxID", "fid", "pos_mm"],
        "pilot_policy": {
            "patients": PILOT_PATIENTS,
            "role": "Development QC only",
            "exclusion": "Exclude all six patients from every reported model-fitting, hyperparameter-selection, validation, and test set.",
            "reason": "Their labels, series alternatives, geometry, patches, and visual outputs were repeatedly inspected during pipeline development.",
        },
        "candidate_discovery": {
            "source_requirements": [
                "Patient must occur in the findings CSV and training manifest.",
                "Candidate description and series number must match ProstateX-Images-Train.csv and one API series record.",
                "T2 candidates use exact normalized description t2_tse_tra.",
                "ADC candidates use descriptions whose normalized value ends with _ADC.",
                "Series UID must occur in the supplied training manifest.",
            ],
            "selection_unit": "Select a T2/ADC pair from the same patient and StudyInstanceUID; confirm FrameOfReferenceUID after DICOM retrieval.",
            "tie_break": "Among multiple eligible pairs, rank ascending by T2 SeriesNumber, ADC SeriesNumber, T2 SeriesInstanceUID, then ADC SeriesInstanceUID. This metadata-only rule must not use labels, image appearance, model scores, or acquisition order as a quality claim.",
            "alternates": "Retain every eligible nonprimary pair in the register. A sensitivity analysis substitutes alternates without duplicating a patient across splits.",
            "no_pair": "If no eligible same-study pair remains, exclude from the paired-channel primary analysis and record eligibility separately for prespecified single-channel analyses.",
        },
        "post_download_hard_checks": [
            "Archive CRC, unique member names, bounded uncompressed size, expected DICOM count.",
            "Patient, study, series, modality, SOP class and SOP instance identities; no repeated SOP Instance UID.",
            "Supported classic single-frame native pixel encoding or explicit review before extension.",
            "Consistent dimensions, orientation and spacing; orthonormal in-plane directions; monotonic regular physical slice steps.",
            "Supplied point and complete 100x100, 0.5-mm T2-parallel grid must lie inside source pixel-center bounds for each used channel; no padding or clipping.",
            "Padding, rescale, modality-LUT, real-world-value, or related intensity semantics trigger explicit review; they are not silently ignored.",
        ],
        "non_exclusion_observations": [
            "Exact native or interpolated zero values alone are not an exclusion criterion.",
            "Nonblank appearance, common study/frame identifiers, or matching coordinates do not establish anatomical registration or clinical quality.",
            "Human or AI visual impressions cannot alter series selection unless a separate qualified-review protocol is approved before outcome analysis.",
        ],
        "split_rule_reserved_for_next_stage": "After full eligibility is known, create patient-level grouped splits. All findings from one patient remain in one split. Freeze patient IDs and seed before patch extraction and modeling.",
        "modeling_order_reserved_for_later": ["T2-only baseline", "ADC-only baseline", "paired T2+ADC baseline", "only then augmentation and search ablations"],
        "provenance_limit": "Current findings and imaging CSV contents are internally hash-tracked but official label-file provenance has not been independently established.",
        "pilot_summary": {
            "patients": 6, "findings": len(finding_rows),
            "ClinSig_TRUE": sum(row["ClinSig"] == "TRUE" for row in finding_rows),
            "ClinSig_FALSE": sum(row["ClinSig"] == "FALSE" for row in finding_rows),
            "candidate_series": len(series_rows), "repeated_T2_patients": ["ProstateX-0001"],
        },
        "source_sha256": {
            str(LABEL_ZIP): digest(LABEL_ZIP), str(INVENTORY): digest(INVENTORY),
            str(PILOT_METADATA): digest(PILOT_METADATA), str(REMAINING_GEOMETRY): digest(REMAINING_GEOMETRY),
            str(REPEAT_COMPARISON): digest(REPEAT_COMPARISON), str(QC01_REPORT): digest(QC01_REPORT),
            "ProstateX-Findings-Train.csv": findings_hash,
        },
    }
    protocol_path = Path("analysis/prostatex_selection_protocol_v0_1.json")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    markdown = f"""# PROSTATEx data-selection protocol v0.1

Status: **frozen for full metadata inventory** on {protocol['frozen_date']}. This is
not a prospective preregistration. It was written after six-patient development QC
and before full-cohort inventory, image download, split generation, or training.

## Research task

The proposed task is finding-level binary classification of recorded `ClinSig` at
a supplied physical position. It is not whole-image lesion detection, segmentation,
or a clinical diagnostic system. A finding is identified only by the full key
`(ProxID, fid, pos_mm)`; `fid` is not globally or patient-wise unique.

## Development-patient boundary

`ProstateX-0000` through `ProstateX-0005` are development-QC patients. All ten
findings from these six patients are excluded from every reported model-fitting,
model-selection, validation, and test set because their labels, series alternatives,
geometry, patches, and visual outputs were repeatedly inspected. They remain useful
only as fixed pipeline tests. The register contains four TRUE and six FALSE findings.

## Series discovery and deterministic selection

Candidate series must match the patient, normalized description and series number
recorded in `ProstateX-Images-Train.csv`, map to exactly one API record, and occur
in the supplied training manifest. T2 uses exact normalized `t2_tse_tra`; ADC uses
descriptions ending in `_ADC`.

Eligible T2/ADC pairs must belong to the same patient and study; their shared frame
identifier is checked after retrieval. If more than one eligible pair remains, the
primary pair is ranked by ascending T2 series number, ADC series number, T2 UID,
then ADC UID. This is an audit-friendly metadata tie-break, not a claim that an
earlier acquisition is clinically superior. All alternatives remain recorded and
must be substituted in a sensitivity analysis without duplicating a patient across
splits. For `ProstateX-0001`, this proposed rule selects T2 series 6 and retains
series 10 as the alternate; the patient is development-only regardless.

## Hard checks after retrieval

- Verify archive integrity, identities, SOP uniqueness, DICOM count, and supported encoding.
- Require consistent dimensions, orientation and spacing, orthonormal in-plane
  directions, and monotonic regular physical slice steps.
- Require the entire 100x100, 0.5-mm T2-parallel sampling grid to remain inside
  each source volume's pixel-center bounds; do not pad or clip.
- Record padding and intensity-mapping semantics and stop for explicit review if
  they are present; do not silently interpret physical ADC units.

Exact zeros are not an automatic exclusion criterion. Shared coordinates or frame
identifiers do not prove anatomical registration, and visual impressions from a
nonclinical reviewer or AI do not establish clinical quality.

## Next stage

The next operation is a metadata-only inventory of the complete labeled training
cohort using these fixed rules. Once eligibility is known, patient-level grouped
splits and a seed must be frozen before full patch extraction or modeling. Baselines
will proceed in the order T2-only, ADC-only, and paired T2+ADC; GAN/GA experiments
come only after the baselines and data checks are complete.

## Generated registers

- `prostatex_pilot_series_decisions.csv`: 13 candidate series and the repeated-T2 decision.
- `prostatex_pilot_finding_register.csv`: 10 full finding keys and current evidence status.
- `prostatex_selection_protocol_v0_1.json`: machine-readable complete protocol and source hashes.

Current CSVs are internally hash-tracked, but independent official label-file
provenance and clinical/anatomical validation remain unresolved.
"""
    markdown_path = Path("analysis/prostatex_selection_protocol_v0_1.md")
    markdown_path.write_text(markdown, encoding="utf-8")
    artifacts = [series_path, findings_path, protocol_path, markdown_path]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__)), "generator_sha256": digest(Path(__file__)),
        "artifacts": {str(path): {"sha256": digest(path), "bytes": path.stat().st_size}
                      for path in artifacts},
    }
    manifest_path = Path("analysis/prostatex_selection_protocol_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Protocol:", protocol_path)
    print("Pilot patients/findings/series:", 6, len(finding_rows), len(series_rows))
    print("Development patients excluded from all reported modeling:", ", ".join(PILOT_PATIENTS))
    print("Repeated T2 primary/alternate for ProstateX-0001: series 6 / series 10")


if __name__ == "__main__":
    main()