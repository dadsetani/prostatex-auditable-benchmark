"""Metadata-only PROSTATEx cohort inventory under selection protocol v0.1."""

import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
from zipfile import ZipFile, ZIP_DEFLATED


API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries"
PILOT_PATIENTS = {f"ProstateX-{index:04d}" for index in range(6)}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_inputs(labels_zip, manifest_path, protocol_path):
    labels_raw = Path(labels_zip).read_bytes()
    manifest_raw = Path(manifest_path).read_bytes()
    protocol_raw = Path(protocol_path).read_bytes()
    protocol = json.loads(protocol_raw)
    if (protocol.get("protocol_id") != "PROSTATEx-selection-v0.1"
            or protocol.get("status") != "FROZEN_FOR_FULL_METADATA_INVENTORY"):
        raise ValueError("Unexpected or unfrozen selection protocol")
    with ZipFile(io.BytesIO(labels_raw)) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("Invalid label ZIP")
        tables, csv_hashes = {}, {}
        for basename in ("ProstateX-Findings-Train.csv", "ProstateX-Images-Train.csv"):
            names = [name for name in archive.namelist() if Path(name).name == basename]
            if len(names) != 1:
                raise ValueError("Expected one " + basename)
            raw = archive.read(names[0])
            tables[basename] = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
            csv_hashes[basename] = sha(raw)
    manifest_uids = set(re.findall(r"^\s*(\d+(?:\.\d+)+)\s*$",
                                   manifest_raw.decode("utf-8-sig"), re.MULTILINE))
    if not manifest_uids:
        raise ValueError("No UIDs in training manifest")
    findings = tables["ProstateX-Findings-Train.csv"]
    images = tables["ProstateX-Images-Train.csv"]
    patients = sorted({row["ProxID"] for row in findings})
    if len(patients) != 204 or len(findings) != 330:
        raise ValueError("Unexpected labeled cohort size")
    keys = {(row["ProxID"], row["fid"], tuple(map(float, row["pos"].split()))) for row in findings}
    if len(keys) != len(findings):
        raise ValueError("Duplicate full finding key")
    return {
        "findings": findings, "images": images, "patients": patients,
        "manifest_uids": manifest_uids, "protocol": protocol,
        "hashes": {"label_zip": sha(labels_raw), "manifest": sha(manifest_raw),
                   "protocol": sha(protocol_raw), **csv_hashes},
    }


def csv_candidates(images, patients):
    patient_set = set(patients)
    candidates = defaultdict(lambda: {"T2_axial": set(), "ADC": set()})
    for row in images:
        patient = row["ProxID"]
        if patient not in patient_set:
            continue
        description = row["DCMSerDescr"].strip()
        normalized = description.lower()
        number = int(row["DCMSerNum"])
        if normalized == "t2_tse_tra":
            candidates[patient]["T2_axial"].add((description, number))
        if normalized.endswith("_adc"):
            candidates[patient]["ADC"].add((description, number))
    return candidates


def finding_summary(findings, patients):
    output = {}
    by_patient = defaultdict(list)
    for row in findings:
        by_patient[row["ProxID"]].append(row)
    for patient in patients:
        rows = by_patient[patient]
        fid_counts = Counter(row["fid"] for row in rows)
        output[patient] = {
            "finding_count": len(rows),
            "ClinSig_TRUE": sum(row["ClinSig"].strip().upper() == "TRUE" for row in rows),
            "ClinSig_FALSE": sum(row["ClinSig"].strip().upper() == "FALSE" for row in rows),
            "repeated_fids": sorted(fid for fid, count in fid_counts.items() if count > 1),
        }
    return output


def build_preinventory(inputs):
    candidates = csv_candidates(inputs["images"], inputs["patients"])
    findings = finding_summary(inputs["findings"], inputs["patients"])
    rows = []
    for patient in inputs["patients"]:
        t2 = sorted(candidates[patient]["T2_axial"], key=lambda row: (row[1], row[0]))
        adc = sorted(candidates[patient]["ADC"], key=lambda row: (row[1], row[0]))
        if not t2:
            status = "REVIEW_REQUIRED_NO_CANONICAL_T2_CSV"
        elif not adc:
            status = "REVIEW_REQUIRED_NO_ADC_CSV"
        elif len(t2) == len(adc) == 1:
            status = "READY_FOR_API_MATCH_UNIQUE_CSV_PAIR"
        else:
            status = "READY_FOR_API_MATCH_WITH_METADATA_TIE_BREAK"
        summary = findings[patient]
        rows.append({
            "patient": patient, **summary,
            "canonical_t2_candidate_count": len(t2),
            "canonical_t2_candidates": json.dumps(t2, separators=(",", ":")),
            "adc_candidate_count": len(adc),
            "adc_candidates": json.dumps(adc, separators=(",", ":")),
            "preinventory_status": status,
            "modeling_role": "DEVELOPMENT_QC_EXCLUDED" if patient in PILOT_PATIENTS else "COHORT_CANDIDATE",
        })
    return rows


def fetch_collection_series(output_dir):
    url = API + "?" + urlencode({"Collection": "PROSTATEx", "Modality": "MR", "format": "json"})
    request = Request(url, headers={"Cache-Control": "no-cache",
                                    "User-Agent": "PROSTATEx-selection-protocol-v0.1"})
    with urlopen(request, timeout=180) as response:
        parsed = urlsplit(response.geturl())
        if parsed.scheme != "https" or parsed.hostname != "services.cancerimagingarchive.net":
            raise ValueError("Unexpected API redirect")
        raw = response.read(32 * 1024**2 + 1)
        if len(raw) > 32 * 1024**2:
            raise ValueError("API response exceeds size limit")
        metadata = {"requested_url": url, "final_url": response.geturl(),
                    "http_status": response.status, "server_date": response.headers.get("Date")}
    rows = json.loads(raw)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Unexpected or empty API response")
    required = {"PatientID", "SeriesInstanceUID", "StudyInstanceUID", "SeriesDescription",
                "SeriesNumber", "ImageCount", "Collection", "Modality"}
    for row in rows:
        if not isinstance(row, dict) or not required <= set(row):
            raise ValueError("Malformed API series record")
        if row["Collection"] != "PROSTATEx" or row["Modality"] != "MR":
            raise ValueError("Unexpected collection or modality")
    snapshot = Path(output_dir) / "api_getSeries_PROSTATEx_MR.json"
    snapshot.write_bytes(raw)
    return rows, metadata, snapshot


def api_candidate_rows(inputs, api_rows):
    csv_by_patient = csv_candidates(inputs["images"], inputs["patients"])
    api_by_patient = defaultdict(list)
    for row in api_rows:
        api_by_patient[row["PatientID"]].append(row)
    candidates = []
    eligible = defaultdict(lambda: {"T2_axial": [], "ADC": []})
    for patient in inputs["patients"]:
        for channel in ("T2_axial", "ADC"):
            for description, number in sorted(csv_by_patient[patient][channel], key=lambda row: (row[1], row[0])):
                matches = [row for row in api_by_patient[patient]
                           if str(row["SeriesDescription"]).strip() == description
                           and str(row["SeriesNumber"]).strip() == str(number)]
                manifest_matches = [row for row in matches if row["SeriesInstanceUID"] in inputs["manifest_uids"]]
                if not matches:
                    candidates.append({"patient": patient, "channel": channel,
                        "description": description, "series_number": number, "series_uid": "",
                        "study_uid": "", "image_count": "", "api_match_count": 0,
                        "manifest_match_count": 0, "status": "NO_API_MATCH"})
                for row in matches:
                    item = {"patient": patient, "channel": channel,
                        "description": description, "series_number": number,
                        "series_uid": row["SeriesInstanceUID"], "study_uid": row["StudyInstanceUID"],
                        "image_count": int(row["ImageCount"]), "api_match_count": len(matches),
                        "manifest_match_count": len(manifest_matches),
                        "status": "ELIGIBLE_METADATA" if row in manifest_matches else "NOT_IN_TRAINING_MANIFEST"}
                    candidates.append(item)
                    if item["status"] == "ELIGIBLE_METADATA":
                        eligible[patient][channel].append(item)
    return candidates, eligible


def choose_pairs(inputs, eligible, preinventory):
    pre_by_patient = {row["patient"]: row for row in preinventory}
    findings = finding_summary(inputs["findings"], inputs["patients"])
    rows, alternatives = [], []
    for patient in inputs["patients"]:
        t2, adc = eligible[patient]["T2_axial"], eligible[patient]["ADC"]
        pairs = [(left, right) for left in t2 for right in adc
                 if left["study_uid"] and left["study_uid"] == right["study_uid"]]
        pairs.sort(key=lambda pair: (pair[0]["series_number"], pair[1]["series_number"],
                                     pair[0]["series_uid"], pair[1]["series_uid"]))
        if not pairs:
            status = (pre_by_patient[patient]["preinventory_status"]
                      if not t2 or not adc else "REVIEW_REQUIRED_NO_SAME_STUDY_PAIR")
            primary = None
        else:
            primary = pairs[0]
            status = "PRIMARY_UNIQUE_PAIR" if len(pairs) == 1 else "PRIMARY_WITH_ALTERNATES"
        if patient in PILOT_PATIENTS:
            status = "DEVELOPMENT_QC_EXCLUDED_" + status
        summary = findings[patient]
        rows.append({
            "patient": patient, **summary, "eligible_t2_series": len(t2),
            "eligible_adc_series": len(adc), "eligible_same_study_pairs": len(pairs),
            "primary_t2_series_number": primary[0]["series_number"] if primary else "",
            "primary_t2_series_uid": primary[0]["series_uid"] if primary else "",
            "primary_adc_series_number": primary[1]["series_number"] if primary else "",
            "primary_adc_series_uid": primary[1]["series_uid"] if primary else "",
            "study_uid": primary[0]["study_uid"] if primary else "", "selection_status": status,
            "modeling_role": "DEVELOPMENT_QC_EXCLUDED" if patient in PILOT_PATIENTS else "COHORT_CANDIDATE",
        })
        for rank, pair in enumerate(pairs[1:], start=2):
            alternatives.append({"patient": patient, "rank": rank,
                "t2_series_number": pair[0]["series_number"], "t2_series_uid": pair[0]["series_uid"],
                "adc_series_number": pair[1]["series_number"], "adc_series_uid": pair[1]["series_uid"],
                "study_uid": pair[0]["study_uid"], "role": "SENSITIVITY_ALTERNATE"})
    return rows, alternatives


def write_csv(path, rows, empty_fields=None):
    fields = list(rows[0]) if rows else empty_fields
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_preinventory(labels_zip, manifest_path, protocol_path, output_prefix):
    inputs = load_inputs(labels_zip, manifest_path, protocol_path)
    rows = build_preinventory(inputs)
    csv_path = Path(str(output_prefix) + ".csv")
    json_path = Path(str(output_prefix) + ".json")
    write_csv(csv_path, rows)
    counts = Counter(row["preinventory_status"] for row in rows)
    report = {"scope": "CSV-and-manifest preinventory only; no API request or image download",
        "created_utc": datetime.now(timezone.utc).isoformat(), "protocol_id": inputs["protocol"]["protocol_id"],
        "patients": len(rows), "findings": len(inputs["findings"]),
        "ClinSig_TRUE": sum(row["ClinSig"].strip().upper() == "TRUE" for row in inputs["findings"]),
        "ClinSig_FALSE": sum(row["ClinSig"].strip().upper() == "FALSE" for row in inputs["findings"]),
        "status_counts": dict(sorted(counts.items())),
        "patients_with_multiple_canonical_t2": [row["patient"] for row in rows if row["canonical_t2_candidate_count"] > 1],
        "patients_with_multiple_adc": [row["patient"] for row in rows if row["adc_candidate_count"] > 1],
        "patients_without_canonical_t2": [row["patient"] for row in rows if row["canonical_t2_candidate_count"] == 0],
        "patients_without_adc": [row["patient"] for row in rows if row["adc_candidate_count"] == 0],
        "input_sha256": inputs["hashes"], "csv_sha256": sha(csv_path.read_bytes())}
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def run_inventory(labels_zip, manifest_path, protocol_path):
    inputs = load_inputs(labels_zip, manifest_path, protocol_path)
    preinventory = build_preinventory(inputs)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = Path("prostatex_full_metadata_" + stamp)
    output.mkdir(exist_ok=False)
    report = {"scope": "Full labeled-cohort metadata inventory; no image download, extraction, split, or training",
              "status": "RUNNING", "created_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_id": inputs["protocol"]["protocol_id"], "input_sha256": inputs["hashes"]}
    report_path = output / "inventory_report.json"
    try:
        api_rows, retrieval, snapshot = fetch_collection_series(output)
        candidates, eligible = api_candidate_rows(inputs, api_rows)
        patients, alternatives = choose_pairs(inputs, eligible, preinventory)
        write_csv(output / "preinventory.csv", preinventory)
        write_csv(output / "series_candidates.csv", candidates)
        write_csv(output / "patient_pair_selection.csv", patients)
        write_csv(output / "pair_alternatives.csv", alternatives,
                  ["patient", "rank", "t2_series_number", "t2_series_uid",
                   "adc_series_number", "adc_series_uid", "study_uid", "role"])
        report.update(status="COMPLETE", retrieval=retrieval, api_snapshot_sha256=sha(snapshot.read_bytes()),
            api_series_count=len(api_rows), api_patient_count=len({row["PatientID"] for row in api_rows}),
            labeled_patients=len(inputs["patients"]), findings=len(inputs["findings"]),
            eligible_primary_pairs=sum(bool(row["primary_t2_series_uid"] and row["primary_adc_series_uid"]) for row in patients),
            patients_with_alternates=sum(row["eligible_same_study_pairs"] > 1 for row in patients),
            selection_status_counts=dict(sorted(Counter(row["selection_status"] for row in patients).items())))
    except Exception as error:
        report.update(status="FAILED", error=type(error).__name__ + ": " + str(error))
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    artifacts = {path.name: {"sha256": sha(path.read_bytes()), "bytes": path.stat().st_size}
                 for path in sorted(output.iterdir()) if path.is_file()}
    hashes_path = output / "artifact_hashes.json"
    hashes_path.write_text(json.dumps(artifacts, indent=2) + "\n", encoding="utf-8")
    bundle = output.with_suffix(".zip")
    with ZipFile(bundle, "x", compression=ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file():
                archive.write(path, arcname=path.name)
    print("Status:", report["status"])
    if "error" in report:
        print("Error:", report["error"])
    else:
        print("Labeled patients / findings:", report["labeled_patients"], "/", report["findings"])
        print("Primary pairs / patients with alternates:", report["eligible_primary_pairs"], "/", report["patients_with_alternates"])
    print("Report:", report_path)
    print("Bundle:", bundle)
    return bundle


if __name__ == "__main__":
    write_preinventory(
        "prism-uploads/ProstateX-TrainingLesionInformationv2.zip",
        "prism-uploads/PROSTATEx-train.tcia",
        "analysis/prostatex_selection_protocol_v0_1.json",
        "analysis/prostatex_full_preinventory",
    )