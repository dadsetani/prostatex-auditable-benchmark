"""Metadata-only inventory. Run after the existing input-discovery cell."""

import csv
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from zipfile import ZipFile

PATIENTS = ["ProstateX-0000", "ProstateX-0001", "ProstateX-0003", "ProstateX-0004", "ProstateX-0005"]
API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries"


def build_candidates(image_rows, responses, manifest_uids):
    output = []
    for patient in PATIENTS:
        for channel in ("T2_axial", "ADC"):
            expected = set()
            for row in image_rows:
                description = row["DCMSerDescr"].strip()
                lower = description.lower()
                matches = ("t2" in lower and ("tra" in lower or "ax" in lower)) if channel == "T2_axial" else "adc" in lower
                if row["ProxID"] == patient and matches:
                    expected.add((description, int(row["DCMSerNum"])))
            if not expected:
                raise ValueError(f"No CSV candidates: {patient}, {channel}")
            for description, number in sorted(expected):
                matches = [series for series in responses[patient]
                           if str(series.get("SeriesDescription", "")).strip() == description
                           and str(series.get("SeriesNumber", "")).strip() == str(number)]
                for series in matches or [{}]:
                    uid = str(series.get("SeriesInstanceUID", ""))
                    output.append({
                        "patient": patient, "channel": channel,
                        "description": description, "series_number": number,
                        "csv_channel_candidates": len(expected), "api_matches": len(matches),
                        "series_uid": uid, "study_uid": series.get("StudyInstanceUID", ""),
                        "image_count": series.get("ImageCount", ""),
                        "in_training_manifest": bool(uid and uid in manifest_uids),
                        "status": "CANDIDATE_ONLY" if len(matches) == 1 and uid in manifest_uids else "REVIEW_REQUIRED",
                    })
    return output


def run_inventory(labels_zip, manifest_path):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(f"prostatex_inventory_{stamp}")
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest_uids = set(re.findall(r"^\s*(\d+(?:\.\d+)+)\s*$", manifest_bytes.decode("utf-8-sig"), re.MULTILINE))
    if not manifest_uids:
        raise ValueError("No series UIDs found in training manifest")
    with ZipFile(labels_zip) as archive:
        names = [name for name in archive.namelist() if Path(name).name == "ProstateX-Images-Train.csv"]
        if len(names) != 1:
            raise ValueError("Expected exactly one imaging CSV")
        image_bytes = archive.read(names[0])
    image_rows = list(csv.DictReader(io.StringIO(image_bytes.decode("utf-8-sig"))))
    output_dir.mkdir(exist_ok=False)
    responses = {}
    for patient in PATIENTS:
        parameters = {"Collection": "PROSTATEx", "PatientID": patient, "Modality": "MR", "format": "json"}
        with urlopen(API + "?" + urlencode(parameters), timeout=60) as response:
            raw = response.read()
        (output_dir / f"{patient}_response.json").write_bytes(raw)
        series_list = json.loads(raw)
        if not isinstance(series_list, list) or not series_list:
            raise ValueError(f"Unexpected or empty API response for {patient}")
        for series in series_list:
            if not isinstance(series, dict) or series.get("PatientID") != patient or series.get("Collection") != "PROSTATEx":
                raise ValueError(f"API patient/collection mismatch: {patient}")
            if not series.get("SeriesInstanceUID"):
                raise ValueError(f"Missing series UID: {patient}")
        responses[patient] = series_list
        print(patient, "API series:", len(series_list))
    candidates = build_candidates(image_rows, responses, manifest_uids)
    with (output_dir / "candidates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    report = {
        "scope": "Metadata candidates only; no final series selection, image download, or training",
        "patients": PATIENTS, "selection_rule": "First five sorted training patient IDs excluding ProstateX-0002; development QC only",
        "retrieval_started_utc": stamp, "api": API,
        "imaging_csv_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "response_sha256": {patient: hashlib.sha256((output_dir / f"{patient}_response.json").read_bytes()).hexdigest() for patient in PATIENTS},
        "candidates": candidates,
    }
    (output_dir / "inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in candidates:
        print(row["patient"], row["channel"], "series", row["series_number"],
              "images", row["image_count"], "CSV choices", row["csv_channel_candidates"],
              "API matches", row["api_matches"], "manifest", row["in_training_manifest"], "UID", row["series_uid"])
    print("Saved:", output_dir)
    print("No final series selection; no image download; no training.")
    return output_dir


inventory_manifest = locate_optional("PROSTATEx-train.tcia")
if inventory_manifest is None:
    raise FileNotFoundError("Attach PROSTATEx-train.tcia to the notebook inputs")
inventory_output = run_inventory(Path(LABEL_ZIP), inventory_manifest)

from IPython.display import display, FileLink
display(FileLink(str(inventory_output / "inventory.json")))