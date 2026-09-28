"""Download only the three reviewed ProstateX-0001 candidates; do not select a final T2."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from zipfile import ZipFile, ZIP_STORED

DOWNLOAD_API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"
EXPECTED = {
    ("T2_axial", 6): "1.3.6.1.4.1.14519.5.2.1.7311.5101.143550427570936841761028976610",
    ("T2_axial", 10): "1.3.6.1.4.1.14519.5.2.1.7311.5101.296510477295449267377313817541",
    ("ADC", 8): "1.3.6.1.4.1.14519.5.2.1.7311.5101.158774312014703328155251833954",
}


def verify_series_zip(path, candidate):
    with ZipFile(path) as archive:
        entries = archive.infolist()
        if sum(entry.file_size for entry in entries) > 512 * 1024**2:
            raise ValueError("Archive exceeds the pilot uncompressed-size limit")
        if len({entry.filename for entry in entries}) != len(entries):
            raise ValueError("Duplicate ZIP member names")
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC failure")
        dicoms = [entry for entry in entries if entry.filename.lower().endswith(".dcm")]
        if len(dicoms) != int(candidate["image_count"]):
            raise ValueError("DICOM file count differs from inventory")
        identities = {
            "patient": "ProstateX-0001", "series_uid": candidate["series_uid"],
            "study_uid": candidate["study_uid"], "description": candidate["description"],
            "modality": "MR",
        }
        sop_uids = set()
        for entry in dicoms:
            sample = read_sample(archive.read(entry))
            if any(sample[field] != expected for field, expected in identities.items()):
                raise ValueError(f"DICOM identity mismatch: {entry.filename}")
            if int(sample["series_number"]) != int(candidate["series_number"]):
                raise ValueError("Series number mismatch")
            if sample["sop_uid"] in sop_uids:
                raise ValueError("Repeated SOP Instance UID")
            sop_uids.add(sample["sop_uid"])
    return {
        "verified_dicom_files": len(dicoms),
        "zip_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "zip_bytes": path.stat().st_size,
    }


def download_review(inventory_path):
    inventory_bytes = Path(inventory_path).read_bytes()
    inventory = json.loads(inventory_bytes)
    candidates = [row for row in inventory["candidates"] if row["patient"] == "ProstateX-0001"]
    keys = [(row["channel"], int(row["series_number"])) for row in candidates]
    if len(candidates) != 3 or set(keys) != set(EXPECTED):
        raise ValueError("Expected exactly T2 series 6 and 10 plus ADC series 8")
    for candidate in candidates:
        key = (candidate["channel"], int(candidate["series_number"]))
        if candidate["series_uid"] != EXPECTED[key] or candidate["api_matches"] != 1 or candidate["in_training_manifest"] is not True:
            raise ValueError("Candidate does not match the reviewed inventory")
        if int(candidate["image_count"]) <= 0:
            raise ValueError("Invalid expected image count")
    output_dir = Path("prostatex_series_review/ProstateX-0001")
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    report = {
        "scope": "Three candidate series only; no final T2 selection, extraction, or training",
        "started_utc": stamp, "api": DOWNLOAD_API,
        "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "validation": "ZIP CRC, file count, DICOM identity, unique SOP UIDs; format-limited reader",
        "series": [],
    }
    report_path = output_dir / f"download_report_{stamp}.json"
    snapshot = output_dir / f"inventory_{stamp}.json"
    snapshot.write_bytes(inventory_bytes)
    for candidate in candidates:
        path = output_dir / f"{candidate['channel']}_series_{candidate['series_number']}.zip"
        reused = path.exists()
        print("Checking cache:" if reused else "Downloading:", path.name, flush=True)
        if not reused:
            temporary = path.with_suffix(".partial")
            url = DOWNLOAD_API + "?" + urlencode({"SeriesInstanceUID": candidate["series_uid"]})
            total = 0
            with urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > 256 * 1024**2:
                        raise ValueError("Download exceeds the per-series pilot limit")
                    handle.write(chunk)
            details = verify_series_zip(temporary, candidate)
            temporary.replace(path)
        else:
            details = verify_series_zip(path, candidate)
        report["series"].append({**candidate, **details, "file": path.name, "reused": reused})
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print("Verified:", path.name, details["verified_dicom_files"], "DICOM files", flush=True)
    bundle = output_dir / f"ProstateX-0001_review_{stamp}.zip"
    with ZipFile(bundle, "x", compression=ZIP_STORED) as archive:
        for record in report["series"]:
            archive.write(output_dir / record["file"], arcname=record["file"])
        archive.write(report_path, arcname="download_report.json")
        archive.write(snapshot, arcname="inventory.json")
    print("Finished: three candidate series only; no final selection or training.")
    return bundle


if "inventory_output" not in globals() or not callable(globals().get("read_sample")):
    raise RuntimeError("Run the corrected reader and metadata-inventory cells first")
review_bundle = download_review(Path(inventory_output) / "inventory.json")
from IPython.display import display, FileLink
display(FileLink(str(review_bundle)))