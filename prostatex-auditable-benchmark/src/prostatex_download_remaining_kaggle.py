"""Download only the eight reviewed series for four remaining QC patients."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from zipfile import ZipFile, ZIP_STORED

REMAINING_API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"
UID_PREFIX = "1.3.6.1.4.1.14519.5.2.1.7311.5101."
EXPECTED_REMAINING = {
    ('ProstateX-0000', 'T2_axial'): (4, '160028252338004527274326500702', 19),
    ('ProstateX-0000', 'ADC'): (7, '339319789559896104041345048780', 19),
    ('ProstateX-0003', 'T2_axial'): (3, '320508777856431308769740730967', 21),
    ('ProstateX-0003', 'ADC'): (6, '241386734735530934020401083547', 19),
    ('ProstateX-0004', 'T2_axial'): (5, '206828891270520544417996275680', 19),
    ('ProstateX-0004', 'ADC'): (7, '278358228783511961204087191158', 19),
    ('ProstateX-0005', 'T2_axial'): (4, '119114186762760923175160291330', 19),
    ('ProstateX-0005', 'ADC'): (7, '314992768159280340618438416270', 19),
}


def verify_remaining_zip(path, candidate):
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
            "patient": candidate["patient"], "series_uid": candidate["series_uid"],
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


def select_remaining(inventory):
    patients = {patient for patient, channel in EXPECTED_REMAINING}
    selected = [row for row in inventory["candidates"] if row["patient"] in patients]
    keys = [(row["patient"], row["channel"]) for row in selected]
    if len(selected) != 8 or set(keys) != set(EXPECTED_REMAINING):
        raise ValueError("Expected exactly one T2 and one ADC for each of four patients")
    for row in selected:
        number, suffix, count = EXPECTED_REMAINING[(row["patient"], row["channel"])]
        if (int(row["series_number"]) != number or row["series_uid"] != UID_PREFIX + suffix
                or int(row["image_count"]) != count or row["api_matches"] != 1
                or row["csv_channel_candidates"] != 1 or row["in_training_manifest"] is not True):
            raise ValueError(f"Inventory mismatch: {row['patient']} {row['channel']}")
    for patient in patients:
        if len({row["study_uid"] for row in selected if row["patient"] == patient}) != 1:
            raise ValueError(f"Different reported studies for {patient}")
    return selected


def download_remaining(inventory_path):
    inventory_bytes = Path(inventory_path).read_bytes()
    selected = select_remaining(json.loads(inventory_bytes))
    root = Path("prostatex_series_review")
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    snapshot = root / f"remaining_four_inventory_{stamp}.json"
    snapshot.write_bytes(inventory_bytes)
    report_path = root / f"remaining_four_report_{stamp}.json"
    report = {
        "scope": "Eight series for four QC patients; no extraction, geometric validation, or training",
        "started_utc": stamp, "api": REMAINING_API, "status": "running",
        "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "series": [],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for row in selected:
            relative = Path(row["patient"]) / f"{row['channel']}_series_{row['series_number']}.zip"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            reused = path.exists()
            print("Checking cache:" if reused else "Downloading:", relative, flush=True)
            if not reused:
                temporary = path.with_suffix(".partial")
                url = REMAINING_API + "?" + urlencode({"SeriesInstanceUID": row["series_uid"]})
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
                details = verify_remaining_zip(temporary, row)
                temporary.replace(path)
            else:
                details = verify_remaining_zip(path, row)
            report["series"].append({**row, **details, "file": str(relative), "reused": reused})
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print("Verified:", relative, details["verified_dicom_files"], "DICOM files", flush=True)
    except Exception as error:
        report.update(status="failed", failed_series=row, error=f"{type(error).__name__}: {error}")
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print("Failure report:", report_path)
        raise
    report["status"] = "complete"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    bundle = root / f"remaining_four_review_{stamp}.zip"
    with ZipFile(bundle, "x", compression=ZIP_STORED) as archive:
        for row in report["series"]:
            archive.write(root / row["file"], arcname=row["file"])
        archive.write(report_path, arcname="download_report.json")
        archive.write(snapshot, arcname="inventory.json")
    print("Complete: 8 series,", sum(row["verified_dicom_files"] for row in report["series"]), "DICOM files.")
    print("ProstateX-0001 unchanged; no extraction or training.")
    return bundle


if "inventory_output" not in globals() or not callable(globals().get("read_sample")):
    raise RuntimeError("Run the corrected reader and inventory cells first")
remaining_bundle = download_remaining(Path(inventory_output) / "inventory.json")
from IPython.display import display, FileLink
display(FileLink(str(remaining_bundle)))