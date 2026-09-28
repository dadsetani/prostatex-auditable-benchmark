"""Fresh official-source comparison of one ADC series, not clinical validation."""

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
from zipfile import ZipFile, ZIP_STORED

import numpy as np

REFERENCE_UID = "1.3.6.1.4.1.14519.5.2.1.7311.5101.339319789559896104041345048780"
REFERENCE_API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"


def index_reference_adc(raw_zip):
    indexed = {}
    with ZipFile(io.BytesIO(raw_zip)) as archive:
        entries = archive.infolist()
        if len(entries) != len({entry.filename for entry in entries}):
            raise ValueError("Duplicate archive members")
        if sum(entry.file_size for entry in entries) > 128 * 1024**2:
            raise ValueError("Archive exceeds uncompressed-size limit")
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC failure")
        dicoms = [entry for entry in entries if entry.filename.lower().endswith(".dcm")]
        if len(dicoms) != 19:
            raise ValueError("Expected 19 DICOM files for this specific series")
        for entry in dicoms:
            raw = archive.read(entry)
            sample = read_sample(raw)
            if sample["patient"] != "ProstateX-0000" or sample["series_uid"] != REFERENCE_UID or sample["modality"] != "MR":
                raise ValueError("Unexpected patient, series, or modality")
            uid = sample["sop_uid"]
            if uid in indexed:
                raise ValueError("Duplicate SOP Instance UID")
            pixels = np.frombuffer(sample["pixels"], dtype="<u2").reshape(sample["rows"], sample["columns"])
            indexed[uid] = {
                "filename": entry.filename,
                "dicom_sha256": hashlib.sha256(raw).hexdigest(),
                "pixel_bytes_sha256": hashlib.sha256(sample["pixels"]).hexdigest(),
                "zero_mask_sha256": hashlib.sha256((pixels == 0).tobytes()).hexdigest(),
                "zero_pixel_count": int(np.count_nonzero(pixels == 0)),
                "shape": list(pixels.shape),
                "selected_headers": {key: value.tolist() if isinstance(value, np.ndarray) else value
                                     for key, value in sample.items() if key != "pixels"},
            }
    return indexed


def compare_reference_adc(previous, fresh):
    shared = sorted(set(previous) & set(fresh))
    comparisons = []
    for uid in shared:
        old, new = previous[uid], fresh[uid]
        changed = sorted(key for key in set(old["selected_headers"]) | set(new["selected_headers"])
                         if old["selected_headers"].get(key) != new["selected_headers"].get(key))
        comparisons.append({
            "sop_uid": uid,
            "dicom_file_hash_equal": old["dicom_sha256"] == new["dicom_sha256"],
            "stored_pixels_equal": old["shape"] == new["shape"] and old["pixel_bytes_sha256"] == new["pixel_bytes_sha256"],
            "zero_masks_equal": old["shape"] == new["shape"] and old["zero_mask_sha256"] == new["zero_mask_sha256"],
            "selected_header_fields_changed": changed,
        })
    same_set = set(previous) == set(fresh)
    exact = same_set and len(shared) == 19 and all(row["dicom_file_hash_equal"] for row in comparisons)
    return {
        "status": "MATCH_EXACT_DICOM_FILES" if exact else "REVIEW_REQUIRED",
        "sop_sets_equal": same_set, "compared_dicom_files": len(shared),
        "missing_from_fresh": sorted(set(previous) - set(fresh)),
        "new_in_fresh": sorted(set(fresh) - set(previous)),
        "all_stored_pixels_equal": same_set and len(shared) == 19 and all(row["stored_pixels_equal"] for row in comparisons),
        "all_zero_masks_equal": same_set and len(shared) == 19 and all(row["zero_masks_equal"] for row in comparisons),
        "comparisons": comparisons,
    }


def fresh_official_adc_check(previous_bundle):
    previous_bundle = Path(previous_bundle)
    with ZipFile(previous_bundle) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("Invalid previous bundle")
        download = json.loads(archive.read("download_report.json"))
        matches = [row for row in download["series"] if row["series_uid"] == REFERENCE_UID]
        if len(matches) != 1:
            raise ValueError("Expected one reference ADC record")
        previous_raw = archive.read(matches[0]["file"])
        if hashlib.sha256(previous_raw).hexdigest() != matches[0]["zip_sha256"]:
            raise ValueError("Previous series hash mismatch")
    previous = index_reference_adc(previous_raw)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = Path(f"official_adc_check_{stamp}")
    output.mkdir(exist_ok=False)
    url = REFERENCE_API + "?" + urlencode({"SeriesInstanceUID": REFERENCE_UID})
    report = {
        "scope": "Fresh official-source retrieval compared by SOP UID; one ADC series only; no clinical validation or label CSV verification",
        "started_utc": datetime.now(timezone.utc).isoformat(), "requested_url": url,
        "patient": "ProstateX-0000", "series_uid": REFERENCE_UID,
        "previous_bundle_sha256": hashlib.sha256(previous_bundle.read_bytes()).hexdigest(),
        "previous_adc_zip_sha256": hashlib.sha256(previous_raw).hexdigest(),
        "local_cache_reused": False,
    }
    try:
        request = Request(url, headers={"Cache-Control": "no-cache"})
        print("Fresh request to TCIA; no local cache reuse.", flush=True)
        with urlopen(request, timeout=120) as response:
            final_url = response.geturl()
            parsed = urlsplit(final_url)
            if parsed.scheme != "https" or parsed.hostname != "services.cancerimagingarchive.net":
                raise ValueError("Unexpected redirect; review source before continuing")
            report["final_url"] = final_url
            report["http_status"] = response.status
            report["server_date"] = response.headers.get("Date")
            fresh_raw = response.read(32 * 1024**2 + 1)
        if len(fresh_raw) > 32 * 1024**2:
            raise ValueError("Fresh response exceeds download-size limit")
        (output / "fresh_ADC.zip").write_bytes(fresh_raw)
        fresh = index_reference_adc(fresh_raw)
        report.update(compare_reference_adc(previous, fresh))
        report.update(
            fresh_adc_zip_sha256=hashlib.sha256(fresh_raw).hexdigest(),
            previous_instances=previous, fresh_instances=fresh,
            completed_utc=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        print("Failure report:", output / "comparison.json")
        raise
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    bundle = output.with_suffix(".zip")
    with ZipFile(bundle, "x", compression=ZIP_STORED) as archive:
        for name in ("comparison.json", "fresh_ADC.zip"):
            archive.write(output / name, arcname=name)
    print("Status:", report["status"])
    print("Compared DICOM files:", report["compared_dicom_files"])
    print("Stored pixels equal:", report["all_stored_pixels_equal"])
    print("Zero masks equal:", report["all_zero_masks_equal"])
    print("Saved:", bundle)
    return bundle


if __name__ == "__main__":
    if "remaining_bundle" not in globals() or not callable(globals().get("read_sample")):
        raise RuntimeError("Run the previous reader cells and set remaining_bundle to the Step 5 ZIP")
    official_check_bundle = fresh_official_adc_check(remaining_bundle)
    from IPython.display import FileLink, display
    display(FileLink(str(official_check_bundle)))