"""Freeze the completed AI-assisted blinded technical visual-QC pass."""

import csv
import hashlib
import json
from pathlib import Path


REVIEW_REQUIRED = {
    "BQC019": (
        "ADC patch is nearly blank in the blinded preview; recorded ADC zero fraction is 0.988. "
        "Retain as REVIEW_REQUIRED pending source-series and qualified-review investigation."
    )
}


def sha256_path(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finalize(analysis_dir="analysis"):
    analysis_dir = Path(analysis_dir)
    qc_dir = analysis_dir / "prostatex_blinded_visual_qc_v0_3"
    source = qc_dir / "review.csv"
    output = qc_dir / "review_ai_technical_frozen.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = handle.readline() if False else None
    fieldnames = list(rows[0])
    for row in rows:
        if row["review_id"] in REVIEW_REQUIRED:
            row["visual_qc_status"] = "REVIEW_REQUIRED"
            row["notes"] = REVIEW_REQUIRED[row["review_id"]]
        else:
            row["visual_qc_status"] = "NO_OBVIOUS_TECHNICAL_ISSUE"
            row["notes"] = (
                "No obvious blank-channel, severe interpolation, or gross truncation issue in the "
                "identity- and label-hidden full-cohort contact-sheet review."
            )
        row["reviewer"] = "OpenAI Codex AI-assisted technical visual pass"
        row["reviewed_utc"] = "2026-09-27T22:59:57Z"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    for row in rows:
        counts[row["visual_qc_status"]] = counts.get(row["visual_qc_status"], 0) + 1
    report = {
        "qc_id": "PROSTATEx-blinded-technical-visual-QC-v0.3",
        "frozen_date": "2026-09-27",
        "status": "COMPLETE_WITH_REVIEW_REQUIRED",
        "scope": "AI-assisted technical display review; not radiologist review, registration validation, diagnosis, or clinical image-quality assessment",
        "reviewed_findings": len(rows),
        "status_counts": counts,
        "review_required_ids": sorted(REVIEW_REQUIRED),
        "blinding": {
            "preview_omitted_identity": True,
            "preview_omitted_ClinSig": True,
            "review_csv_omitted_identity": True,
            "identity_key_retained_separately": True,
            "formal_independent_blinding_claimed": False,
        },
        "review_basis": [
            "Nine identity- and label-hidden contact sheets covering all 306 paired patches.",
            "Dedicated enlarged review of high ADC zero-fraction and low-variance cases.",
            "Per-channel descriptive intensity statistics; no label-dependent thresholding.",
        ],
        "completed_review": {
            "file": output.name,
            "bytes": output.stat().st_size,
            "sha256": sha256_path(output),
        },
        "decision_policy": "REVIEW_REQUIRED does not automatically exclude or modify a patch; investigate source data before freezing model inputs.",
        "targeted_investigation": "BQC019 was unblinded only after statuses were frozen; the native ADC footprint, not resampling, was confirmed to be nearly all zero.",
        "next_step": "Obtain qualified label-hidden review of ProstateX-0052 finding 1 and freeze a cohort-wide handling rule before model fitting.",
    }
    report_path = analysis_dir / "prostatex_blinded_visual_qc_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))
    print("Saved:", output, report_path)


if __name__ == "__main__":
    finalize()