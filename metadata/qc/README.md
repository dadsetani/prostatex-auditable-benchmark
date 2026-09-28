# Blinded Technical Visual QC

Review `qc_preview.pdf` in `review_id` order and complete `review.csv`.
The preview and worksheet omit patient identity, finding identity, fold, zone, and ClinSig label.
Do not open the separately retained identity key until the visual statuses are frozen.

Allowed `visual_qc_status` values:
- `PENDING`
- `NO_OBVIOUS_TECHNICAL_ISSUE`
- `REVIEW_REQUIRED`
- `UNASSESSABLE`

Check only for obvious technical display/content problems, such as a blank or nearly constant
patch, severe interpolation failure, gross truncation, or an unusable channel. Independent
percentile windows are display-only. Differences between T2 and ADC intensity or appearance
are not by themselves failures. The center marker is the supplied finding position, not an
independently verified lesion center. Do not infer labels, diagnosis, registration quality,
or clinical suitability. No visual status automatically changes dataset eligibility.