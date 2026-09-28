# Reproducibility Guide

## 1. Frozen Inputs

Use the following records without modification:

- `config/prostatex_selection_protocol_v0_3.json`
- `config/prostatex_modeling_protocol_v0_1.json`
- `config/prostatex_classical_baselines_protocol_v0_1.json`
- `metadata/folds/prostatex_patient_folds_v0_3.csv`
- `metadata/eligibility/prostatex_eligibility_register_v0_3.csv`

## 2. Data Retrieval

The retrieval utility accepts a fold number, frozen plan, label archive, and selection protocol:

```bash
python src/prostatex_fold_retrieval_kaggle.py \
  0 metadata/prostatex_download_plan_v0_3.csv \
  PATH_TO_LABEL_ARCHIVE.zip \
  config/prostatex_selection_protocol_v0_3.json
```

Repeat for folds 0 through 4. Verify every produced archive against its manifest before patch extraction.

## 3. Patch Extraction

The extraction script expects the downloaded series bundles and the frozen analysis records:

```bash
python src/prostatex_extract_frozen_patches.py 0 \
  --upload-dir PATH_TO_RETRIEVED_ARCHIVES \
  --analysis-dir metadata \
  --output-root PATH_TO_PATCH_OUTPUT
```

The original workspace used a flatter analysis-directory layout. If reconstructing directly from this organized repository, either stage the required metadata files in one input directory or pass paths after reviewing the script constants. Do not change the frozen eligibility or fold assignments.

## 4. CNN Baselines

```bash
python src/prostatex_run_cnn_baselines.py \
  --patch-root PATH_TO_FROZEN_PATCH_BUNDLES \
  --output-root model-runs/cnn \
  --all --validate-inputs

python src/prostatex_summarize_cnn_baselines.py \
  --run-root model-runs/cnn \
  --output-dir results/generated/cnn-summary
```

## 5. Classical Baselines

```bash
python src/prostatex_run_classical_baselines.py \
  --patch-root PATH_TO_FROZEN_PATCH_BUNDLES \
  --output-root model-runs/classical \
  --all --validate-inputs

python src/prostatex_summarize_classical_baselines.py \
  --run-root model-runs/classical \
  --output-dir results/generated/classical-summary
```

## 6. Comparisons and Figures

The comparison scripts consume summary ZIP archives produced by the summarization steps:

```bash
python src/prostatex_compare_cnn_baselines.py \
  --summary-archive PATH_TO_CNN_SUMMARY.zip \
  --output-dir results/generated/cnn-comparison \
  --replicates 2000

python src/prostatex_compare_classical_and_cnn.py \
  --classical-archive PATH_TO_CLASSICAL_SUMMARY.zip \
  --cnn-archive PATH_TO_CNN_SUMMARY.zip \
  --output-dir results/generated/classical-comparison \
  --replicates 2000
```

Compare regenerated JSON and CSV outputs with `results/summaries/` and `results/predictions/`. Preserve patient-level resampling and the frozen random seeds.