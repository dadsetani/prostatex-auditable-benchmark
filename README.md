# Auditable PROSTATEx T2/ADC Benchmark

This repository contains the manuscript, frozen cohort records, patient-level folds, processing and modeling code, out-of-fold predictions, statistical summaries, and audit records for an internal PROSTATEx benchmark of clinically significant prostate cancer classification from T2-weighted and ADC MRI patches.

## Scope

- Frozen assignments for 197 patients, followed by a technically eligible cohort of 191 patients and 306 accepted findings.
- 65 clinically significant and 241 non-significant findings.
- Five fixed patient-level outer folds.
- T2-only, ADC-only, and paired T2/ADC fixed CNN baselines.
- Matched logistic-regression, RBF-SVM, and random-forest controls.
- Patient-level bootstrap confidence intervals.

This is a reproducible computational benchmark, not an externally validated clinical diagnostic system.

## Repository Contents

- `manuscript/`: LaTeX manuscript and publication figures.
- `src/`: retrieval, eligibility, patch extraction, quality-control, modeling, summarization, and comparison scripts.
- `config/`: frozen selection and modeling protocols.
- `metadata/`: patient folds, eligibility records, source-series records, patch metadata, and technical QC records.
- `results/predictions/`: pooled out-of-fold CNN and classical prediction tables.
- `results/summaries/`: model summaries and paired comparisons.
- `results/audits/`: integrity and consistency audit records.
- `artifacts/`: checksums for larger archives stored outside Git.
- `docs/`: data-access and reproducibility guidance.

## Data Access

PROSTATEx source images are available from The Cancer Imaging Archive. Raw DICOM data are not redistributed in this repository. See `docs/data-access.md` and the dataset citation in the manuscript.

## Environment

Create an isolated Python environment using Python 3.12 or a compatible version, then install:

```bash
python -m pip install -r requirements.txt
```

The archived audits record the environments used during different acquisition and patch-processing stages. The modeling scripts additionally require PyTorch and scikit-learn.

## Main Workflow

1. Retrieve the required PROSTATEx series using the frozen download plan and selection protocol.
2. Reconstruct and audit the five frozen patch bundles.
3. Run the fixed CNN and classical baselines.
4. Summarize complete outer-fold predictions.
5. Regenerate bootstrap comparisons and figures.

Representative commands and expected inputs are documented in `docs/reproducibility.md`.

## Manuscript Build

```bash
cd manuscript
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Large Artifacts

Full model-run archives and frozen patch archives are intentionally excluded from Git. Their expected filenames, sizes, and SHA-256 checksums are listed in `artifacts/external-artifacts.json`. They should be attached to a GitHub release or deposited in a permanent archival service such as Zenodo.

## License

No software or data license has yet been selected. Add an appropriate code license and confirm the applicable PROSTATEx/TCIA terms before making the repository public.