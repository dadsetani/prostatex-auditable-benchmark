# Data Access

The source MRI examinations and finding labels originate from the public PROSTATEx collection. Use the dataset citation in `manuscript/references.tex` to locate the authoritative collection record.

This repository does not redistribute raw DICOM images. The frozen download plan, selected study and series identifiers, finding coordinates, patient-level folds, and patch metadata support reconstruction from the authoritative source.

## Provenance

Kaggle was used only as a computational environment for selected workflow stages. MRI series were retrieved by series UID directly from the official TCIA NBIA API at `services.cancerimagingarchive.net`. The retrieval code rejects unexpected redirect hosts and audits archive CRC, DICOM identity, and SHA-256 values.

The source label archive and extracted findings table were hash-tracked. The provenance of the original label-file container was not independently established, as stated in the manuscript limitations.

## Licensing

The MIT License applies only to original source code. It does not relicense PROSTATEx/TCIA images, labels, identifiers, DICOM metadata, or derived imaging data. Redistribution remains subject to the applicable source terms and citation requirements.
