# PROSTATEx data-selection protocol v0.3

Status: **frozen for batched image retrieval** on 2026-09-26.

The ProstateX-0025 exception is resolved by full-grid geometry. One finding maps to study `1.3.6.1.4.1.14519.5.2.1.7311.5101.101137934734515856187339619047`; the other four map to study `1.3.6.1.4.1.14519.5.2.1.7311.5101.260163101010991858039197694584`. Both study pairs remain primary sources for this patient, which stays in outer fold 2.

The primary cohort contains 197 patients and 319 findings after excluding six development-QC patients and ProstateX-0191 from the paired primary analysis. The retrieval plan contains 396 series and is executed one frozen outer fold at a time. Post-download identity, encoding, geometry, intensity-semantic, frame, and complete-grid bounds checks are mandatory before patch extraction.