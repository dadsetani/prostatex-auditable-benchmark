# External Artifact Policy

Full run archives, model binaries, raw DICOM series, and frozen patch arrays are not stored in Git. The machine-readable manifest in `artifacts/external-artifacts.json` records the verified archives available in the source workspace.

For a permanent release:

1. Upload full run and approved derived-data archives to Zenodo or a GitHub release.
2. Verify every uploaded file against its SHA-256 value.
3. Add the permanent DOI and download location to the manifest.
4. Update the manuscript Data and Code Availability Statement.
5. Do not publish raw images unless the source collection terms explicitly permit redistribution.