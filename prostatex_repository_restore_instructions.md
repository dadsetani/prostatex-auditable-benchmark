# Restore the PROSTATEx Repository ZIP

The repository package is split into five Base64 text files because Prism may remove a large generated ZIP or a single large Base64 file.

Download these files from the `analysis` directory and place them together in one local folder:

1. `prostatex_repository_base64_part_01.txt`
2. `prostatex_repository_base64_part_02.txt`
3. `prostatex_repository_base64_part_03.txt`
4. `prostatex_repository_base64_part_04.txt`
5. `prostatex_repository_base64_part_05.txt`

The order and individual hashes are also recorded in `prostatex_repository_package_manifest.json`.

## Windows PowerShell

Open PowerShell in the folder containing the five downloaded files and run:

```powershell
$parts = 1..5 | ForEach-Object {
    Join-Path $PWD.Path ("prostatex_repository_base64_part_{0:D2}.txt" -f $_)
}
$base64Text = ($parts | ForEach-Object {
    (Get-Content -LiteralPath $_ -Raw).Trim()
}) -join ""
$outputFile = Join-Path $PWD.Path "prostatex_auditable_benchmark_repository.zip"
[IO.File]::WriteAllBytes(
    $outputFile,
    [Convert]::FromBase64String($base64Text)
)
Get-FileHash -Algorithm SHA256 $outputFile
```

## Linux or macOS

```bash
cat prostatex_repository_base64_part_*.txt \
  | tr -d '\n\r\t ' \
  | base64 --decode \
  > prostatex_auditable_benchmark_repository.zip

sha256sum prostatex_auditable_benchmark_repository.zip
```

On macOS, replace `base64 --decode` with `base64 -D` if necessary.

## Verification

Expected ZIP size: `1861620` bytes

Expected ZIP SHA-256:

```text
f742faa3d751423cfa8a01aa7e6ed1114ed2db927e7568169040d9b500657911
```

After verification, extract the ZIP. It contains the complete `prostatex-auditable-benchmark` directory with 155 files.