# Verify that LFS-tracked PDFs are real PDF files, not Git LFS pointer stubs.
param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Push-Location $Root
try {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Error "git is not installed or not on PATH."
    }

    if (-not (Get-Command git-lfs -ErrorAction SilentlyContinue)) {
        Write-Error "Git LFS is not installed. Run: git lfs install"
    }

    $pdfFiles = git ls-files "*.pdf"
    if (-not $pdfFiles) {
        Write-Host "No tracked PDF files found."
        exit 0
    }

    $pointerPrefix = "version https://git-lfs.github.com/spec/v1"
    $failures = @()

    foreach ($relativePath in $pdfFiles) {
        if (-not (Test-Path -LiteralPath $relativePath)) {
            $failures += [PSCustomObject]@{
                Path = $relativePath
                Issue = "File missing from working tree"
            }
            continue
        }

        $firstLine = Get-Content -LiteralPath $relativePath -TotalCount 1 -ErrorAction Stop

        if ($firstLine -like "$pointerPrefix*") {
            $failures += [PSCustomObject]@{
                Path = $relativePath
                Issue = "LFS pointer not downloaded (run: git lfs pull)"
            }
            continue
        }

        if ($firstLine -notlike "%PDF-*") {
            $failures += [PSCustomObject]@{
                Path = $relativePath
                Issue = "Unexpected header: $firstLine"
            }
        }
    }

    if ($failures.Count -gt 0) {
        Write-Host "LFS verification failed for $($failures.Count) file(s):" -ForegroundColor Red
        $failures | Format-Table -AutoSize
        exit 1
    }

    Write-Host "OK: All $($pdfFiles.Count) tracked PDFs are valid (start with %PDF-)." -ForegroundColor Green
    exit 0
}
finally {
    Pop-Location
}
