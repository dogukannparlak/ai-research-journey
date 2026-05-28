# Install Git LFS (Windows), enable hooks, and download LFS content for this repo.
param(
    [string]$Root = (Split-Path -Parent $PSScriptRoot)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Ensure-GitLfs {
    if (Get-Command git-lfs -ErrorAction SilentlyContinue) {
        return
    }

    try {
        $null = git lfs version 2>&1
        return
    } catch {
        # continue to install
    }

    Write-Host "Git LFS not found. Installing with winget..."
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Error @"
Git LFS is not installed and winget is unavailable.
Install Git LFS manually:
  winget install GitHub.GitLFS
Or download from: https://git-lfs.com
"@
    }

    winget install --id GitHub.GitLFS -e --accept-package-agreements --accept-source-agreements
}

function Set-AutoCrlfIfNeeded {
    $current = git config --global core.autocrlf 2>$null
    if ($current -eq "true") {
        Write-Host "Setting global core.autocrlf=false (was true) to avoid binary file corruption."
        git config --global core.autocrlf false
    }
}

Push-Location $Root
try {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Error "git is not installed or not on PATH."
    }

    Ensure-GitLfs
    Set-AutoCrlfIfNeeded

    Write-Host "Enabling Git LFS hooks..."
    git lfs install

    Write-Host "Downloading LFS objects..."
    git lfs pull

    Write-Host "Verifying PDFs..."
    & (Join-Path $PSScriptRoot "verify-lfs.ps1") -Root $Root
}
finally {
    Pop-Location
}
