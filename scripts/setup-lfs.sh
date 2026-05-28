#!/usr/bin/env bash
# Install Git LFS (macOS/Linux), enable hooks, and download LFS content for this repo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

install_git_lfs() {
  if git lfs version >/dev/null 2>&1; then
    return 0
  fi

  echo "Git LFS not found. Installing..."

  case "$(uname -s)" in
    Darwin)
      if command -v brew >/dev/null 2>&1; then
        brew install git-lfs
      else
        echo "error: Homebrew not found. Install Git LFS manually: https://git-lfs.com" >&2
        exit 1
      fi
      ;;
    Linux)
      if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y git-lfs
      elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y git-lfs
      elif command -v yum >/dev/null 2>&1; then
        sudo yum install -y git-lfs
      elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -S --noconfirm git-lfs
      elif command -v zypper >/dev/null 2>&1; then
        sudo zypper install -y git-lfs
      else
        echo "error: Unsupported package manager. Install Git LFS manually: https://git-lfs.com" >&2
        exit 1
      fi
      ;;
    *)
      echo "error: Unsupported OS. Install Git LFS manually: https://git-lfs.com" >&2
      exit 1
      ;;
  esac
}

configure_autocrlf() {
  current="$(git config --global core.autocrlf || true)"
  if [ "$current" = "true" ]; then
    echo "Setting global core.autocrlf=false (was true) to avoid binary file corruption."
    git config --global core.autocrlf false
  fi
}

cd "$ROOT"

if ! command -v git >/dev/null 2>&1; then
  echo "error: git is not installed or not on PATH." >&2
  exit 1
fi

install_git_lfs
configure_autocrlf

echo "Enabling Git LFS hooks..."
git lfs install

echo "Downloading LFS objects..."
git lfs pull

echo "Verifying PDFs..."
bash "$ROOT/scripts/verify-lfs.sh"
