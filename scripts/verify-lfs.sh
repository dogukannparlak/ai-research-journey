#!/usr/bin/env bash
# Verify that LFS-tracked PDFs are real PDF files, not Git LFS pointer stubs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POINTER_PREFIX="version https://git-lfs.github.com/spec/v1"
SAMPLE_PDF="History/2010s/2012/2012 - ImageNet Classification with Deep Convolutional Neural Networks - Krizhevsky et al..pdf"

cd "$ROOT"

if ! command -v git >/dev/null 2>&1; then
  echo "error: git is not installed or not on PATH." >&2
  exit 1
fi

if ! git lfs version >/dev/null 2>&1; then
  echo "error: Git LFS is not installed. Run: ./scripts/setup-lfs.sh" >&2
  exit 1
fi

mapfile -t pdf_files < <(git ls-files "*.pdf")
if [ "${#pdf_files[@]}" -eq 0 ]; then
  echo "No tracked PDF files found."
  exit 0
fi

failures=0

for relative_path in "${pdf_files[@]}"; do
  if [ ! -f "$relative_path" ]; then
    echo "FAIL: $relative_path — file missing from working tree"
    failures=$((failures + 1))
    continue
  fi

  first_line="$(head -n 1 "$relative_path")"

  if [[ "$first_line" == "$POINTER_PREFIX"* ]]; then
    echo "FAIL: $relative_path — LFS pointer not downloaded (run: git lfs pull)"
    failures=$((failures + 1))
    continue
  fi

  if [[ "$first_line" != %PDF-* ]]; then
    echo "FAIL: $relative_path — unexpected header: $first_line"
    failures=$((failures + 1))
  fi
done

if [ "$failures" -gt 0 ]; then
  echo "LFS verification failed for $failures file(s)." >&2
  exit 1
fi

echo "OK: All ${#pdf_files[@]} tracked PDFs are valid (start with %PDF-)."
if [ -f "$SAMPLE_PDF" ]; then
  echo "Sample: $(head -n 1 "$SAMPLE_PDF")"
fi
