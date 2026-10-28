#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root, regardless of the caller's directory.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Expected final output from the extraction process.
DONE_MARKER="caches/RN50/3/oxford_pets/text_weights_cupl_t_all.pt"

# Skip extraction if the marker exists, unless forced.
if [[ -f "$DONE_MARKER" && "${FORCE:-0}" != "1" ]]; then
    echo "Feature cache marker found. Skipping extraction."
    echo "Set FORCE=1 to run extraction again."
    exit 0
fi

echo "Extracting features using the device selected by the Python script..."

python extract_features_all.py

# Verify that the expected completion marker exists after extraction.
if [[ ! -f "$DONE_MARKER" ]]; then
    echo "Error: Expected completion marker was not created:"
    echo "  $DONE_MARKER"
    echo "Feature extraction may not have completed successfully."
    exit 1
fi

echo "Feature extraction finished."