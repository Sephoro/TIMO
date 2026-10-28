#!/usr/bin/env bash
set -euo pipefail



FILE_ID='1eQozsqRSj8h8gPKtFsOc05shFKSTHakk'
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/../DATA}"
REQUIRED_DIRS=('caltech-101' 'dtd' 'oxford_pets')

data_ready() {
    for d in "${REQUIRED_DIRS[@]}"; do
        [ -d "$DATA_DIR/$d" ] || return 1
    done
}

if data_ready; then
    echo "Datasets already present in $DATA_DIR."
    exit 0
fi

command -v gdown >/dev/null 2>&1 || {
    echo "gdown not found. Activate the conda environment first."
    exit 1
}

mkdir -p "$DATA_DIR"

# Use temporary storage on the same filesystem as the destination.
TMP_DIR="$(mktemp -d -p "$DATA_DIR" .download.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Downloading data1.tar.gz..."
gdown --id "$FILE_ID" -O "$TMP_DIR/data1.tar.gz"

echo "Extracting archive..."
mkdir -p "$TMP_DIR/unpacked"
tar -xzf "$TMP_DIR/data1.tar.gz" -C "$TMP_DIR/unpacked"

# Locate the extracted DATA directory, since the archive contains
# the entire DATA folder rather than individual dataset folders.
SRC="$(find "$TMP_DIR/unpacked" -maxdepth 4 -type d -name 'DATA' -print -quit)"

# If the archive contains the dataset folders directly, use that directory.
if [ -z "$SRC" ]; then
    SRC="$TMP_DIR/unpacked"
fi

for d in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$SRC/$d" ]; then
        echo "Error: Expected dataset directory '$d' not found in $SRC."
        echo "Check the archive's internal directory structure."
        exit 1
    fi
done

echo "Moving datasets into $DATA_DIR..."
for d in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$DATA_DIR/$d" ]; then
        rm -rf "$DATA_DIR/$d"
    fi
    mv "$SRC/$d" "$DATA_DIR/$d"
done

# Move any loose files from the DATA folder, such as split JSON files.
find "$SRC" -maxdepth 1 -type f -exec mv -t "$DATA_DIR" -- {} +

if data_ready; then
    echo "Done! Datasets are available in $DATA_DIR."
else
    echo "Error: Some dataset directories are missing."
    exit 1
fi