#!/usr/bin/env bash
# Downloads and unpacks the datasets (Caltech101, DTD, Oxford Pets) into ../DATA,
# one level above the repository, which is where the configs expect them.
set -euo pipefail

FILE_ID='1WoXpegQrOLPW8ndYbfumP4HMAB0d0WEq'
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/../DATA}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

REQUIRED_DIRS=('caltech-101' 'dtd' 'oxford_pets')

data_ready() {
    for d in "${REQUIRED_DIRS[@]}"; do
        [ -d "$DATA_DIR/$d" ] || return 1
    done
    return 0
}

if data_ready; then
    echo "Datasets already present in $DATA_DIR, nothing to do."
    exit 0
fi

command -v gdown >/dev/null 2>&1 || { echo 'gdown not found: activate the conda environment first.'; exit 1; }

echo 'Downloading datasets...'
gdown --id "$FILE_ID" -O "$TMP_DIR/data_archive"

echo 'Unpacking...'
mkdir -p "$TMP_DIR/unpacked"
if tar -tf "$TMP_DIR/data_archive" >/dev/null 2>&1; then
    tar -xf "$TMP_DIR/data_archive" -C "$TMP_DIR/unpacked"
elif unzip -tq "$TMP_DIR/data_archive" >/dev/null 2>&1; then
    unzip -q "$TMP_DIR/data_archive" -d "$TMP_DIR/unpacked"
else
    echo 'Downloaded file is neither tar nor zip. Is the Drive link public ("anyone with the link")?'
    exit 1
fi

# The archive may or may not have a wrapper folder: find the one containing caltech-101
SRC="$(find "$TMP_DIR/unpacked" -maxdepth 3 -type d -name 'caltech-101' -printf '%h\n' | head -n 1)"
if [ -z "$SRC" ]; then
    echo 'caltech-101 not found inside the archive.'
    exit 1
fi

mkdir -p "$DATA_DIR"
cp -r "$SRC"/. "$DATA_DIR"/

if data_ready; then
    echo "Done: datasets are in $DATA_DIR"
else
    echo "Unpacked, but some of these are missing from $DATA_DIR: ${REQUIRED_DIRS[*]}"
    exit 1
fi