#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Configuration
# ============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

DATA_SCRIPT="${DATA_SCRIPT:-$REPO_ROOT/setup_data.sh}"
FEATURE_SCRIPT="${FEATURE_SCRIPT:-$REPO_ROOT/extract_features.sh}"
EXPERIMENT_SCRIPT="${EXPERIMENT_SCRIPT:-$REPO_ROOT/run.sh}"
ANALYSIS_SCRIPT="${ANALYSIS_SCRIPT:-$REPO_ROOT/analyse_results.py}"

STATE_DIR="$REPO_ROOT/.pipeline_state"
mkdir -p "$STATE_DIR"

MODES=(base tgi igt both)
NOISE_LEVELS=(2 4 8 16 32)

# ============================================================
# Error handling
# ============================================================

CURRENT_STAGE="initialisation"

on_error() {
    local exit_code=$?

    echo
    echo "ERROR: Pipeline failed during: $CURRENT_STAGE"
    echo "Exit code: $exit_code"
    echo "Fix the problem and run this script again."
    echo "Stages marked complete will be skipped."
    exit "$exit_code"
}

trap on_error ERR

# ============================================================
# Helpers
# ============================================================

check_script() {
    if [[ ! -f "$1" ]]; then
        echo "Error: Required script not found: $1" >&2
        exit 1
    fi
}

run_stage() {
    local stage="$1"
    shift

    local marker="$STATE_DIR/$stage.done"

    if [[ -f "$marker" ]]; then
        echo
        echo "Skipping completed stage: $stage"
        return 0
    fi

    CURRENT_STAGE="$stage"

    echo
    echo "============================================================"
    echo "Starting stage: $stage"
    echo "============================================================"

    "$@"

    touch "$marker"
    echo "Completed stage: $stage"
}

# ============================================================
# Validate required scripts
# ============================================================

check_script "$DATA_SCRIPT"
check_script "$FEATURE_SCRIPT"
check_script "$EXPERIMENT_SCRIPT"
check_script "$ANALYSIS_SCRIPT"

# ============================================================
# Stage 1: Download datasets
# ============================================================

run_stage "download" bash "$DATA_SCRIPT"

# ============================================================
# Stage 2: Extract features
# ============================================================

run_stage "features" bash "$FEATURE_SCRIPT"

# ============================================================
# Stage 3: Baseline and ablation experiments
# Order: base -> tgi -> igt -> both
# ============================================================

for MODE in "${MODES[@]}"; do
    run_stage "experiment_$MODE" \
        bash "$EXPERIMENT_SCRIPT" "$MODE"
done

# ============================================================
# Stage 4: Junk-prompt robustness experiments
# Each noise level gets its own checkpoint.
# ============================================================

for NOISE in "${NOISE_LEVELS[@]}"; do
    run_stage "experiment_junk_$NOISE" \
        bash "$EXPERIMENT_SCRIPT" base --noise "$NOISE"
done

# ============================================================
# Stage 5: Analyse results
# Analysis is inexpensive, so rerun it to refresh the CSVs.
# ============================================================

for MODE in "${MODES[@]}"; do
    CURRENT_STAGE="analysis_$MODE"

    echo
    echo "============================================================"
    echo "Analysing $MODE results"
    echo "============================================================"

    python "$ANALYSIS_SCRIPT" --mode "$MODE"
done

CURRENT_STAGE="analysis_junk"

echo
echo "============================================================"
echo "Analysing junk-prompt results"
echo "============================================================"

python "$ANALYSIS_SCRIPT" --mode junk

# ============================================================
# Finished
# ============================================================

echo
echo "============================================================"
echo "ALL EXPERIMENTS AND ANALYSIS COMPLETED"
echo "============================================================"
echo "Raw results:      $REPO_ROOT/outputs/"
echo "Analysis results: $REPO_ROOT/outputs/analysis/"
echo "Pipeline state:   $STATE_DIR/"