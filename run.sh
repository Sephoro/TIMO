#!/usr/bin/env bash
set -euo pipefail

# Defaults
TIMO_MODE="base"
NOISE="0"

# Parse arguments
if [[ $# -ge 1 && "$1" != "--noise" && "$1" != "--all-noise" ]]; then
    TIMO_MODE="$1"
    shift
fi

ALL_NOISE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --noise)
            if [[ $# -lt 2 || ! "$2" =~ ^[0-9]+$ ]]; then
                echo "Error: --noise requires a non-negative integer."
                exit 1
            fi
            NOISE="$2"
            shift 2
            ;;
        --all-noise)
            ALL_NOISE=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [base|tgi|igt|both] [--noise N | --all-noise]"
            exit 1
            ;;
    esac
done

# Validate mode
if [[ ! "$TIMO_MODE" =~ ^(base|tgi|igt|both)$ ]]; then
    echo "Invalid mode '$TIMO_MODE'. Choose base, tgi, igt, or both."
    exit 1
fi

# Select noise levels
if [[ "$ALL_NOISE" == true ]]; then
    NOISE_LEVELS=(2 4 8 16 32)
else
    NOISE_LEVELS=("$NOISE")
fi

# Run experiments
for DATA in caltech101 dtd oxford_pets; do
    for SHOT in 1 4 16; do
        for SEED in 1 2 3; do
            for NOISE_LEVEL in "${NOISE_LEVELS[@]}"; do
                echo "Dataset=$DATA Shot=$SHOT Seed=$SEED Mode=$TIMO_MODE Junk=$NOISE_LEVEL"

                python main.py \
                    --config "configs/${DATA}.yaml" \
                    --shot "$SHOT" \
                    --seed "$SEED" \
                    --timo_mode "$TIMO_MODE" \
                    --junk "$NOISE_LEVEL"
            done
        done
    done
done

echo "All experiments completed."