#!/bin/bash

# Defaults
TIMO_MODE="base"
NOISE=0

# Optional mode argument
if [[ $# -ge 1 && "$1" != "--noise" ]]; then
    TIMO_MODE="$1"
fi

# Validate mode
if [[ ! "$TIMO_MODE" =~ ^(base|tgi|igt|both)$ ]]; then
    echo "Invalid mode '$TIMO_MODE'. Using 'base' instead."
    TIMO_MODE="base"
fi

# Optional noise flag
if [[ "${2:-}" == "--noise" ]]; then
    if [[ "${3:-}" =~ ^[0-9]+$ ]]; then
        NOISE="$3"
    else
        echo "Invalid or missing noise value. Using 0."
    fi
elif [[ "${1:-}" == "--noise" ]]; then
    if [[ "${2:-}" =~ ^[0-9]+$ ]]; then
        NOISE="$2"
    else
        echo "Invalid or missing noise value. Using 0."
    fi
fi

for DATA in caltech101 dtd oxford_pets
do
    for SHOT in 1 4 16
    do
        for SEED in 1 2 3
        do
            python main.py \
                --config "configs/${DATA}.yaml" \
                --shot "${SHOT}" \
                --seed "${SEED}" \
                --timo_mode "${TIMO_MODE}" \
                --junk "${NOISE}"
        done
    done
done