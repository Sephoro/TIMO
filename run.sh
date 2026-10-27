#!/bin/bash

TIMO_MODE="${1:-both}"
NOISE=0

if [[ ! "$TIMO_MODE" =~ ^(tgi|igt|both)$ ]]; then
    echo "Usage: $0 {tgi|igt|both} [--noise NUMBER]"
    exit 1
fi

# Optional noise flag
if [[ "${2:-}" == "--noise" ]]; then
    NOISE="${3:-}"

    if [[ ! "$NOISE" =~ ^[0-9]+$ ]] || [[ $# -ne 3 ]]; then
        echo "Usage: $0 {tgi|igt|both} [--noise NUMBER]"
        echo "NUMBER must be a non-negative integer."
        exit 1
    fi
elif [[ $# -gt 1 ]]; then
    echo "Usage: $0 {tgi|igt|both} [--noise NUMBER]"
    exit 1
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