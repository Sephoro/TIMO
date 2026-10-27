#!/bin/bash

TIMO_MODE="${1:-both}"

if [[ ! "$TIMO_MODE" =~ ^(tgi|igt|both)$ ]]; then
    echo "Usage: $0 {tgi|igt|both}"
    exit 1
fi

for DATA in caltech101 dtd oxford_pets
do
    for SHOT in 1 4 16
    do
        for SEED in 1 2 3
        do
            CUDA_VISIBLE_DEVICES=0 python main.py \
                --config configs/${DATA}.yaml \
                --shot ${SHOT} \
                --seed ${SEED} \
                --timo_mode "${TIMO_MODE}"
        done
    done
done