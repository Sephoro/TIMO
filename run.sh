for DATA in caltech101 dtd oxford_pets
do 
    for SHOOT in 1 4 16
    do
        for SEED in 1 2 3
        do
            CUDA_VISIBLE_DEVICES=0 python main.py --config configs/${DATA}.yaml --shot ${SHOOT} --seed ${SEED} 
        done
    done
done