#!/bin/bash
N_EPOCHS=10
for seed in {0..4}
do
    echo "Seed = $seed"
    # 400-200-100-50-25 and 400-200-100
    python main.py --gpu-id 0 --model-name mlpnet --n-epochs $N_EPOCHS --save-result-file sample.csv --ckpt-type final --dump-final-models --train-only --num-models 2 --net-config "400 200 100 50 25; 400 200 100" --train-seed $seed
done