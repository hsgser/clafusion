#!/bin/bash
PERSONAL_CLASS_IDX=4
N_EPOCHS=10
for seed in {0..4}
do
    echo "Seed = $seed"
    for i in 1
  	do
        FRAC=$(echo "$i / 10" | bc -l)
        echo "----- frac = $FRAC"
        python split_main.py --gpu-id 0 --model-name mlpnet --n-epochs $N_EPOCHS --save-result-file sample.csv --ckpt-type final --dump-final-models --dump-datasets --train-only --num-models 2 --partition-type personalized --personal-class-idx $PERSONAL_CLASS_IDX --net-config "400 200 100; 400 200 100 50" --partition-train-seed $seed --personal-split-frac $FRAC
  	done
done