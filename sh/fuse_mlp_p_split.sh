#!/bin/bash
FRAC=0.1
N_SAMPLES=400
BALANCE=merge # add, merge
for seed in {0..4}
do
    echo "Seed = $seed"
    DATA_PATH=./mlp_p_split_${FRAC}/seed_$seed
    for i in {0..10}
    do
        STEP=$(echo "$i / 10" | bc -l)
        echo "----- step = $STEP"
        python split_main.py --gpu-id 0 --model-name mlpnet --n-epochs 10 --save-result-file seed_$seed/${BALANCE}_constraint.csv --exact --correction --ground-metric euclidean --activation-histograms --activation-mode raw --geom-ensemble-type acts --act-num-samples $N_SAMPLES --ground-metric-normalize none --activation-seed 21 --prelu-acts --load-models $DATA_PATH --ckpt-type final --past-correction --not-squared --dist-normalize --skip-last-layer --load-personalized-datasets $DATA_PATH --balance-method $BALANCE --free-end-layers 1 --prediction-wts --ensemble-step $STEP --relu-approx-method sum
    done
done