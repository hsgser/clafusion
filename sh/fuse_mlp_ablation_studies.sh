#!/bin/bash
N_SAMPLES=200
# Change alignmet in function get_alignment_map() of align_layers.py
for seed in {0..4}
do
    echo "-- Seed = $seed"
    DATA_PATH=./mlp_ablation_studies/seed_$seed
    # add
    python main.py --gpu-id 0 --model-name mlpnet --n-epochs 10 --save-result-file seed_$seed/add.csv --exact --correction --ground-metric euclidean --weight-stats --activation-histograms --activation-mode raw --geom-ensemble-type acts --act-num-samples $N_SAMPLES --ground-metric-normalize none --activation-seed 21 --prelu-acts --recheck-acc --load-models $DATA_PATH --ckpt-type final --past-correction --not-squared --dist-normalize --print-distances
    # merge
    python main.py --gpu-id 0 --model-name mlpnet --n-epochs 10 --save-result-file seed_$seed/merge.csv --exact --correction --ground-metric euclidean --weight-stats --activation-histograms --activation-mode raw --geom-ensemble-type acts --act-num-samples $N_SAMPLES --ground-metric-normalize none --activation-seed 21 --prelu-acts --recheck-acc --load-models $DATA_PATH --ckpt-type final --past-correction --not-squared --dist-normalize --print-distances --balance-method merge --relu-approx-method avg
done