#!/bin/bash
N_SAMPLES=200
FREE_END_LAYERS=1
LAYER_MEASURES=(neuron activation activation)
LAYER_METRICS=(euclidean cka wd)
for seed in {0..4}
do
    echo "-- Seed = $seed"
    DATA_PATH=./mlp_ablation_studies/seed_$seed
    for idx in {0..2}
    do
        LAYER_MEASURE=${LAYER_MEASURES[idx]}
        LAYER_METRIC=${LAYER_METRICS[idx]}
        echo "---- Layer measure = $LAYER_MEASURE and layer metric = $LAYER_METRIC"
        for START_LAYER_IDX in {1..2}
        do
            echo "------ Align from hidden layer $START_LAYER_IDX"
            for FREE_END_LAYERS in {0..1}
            do
                echo "-------- Free $FREE_END_LAYERS last layers"
                python main.py --gpu-id 0 --model-name mlpnet --n-epochs 10 --save-result-file seed_$seed/alignment.csv --exact --correction --ground-metric euclidean --weight-stats --activation-histograms --activation-mode raw --geom-ensemble-type acts --act-num-samples $N_SAMPLES --ground-metric-normalize none --activation-seed 21 --prelu-acts --recheck-acc --load-models $DATA_PATH --ckpt-type final --past-correction --not-squared --dist-normalize --print-distances --layer-measure $LAYER_MEASURE --layer-metric $LAYER_METRIC --ground-metric-eff --start-layer-idx $START_LAYER_IDX --free-end-layers $FREE_END_LAYERS
            done
        done
    done
done