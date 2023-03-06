#!/bin/bash
FUSION=ot
N_SAMPLES=200
RETRAIN=120
DECAY=5
DECAY_FACTOR=2.0
DECAY_EPOCHS=20_40_60_80_100
for seed in {40..44}
do
    echo "----- seed = $seed"
    DATA_PATH=./vgg13_doub_vgg11_half/seed_$seed
    python main.py --gpu-id 0 --model-name vgg13_doub_nobias --second-model-name vgg11_half_nobias --n-epochs 300 --save-result-file ${FUSION}_fusion_$seed.csv --exact --correction --ground-metric euclidean --weight-stats --activation-histograms --activation-mode raw --geom-ensemble-type acts --act-num-samples $N_SAMPLES --ground-metric-normalize none --activation-seed 21 --prelu-acts --recheck-cifar --load-models $DATA_PATH --ckpt-type best --dataset cifar10 --past-correction --not-squared --dist-normalize --print-distances --layer-measure activation --layer-metric cka --retrain $RETRAIN --retrain-lr-decay $DECAY --retrain-lr-decay-factor $DECAY_FACTOR --retrain-lr-decay-epochs $DECAY_EPOCHS --fusion-method ${FUSION}
done