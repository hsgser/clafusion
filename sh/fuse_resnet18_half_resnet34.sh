#!/bin/bash
FUSION=ot
MAP=cla
N_SAMPLES=200
RETRAIN=120
DECAY=1
DECAY_FACTOR=2.0
DECAY_EPOCHS=20_40_60_80_100
DATASET=cifar100 # cifar10, cifar100
for seed in {40..44}
do
    echo "----- seed = $seed"
    DATA_PATH=./resnet34_resnet18_half_${DATASET}/seed_$seed
    python main.py --gpu-id 0 --model-name resnet34_nobias_nobn --second-model-name resnet18_half_nobias_nobn --n-epochs 300 --save-result-file ${FUSION}_fusion_$seed.csv --exact --correction --ground-metric euclidean --weight-stats --activation-histograms --activation-mode raw --geom-ensemble-type acts --act-num-samples $N_SAMPLES --ground-metric-normalize none --activation-seed 21 --prelu-acts --recheck-cifar --load-models $DATA_PATH --ckpt-type best --dataset $DATASET --past-correction --not-squared --dist-normalize --print-distances --layer-measure activation --layer-metric cka --retrain $RETRAIN --retrain-lr-decay $DECAY --retrain-lr-decay-factor $DECAY_FACTOR --retrain-lr-decay-epochs $DECAY_EPOCHS --handle-skips --ground-metric-eff --dump-final-models --fusion-method ${FUSION} --mapping-type ${MAP}
done