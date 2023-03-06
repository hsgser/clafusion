#!/bin/bash
N_SAMPLES=200
SEED=42
DATASET=cifar10
DATA_PATH=./resnet34_resnet34_half_${DATASET}/seed_$SEED
RETRAIN=120
DECAY=1
DECAY_FACTOR=2.0
DECAY_EPOCHS=20_40_60_80_100
TEMPERATURE=20 # 20, 10, 8, 4, 1
for ALPHA in 0.05 0.1 0.5 0.7 0.95 0.99
do
    echo "ALPHA = $ALPHA"
    python distillation_big_only.py --gpu-id 0 --model-name resnet34_nobias_nobn --second-model-name resnet34_half_nobias_nobn --n-epochs 300 --save-result-file distill_resnet_temp${TEMPERATURE}.csv --exact --correction --ground-metric euclidean --weight-stats --activation-histograms --activation-mode raw --geom-ensemble-type acts --act-num-samples $N_SAMPLES --ground-metric-normalize none --activation-seed 21 --prelu-acts --recheck-cifar --load-models $DATA_PATH --ckpt-type best --dataset $DATASET --past-correction --not-squared --dist-normalize --print-distances --layer-measure activation --layer-metric cka --retrain $RETRAIN --retrain-lr-decay $DECAY --retrain-lr-decay-factor $DECAY_FACTOR --retrain-lr-decay-epochs $DECAY_EPOCHS --handle-skips --ground-metric-eff --alpha $ALPHA --temperature $TEMPERATURE
done