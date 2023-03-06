#!/bin/bash
FUSION=ot
MAP=cla
DATASET=cifar10 # cifar10, cifar100, tinyimagenet, esc50
if [ $DATASET = 'tinyimagenet' ]
then
    N_SAMPLES=400
    RETRAIN=90
    DECAY=1
    DECAY_FACTOR=10.0
    DECAY_EPOCHS=30_60_90
    for seed in {40..44}
    do
        echo "----- seed = $seed"
        DATA_PATH=./resnet34_resnet18_${DATASET}/seed_$seed
        # Skip ensemble learning to avoid OOM
        python main.py --gpu-id 0 --model-name resnet34_nobias_nobn --second-model-name resnet18_nobias_nobn --n-epochs 300 --save-result-file ${FUSION}_fusion_$seed.csv --exact --correction --ground-metric euclidean --weight-stats --activation-histograms --activation-mode raw --geom-ensemble-type acts --act-num-samples $N_SAMPLES --ground-metric-normalize none --activation-seed 21 --prelu-acts --load-models $DATA_PATH --ckpt-type best --dataset $DATASET --past-correction --not-squared --dist-normalize --print-distances --layer-measure activation --layer-metric cka --retrain $RETRAIN --retrain-lr-decay $DECAY --retrain-lr-decay-factor $DECAY_FACTOR --retrain-lr-decay-epochs $DECAY_EPOCHS --handle-skips --ground-metric-eff --dump-final-models --fusion-method ${FUSION} --mapping-type ${MAP} --skip-ensemble
        # Run ensemble learning
        python main.py --gpu-id 0 --model-name resnet34_nobias_nobn --second-model-name resnet18_nobias_nobn --n-epochs 300 --save-result-file ${FUSION}_fusion_$seed.csv --exact --correction --ground-metric euclidean --weight-stats --activation-histograms --activation-mode raw --geom-ensemble-type acts --act-num-samples $N_SAMPLES --ground-metric-normalize none --activation-seed 21 --prelu-acts --load-models $DATA_PATH --ckpt-type best --dataset $DATASET --past-correction --not-squared --dist-normalize --print-distances --layer-measure activation --layer-metric cka --retrain $RETRAIN --retrain-lr-decay $DECAY --retrain-lr-decay-factor $DECAY_FACTOR --retrain-lr-decay-epochs $DECAY_EPOCHS --handle-skips --ground-metric-eff --dump-final-models --fusion-method ${FUSION} --mapping-type ${MAP} --ensemble-only
    done
elif [ $DATASET = 'esc50' ]
then
    N_SAMPLES=200
    RETRAIN=90
    DECAY=1
    DECAY_FACTOR=10.0
    DECAY_EPOCHS=30_60_90
    for seed in {40..44}
    do
        echo "----- seed = $seed"
        DATA_PATH=./resnet34_resnet18_${DATASET}/seed_$seed
        # Skip ensemble learning to avoid OOM
        python main.py --gpu-id 0 --model-name resnet34_nobias_nobn --second-model-name resnet18_nobias_nobn --n-epochs 300 --save-result-file ${FUSION}_fusion_$seed.csv --exact --correction --ground-metric euclidean --weight-stats --activation-histograms --activation-mode raw --geom-ensemble-type acts --act-num-samples $N_SAMPLES --ground-metric-normalize none --activation-seed 21 --prelu-acts --load-models $DATA_PATH --ckpt-type best --dataset $DATASET --past-correction --not-squared --dist-normalize --print-distances --layer-measure activation --layer-metric cka --retrain $RETRAIN --retrain-lr-decay $DECAY --retrain-lr-decay-factor $DECAY_FACTOR --retrain-lr-decay-epochs $DECAY_EPOCHS --handle-skips --ground-metric-eff --dump-final-models --fusion-method ${FUSION} --mapping-type ${MAP} --skip-ensemble
        # Run ensemble learning
        python main.py --gpu-id -1 --model-name resnet34_nobias_nobn --second-model-name resnet18_nobias_nobn --n-epochs 300 --save-result-file ${FUSION}_fusion_$seed.csv --exact --correction --ground-metric euclidean --weight-stats --activation-histograms --activation-mode raw --geom-ensemble-type acts --act-num-samples $N_SAMPLES --ground-metric-normalize none --activation-seed 21 --prelu-acts --load-models $DATA_PATH --ckpt-type best --dataset $DATASET --past-correction --not-squared --dist-normalize --print-distances --layer-measure activation --layer-metric cka --retrain $RETRAIN --retrain-lr-decay $DECAY --retrain-lr-decay-factor $DECAY_FACTOR --retrain-lr-decay-epochs $DECAY_EPOCHS --handle-skips --ground-metric-eff --dump-final-models --fusion-method ${FUSION} --mapping-type ${MAP} --ensemble-only
    done
else
    N_SAMPLES=200
    RETRAIN=0
    DECAY=1
    DECAY_FACTOR=2.0
    DECAY_EPOCHS=20_40_60_80_100
    for seed in {40..44}
    do
        echo "----- seed = $seed"
        DATA_PATH=./resnet34_resnet18_${DATASET}/seed_$seed
        python main.py --gpu-id 1 --model-name resnet34_nobias_nobn --second-model-name resnet18_nobias_nobn --n-epochs 300 --save-result-file ${FUSION}_fusion_$seed.csv --exact --correction --ground-metric euclidean --weight-stats --activation-histograms --activation-mode raw --geom-ensemble-type acts --act-num-samples $N_SAMPLES --ground-metric-normalize none --activation-seed 21 --prelu-acts --load-models $DATA_PATH --ckpt-type best --dataset $DATASET --past-correction --not-squared --dist-normalize --print-distances --layer-measure activation --layer-metric cka --retrain $RETRAIN --retrain-lr-decay $DECAY --retrain-lr-decay-factor $DECAY_FACTOR --retrain-lr-decay-epochs $DECAY_EPOCHS --handle-skips --ground-metric-eff --dump-final-models --fusion-method ${FUSION} --mapping-type ${MAP}
    done
fi
