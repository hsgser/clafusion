#!/bin/bash
RETRAIN=1
DECAY=1
DECAY_FACTOR=1.0
DECAY_EPOCHS=30_60_90
DATASET=esc50 # cifar10, cifar100
for seed in {40..40}
do
    echo "----- seed = $seed"
    DATA_PATH=./resnet34_resnet18_${DATASET}_sgd/seed_${seed}
    CUDA_VISIBLE_DEVICES=0 taskset --cpu-list 00-09 python retrain_from_checkpoints.py --gpu-id 0 --num-models 2 --model-name-list "resnet34_nobias_nobn; resnet18_nobias_nobn" --save-result-file $seed.csv --load-models $DATA_PATH --ckpt-type best --dataset $DATASET --retrain $RETRAIN --retrain-lr-decay $DECAY --retrain-lr-decay-factor $DECAY_FACTOR --retrain-lr-decay-epochs $DECAY_EPOCHS --dump-final-models
done
