#!/bin/bash
# change training seed in cifar/hyperparameters/*.py
DATASET=cifar10 # cifar10, cifar100, tinyimagenet, esc50
python train_cifar_models.py ${DATASET}@resnet18 nobias_nobn
python train_cifar_models.py ${DATASET}@resnet34 nobias_nobn