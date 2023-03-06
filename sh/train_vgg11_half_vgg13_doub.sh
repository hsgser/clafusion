#!/bin/bash
# change training seed in cifar/hyperparameters/*.py
python train_cifar_models.py cifar10@vgg11 half_nobias
python train_cifar_models.py cifar10@vgg13 doub_nobias