import os
import sys

import hyperparameters.vgg11_cifar10_baseline as vgg_hyperparams
import train as cifar_train
from basic_config import PATH_TO_CIFAR
from log import logger


sys.path.append(PATH_TO_CIFAR)


exp_path = sys.argv[1]
gpu_id = int(sys.argv[2])
logger.info("gpu_id is %d", gpu_id)
logger.info("exp_path is %s", exp_path)

config = vgg_hyperparams.config

model_types = ["model_0", "model_1", "geometric", "naive_averaging"]
for model in model_types:
    for ckpt in ["best", "final"]:
        if os.path.exists(os.path.join(exp_path, model)):
            cifar_train.get_pretrained_model(
                config, os.path.join(exp_path, model, ckpt + ".checkpoint"), device_id=gpu_id
            )
