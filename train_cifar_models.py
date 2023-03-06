import copy
import importlib
import os
import sys

import train as cifar_train
import utils
from basic_config import PATH_TO_CIFAR, TMP_DATETIME_FILE
from log import get_first_timestamp, logger
from tensorboardX import SummaryWriter


sys.path.append(PATH_TO_CIFAR)


num_models = 1
TENSORBOARD_ROOT = "./tensorboard"


def main():
    gpus = [0] * num_models

    if len(sys.argv) >= 2:
        dataset = str(sys.argv[1])
        if "@" in dataset:
            dataset, architecture_type = dataset.split("@")
        else:
            architecture_type = "vgg11"
    else:
        dataset = "cifar10"
        architecture_type = "vgg11"

    if len(sys.argv) >= 3:
        sub_type = str(sys.argv[2]) + "_"
        sub_type_str = str(sys.argv[2])
    else:
        sub_type = ""
        sub_type_str = "plain"

    if len(sys.argv) >= 4:
        gpu_num = int(sys.argv[3])
        gpus = [gpu_num] * num_models

    if dataset.lower()[0:7] == "cifar10":
        config_file = importlib.import_module(f"hyperparameters.{architecture_type}_{sub_type}cifar10_baseline")
    else:
        config_file = importlib.import_module(f"hyperparameters.{architecture_type}_{sub_type}{dataset}_baseline")
    base_config = config_file.config
    logger.info("gpus are {}".format(gpus))
    logger.info(f"Dataset is {dataset} and sub_type is {sub_type_str}")

    timestamp = get_first_timestamp()
    exp_name = "exp_{}_{}_{}_{}".format(
        dataset,
        architecture_type,
        sub_type_str,
        timestamp,
    )
    tensorboard_dir = os.path.join(TENSORBOARD_ROOT, exp_name)
    utils.mkdir(tensorboard_dir)
    logger.info("Tensorboard experiment directory: {}".format(tensorboard_dir))
    tensorboard_obj = SummaryWriter(log_dir=tensorboard_dir)

    assert len(gpus) == num_models
    for idx in range(num_models):
        model_config = copy.deepcopy(base_config)
        model_config["dataset"] = dataset
        model_config["seed"] = model_config["seed"] + idx
        model_config["nick"] = "seed{}".format(model_config["seed"])
        model_config["start_acc"] = -1
        logger.info("model_config is {}".format(model_config))
        logger.info("Model with idx {} runnning with seed {} on GPU {}".format(idx, model_config["seed"], +gpus[idx]))

        model_output_dir = "./cifar_models/{}/model_{}/".format(exp_name, idx)
        logger.info("This model with idx {} will be saved at {}".format(idx, model_output_dir))

        accuracy = cifar_train.main(model_config, model_output_dir, gpus[idx], tensorboard_obj=tensorboard_obj)
        logger.info("The accuracy of model with idx {} is {}".format(idx, accuracy * 100))

    logger.info("Done training all the models")
    os.remove(TMP_DATETIME_FILE)


if __name__ == "__main__":
    main()
