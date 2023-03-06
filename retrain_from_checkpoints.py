import os

import parameters
import routines
import torch
import utils
from basic_config import TMP_DATETIME_FILE
from log import logger
from tensorboardX import SummaryWriter


if __name__ == "__main__":
    args = parameters.get_parameters()
    logger.info("The parameters are: \n {}".format(args))

    if args.deterministic:
        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    load_results = routines.initial_loading(args)
    args = load_results["args"]
    config = load_results["config"]
    models = load_results["models"]
    accuracies = load_results["accuracies"]
    epochs = load_results["epochs"]
    train_loader = load_results["train_loader"]
    test_loader = load_results["test_loader"]
    retrain_loader = load_results["retrain_loader"]

    logger.info("-------- Retraining the models ---------")
    nicks = [f"model_{i}" for i in range(args.num_models)]
    initial_acc = accuracies
    if args.tensorboard:
        tensorboard_dir = os.path.join(args.tensorboard_root, args.exp_name)
        utils.mkdir(tensorboard_dir)
        logger.info("Tensorboard experiment directory: {}".format(tensorboard_dir))
        tensorboard_obj = SummaryWriter(log_dir=tensorboard_dir)
    else:
        tensorboard_obj = None
    _, best_retrain_acc = routines.retrain_models(
        args,
        models,
        retrain_loader,
        test_loader,
        config,
        tensorboard_obj=tensorboard_obj,
        initial_acc=initial_acc,
        nicks=nicks,
    )
    for idx, nick in enumerate(nicks):
        setattr(args, f"retrain_{nick}", best_retrain_acc[idx])

    if args.save_result_file != "":
        results_dic = {}
        results_dic["exp_name"] = args.exp_name

        for idx, acc in enumerate(accuracies):
            results_dic["model{}_acc".format(idx)] = acc

        for nick in nicks:
            results_dic[f"retrain_{nick}"] = getattr(args, f"retrain_{nick}")

        utils.save_results_params_csv(os.path.join(args.csv_dir, args.save_result_file), results_dic, args)

        logger.info("----- Saved results at {} ------".format(args.save_result_file))
        logger.info(results_dic)

    logger.info("FYI: the parameters were: \n{}".format(args))
    os.remove(TMP_DATETIME_FILE)
