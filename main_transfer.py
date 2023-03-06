import os
import random

import baseline
import parameters
import routines
import torch
import utils
from align_layers import align_to_transfer_map, get_alignment_map
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

    # run model transfer
    logger.info("------- Model transfer -------")
    num_layer0 = len(list(models[0].parameters()))
    num_layer1 = len(list(models[1].parameters()))
    logger.info(f"Number of layers of model 0: {num_layer0}")
    logger.info(f"Number of layers of model 1: {num_layer1}")
    num_layers = [num_layer0, num_layer1]
    model_names = [args.model_name, args.second_model_name]

    # get model type
    if args.model_name in ["mlpnet", "cifarmlpnet"]:
        model_type = "mlp"
    elif "vgg" in args.model_name:
        model_type = "vgg"
    elif "resnet" in args.model_name:
        model_type = "resnet"
    # get transfer mapping
    if args.mapping_type == "chain":
        # mapping = None
        mapping = list(range(num_layer0))
    elif args.mapping_type == "random":
        if model_type == "resnet":
            idx_list = [x for x in range(num_layer0 - 1) if x not in [9, 18, 31]]  # last layer is a FC layer
        else:
            idx_list = list(range(num_layer0))
        mapping = sorted(random.sample(idx_list, num_layer1 - 4))  # 3 shortcuts + 1 FC layer
    elif args.mapping_type == "cla":
        mapping, _, _, _ = get_alignment_map(args, models, num_layers, model_names)
        mapping = (mapping - 1).astype(int)  # index starts from 0 for transfer map
    if model_type != "mlp":
        mapping = align_to_transfer_map(mapping, model_type)
    if args.transfer_method == "transfer_only":
        transfer_acc, transfer_model = baseline.transfer_networks(
            args,
            models,
            test_loader,
            model_type=model_type,
            keep_weights=args.keep_pretrained_weights,
            mapping=mapping,
        )
    elif args.transfer_method == "transfer_avg":
        transfer_acc, transfer_model = baseline.transfer_networks_and_naive_ensembling(
            args, models, test_loader, model_type=model_type, mapping=mapping
        )
    elif args.transfer_method == "transfer_ot":
        transfer_acc, transfer_model = baseline.transfer_networks_and_otfusion(
            args, models, train_loader, test_loader, model_type=model_type, mapping=mapping
        )
    elif args.transfer_method == "transfer_add":
        transfer_acc, transfer_model = baseline.transfer_networks_and_add_layers(
            args, models, train_loader, test_loader, num_layers, model_names, model_type=model_type, mapping=mapping
        )
    n_models = len(transfer_model)
    # get model size
    for idx, model in enumerate(transfer_model):
        setattr(args, f"params_transfer_model{idx}", utils.get_model_size(model))

    # TODO: Suppport retrain multiple original models
    if args.retrain > 0:
        logger.info("-------- Retraining the models ---------")
        if args.tensorboard:
            tensorboard_dir = os.path.join(args.tensorboard_root, args.exp_name)
            utils.mkdir(tensorboard_dir)
            logger.info("Tensorboard experiment directory: {}".format(tensorboard_dir))
            tensorboard_obj = SummaryWriter(log_dir=tensorboard_dir)
        else:
            tensorboard_obj = None

        nicks = [f"transfer_model{i}" for i in range(n_models)]
        initial_acc = [transfer_acc[i] for i in range(n_models)]
        _, best_retrain_acc = routines.retrain_models(
            args,
            transfer_model,
            retrain_loader,
            test_loader,
            config,
            tensorboard_obj=tensorboard_obj,
            initial_acc=initial_acc,
            nicks=nicks,
        )

        for idx in range(n_models):
            setattr(args, f"retrain_transfer_model{idx}_best", best_retrain_acc[idx])

    if args.save_result_file != "":
        results_dic = {}
        results_dic["exp_name"] = args.exp_name

        for idx, acc in enumerate(accuracies):
            results_dic["model{}_acc".format(idx)] = acc

        results_dic["transfer_acc"] = transfer_acc
        results_dic["transfer_mapping"] = mapping

        # Save retrain statistics!
        if args.retrain > 0:
            for idx in range(n_models):
                results_dic[f"retrain_transfer_model{idx}_best"] = getattr(args, f"retrain_transfer_model{idx}_best")

        utils.save_results_params_csv(os.path.join(args.csv_dir, args.save_result_file), results_dic, args)

        logger.info("----- Saved results at {} ------".format(args.save_result_file))
        logger.info(results_dic)

    logger.info("FYI: the parameters were: \n{}".format(args))
    os.remove(TMP_DATETIME_FILE)
