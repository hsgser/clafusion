import copy
import os
import sys
import time
from basic_config import PATH_TO_CIFAR, TMP_DATETIME_FILE
sys.path.append(PATH_TO_CIFAR)

import baseline
import numpy as np
import parameters
import partition
import routines
import torch
import train as cifar_train
import utils
import wasserstein_ensemble
from align_layers import align_two_data_separated_models
from data import get_dataloader
from log import logger
from tensorboardX import SummaryWriter


if __name__ == "__main__":
    args = parameters.get_parameters()
    # set random seed
    logger.info(f"Partition training seed is {args.partition_train_seed}")
    torch.manual_seed(args.partition_train_seed)
    np.random.seed(args.partition_train_seed)

    if args.width_ratio != 1:
        if not args.proper_marginals:
            logger.info("setting proper marginals to True (needed for width_ratio!=1 mode)")
            args.proper_marginals = True
        if args.eval_aligned:
            logger.info("setting eval aligned to False (needed for width_ratio!=1 mode)")
            args.eval_aligned = False

    logger.info("The parameters are: \n {}".format(args))

    # loading configuration
    config, second_config = utils._get_config(args)
    args.config = config
    args.second_config = second_config

    # obtain trained models
    personal_testset = None
    other_testset = None

    if args.load_models != "":
        logger.info("------- Loading pre-trained models -------")

        # currently mnist is not supported!
        # assert args.dataset != 'mnist'

        # ensemble_experiment = "exp_2019-04-23_18-08-48/"
        # ensemble_experiment = "exp_2019-04-24_02-20-26"

        ensemble_experiment = args.load_models.split("/")
        if len(ensemble_experiment) > 1:
            # both the path and name of the experiment have been specified
            ensemble_dir = args.load_models
        elif len(ensemble_experiment) == 1:
            # otherwise append the directory before!
            ensemble_root_dir = "{}/{}_models/".format(args.baseroot, (args.dataset).lower())
            ensemble_dir = ensemble_root_dir + args.load_models

        utils.mkdir(ensemble_dir)
        # checkpoint_type = 'final'  # which checkpoint to use for ensembling (either of 'best' or 'final)

        if args.dataset == "mnist":
            train_loader, test_loader = get_dataloader(args)
        else:
            args.cifar_init_lr = config["optimizer_learning_rate"]
            if args.second_model_name is not None:
                assert second_config is not None
                assert args.cifar_init_lr == second_config["optimizer_learning_rate"]
                # also the below things should be fine as it is just dataloader loading!
            logger.info("loading {} dataloaders".format(args.dataset.lower()))
            train_loader, test_loader = cifar_train.get_dataset(
                config, to_download=args.to_download, data_root="./data"
            )

        models = []
        accuracies = []
        local_accuracies = []
        choices = []
        epochs = []
        for idx in range(args.num_models):
            logger.info("loading model with idx {} and checkpoint_type is {}".format(idx, args.ckpt_type))

            if args.dataset.lower() != "mnist" and (
                args.model_name.lower()[0:3] == "vgg" or args.model_name.lower()[0:6] == "resnet"
            ):
                if idx == 0:
                    config_used = config
                elif idx == 1:
                    config_used = second_config

                model, accuracy, epoch, local_accuracy, choice = cifar_train.get_pretrained_model(
                    config_used,
                    os.path.join(ensemble_dir, "model_{}/{}.checkpoint".format(idx, args.ckpt_type)),
                    args.gpu_id,
                    data_separated=True,
                    relu_inplace=not args.prelu_acts,  # if you want pre-relu acts, set relu_inplace to False
                )
            else:
                model, accuracy, local_accuracy, choice, epoch = routines.get_pretrained_model(
                    args,
                    os.path.join(ensemble_dir, "model_{}/{}.checkpoint".format(idx, args.ckpt_type)),
                    data_separated=True,
                    idx=idx,
                )
            models.append(model)
            accuracies.append(accuracy)
            local_accuracies.append(local_accuracy)
            choices.append(choice)
            epochs.append(epoch)
        logger.info("Done loading all the models")

        # Additional flag of recheck_acc to supplement the legacy flag recheck_cifar
        if args.recheck_cifar or args.recheck_acc:
            recheck_accuracies = []
            for model in models:
                log_dict = {}
                log_dict["test_losses"] = []
                recheck_accuracies.append(routines.test(args, model, test_loader, log_dict))
            logger.info("Rechecked accuracies are {}".format(recheck_accuracies))

    else:
        epochs = [args.n_epochs, args.n_epochs]
        # get dataloaders
        logger.info("------- Obtain dataloaders -------")
        if args.dataset == "mnist":
            train_loader, test_loader = get_dataloader(args)
        else:
            args.cifar_init_lr = config["optimizer_learning_rate"]
            if args.second_model_name is not None:
                assert second_config is not None
                assert args.cifar_init_lr == second_config["optimizer_learning_rate"]
                # also the below things should be fine as it is just dataloader loading!
            logger.info("loading {} dataloaders".format(args.dataset.lower()))
            train_loader, test_loader = cifar_train.get_dataset(
                config, to_download=args.to_download, data_root="./data"
            )

        if args.partition_type == "labels":
            logger.info("------- Split dataloaders by labels -------")
            choice = [int(x) for x in args.choice.split()]
            (trailo_a, teslo_a), (trailo_b, teslo_b), other = partition.split_mnist_by_labels(
                args, train_loader, test_loader, choice=choice
            )
            choices = [choice, list(other)]
            logger.info("------- Training independent models -------")
            models, accuracies, local_accuracies = routines.train_data_separated_models(
                args, [trailo_a, trailo_b], [teslo_a, teslo_b], test_loader, choices
            )
        elif args.partition_type == "personalized":
            assert args.dataset == "mnist"
            logger.info("------- Split dataloaders wrt personalized data setting-------")
            trailo_a, trailo_b, personal_trainset, other_trainset = partition.get_personalized_split(
                args,
                personal_label=args.personal_class_idx,
                split_frac=args.personal_split_frac,
                is_train=True,
                return_dataset=True,
            )
            teslo_a, teslo_b, personal_testset, other_testset = partition.get_personalized_split(
                args,
                personal_label=args.personal_class_idx,
                split_frac=args.personal_split_frac,
                is_train=False,
                return_dataset=True,
            )
            if args.dump_datasets:
                logger.info("Save personalized datasets")
                utils.save_datasets(args, personal_trainset, personal_testset, other_trainset, other_testset)
            logger.info("------- Training independent models -------")

            other = list(range(0, 10))
            other.remove(args.personal_class_idx)
            choices = [list(range(0, 10)), other]
            models, accuracies, local_accuracies = routines.train_data_separated_models(
                args, [trailo_a, trailo_b], [teslo_a, teslo_b], test_loader, choices
            )
        elif args.partition_type == "small_big":
            # assert args.dataset == 'mnist'
            logger.info("------- Split dataloaders wrt small big data setting-------")
            trailo_a, trailo_b, personal_trainset, other_trainset = partition.get_small_big_split(
                args, split_frac=args.personal_split_frac, is_train=True, return_dataset=True
            )
            teslo_a, teslo_b, personal_testset, other_testset = partition.get_small_big_split(
                args, split_frac=args.personal_split_frac, is_train=False, return_dataset=True
            )
            if args.dump_datasets:
                logger.info("Save personalized datasets")
                utils.save_datasets(args, personal_trainset, personal_testset, other_trainset, other_testset)
            logger.info("------- Training independent models -------")

            choice = list(range(0, 10))
            choices = [choice, choice]
            model_configs = [args.config, args.second_config]
            models, accuracies, local_accuracies = routines.train_data_separated_models(
                args, [trailo_a, trailo_b], [teslo_a, teslo_b], test_loader, choices, model_configs=model_configs
            )

    # get model size
    for idx, model in enumerate(models):
        setattr(args, f"params_model_{idx}", utils.get_model_size(model))

    # exit if train only
    if args.train_only:
        os.remove(TMP_DATETIME_FILE)
        sys.exit(0)

    # load personalized dataset
    personal_dataset = None
    personal_testset = None
    other_testset = None

    if args.load_personalized_datasets != "":
        # personal_dataset = torch.load(args.load_personalized_datasets)
        personal_dataset, _, personal_testset, other_testset = partition.get_personalized_dataset(args)
    else:
        if args.partition_type == "personalized" or args.partition_type == "small_big":
            if args.partition_dataloader == 0:
                personal_dataset = personal_trainset
            elif args.partition_dataloader == 1:
                personal_dataset = other_trainset

    # currently only support two models
    # TODO: support multiple models
    assert args.num_models == 2
    model1_config = None
    args.aligned_model_index = 1
    pair_models = copy.deepcopy(models)
    pair_accuracies = copy.deepcopy(accuracies)

    # align layers between models
    # if the number of layers of two models is different
    num_layer0 = len(list(models[0].parameters()))
    num_layer1 = len(list(models[1].parameters()))
    if num_layer0 != num_layer1:
        # TODO: Support bias for heterogeneous models
        assert args.disable_bias
        if personal_testset and other_testset:
            personal_test_loader = partition.to_dataloader_from_tens(*personal_testset, args.batch_size_test)
            other_test_loader = partition.to_dataloader_from_tens(*other_testset, args.batch_size_test)
            local_test_loaders = [personal_test_loader, other_test_loader]
        else:
            local_test_loaders = [None, None]
        logger.info("------- Align two models -------")
        pair_models, pair_accuracies, args, model1_config = align_two_data_separated_models(
            args,
            models,
            accuracies,
            local_accuracies,
            [num_layer0, num_layer1],
            personal_dataset,
            local_test_loaders,
            choices,
            epochs,
        )

    setattr(
        args,
        f"params_aligned_model_{args.aligned_model_index}",
        utils.get_model_size(pair_models[args.aligned_model_index]),
    )
    # exit if align only
    if args.align_only:
        os.remove(TMP_DATETIME_FILE)
        sys.exit(0)

    st_time = time.perf_counter()
    activations = utils.get_model_activations(args, pair_models, config=config, personal_dataset=personal_dataset)
    end_time = time.perf_counter()
    setattr(args, "activation_time", end_time - st_time)
    logger.info(f"Activation time: {end_time - st_time}")

    # run geometric aka wasserstein ensembling
    logger.info("------- Geometric Ensembling -------")
    st_time = time.perf_counter()
    geometric_acc, geometric_model = wasserstein_ensemble.geometric_ensembling_modularized(
        args, pair_models, train_loader, test_loader, activations, model1_config
    )
    end_time = time.perf_counter()
    setattr(args, "geometric_time", end_time - st_time)
    logger.info(f"Geometric ensembling time: {end_time - st_time}")
    args.params_geometric = utils.get_model_size(geometric_model)

    # run baselines
    logger.info("------- Prediction based ensembling -------")
    prediction_acc = baseline.prediction_ensembling(args, models, test_loader)

    if args.run_naive_ensemble:
        logger.info("------- Naive ensembling of weights -------")
        naive_acc, naive_model = baseline.naive_ensembling(args, models, test_loader)
        if args.dump_final_models:
            routines.save_final_model(args, "naive_merge", naive_model, naive_acc)
    else:
        # ignore naive ensembling
        naive_acc = -1

    if args.retrain > 0:
        aligned_model_index = args.aligned_model_index
        logger.info("-------- Retraining the models ---------")
        if args.tensorboard:
            tensorboard_dir = os.path.join(args.tensorboard_root, args.exp_name)
            utils.mkdir(tensorboard_dir)
            logger.info("Tensorboard experiment directory: {}".format(tensorboard_dir))
            tensorboard_obj = SummaryWriter(log_dir=tensorboard_dir)
        else:
            tensorboard_obj = None

        if args.retrain_avg_only:
            initial_acc = [geometric_acc, naive_acc]
            _, best_retrain_acc = routines.retrain_models(
                args,
                [geometric_model, naive_model],
                train_loader,
                test_loader,
                config,
                tensorboard_obj=tensorboard_obj,
                initial_acc=initial_acc,
            )
            args.retrain_geometric_best = best_retrain_acc[0]
            args.retrain_naive_best = best_retrain_acc[1]
        elif args.retrain_align_only:
            initial_acc = [pair_accuracies[aligned_model_index]]
            _, best_retrain_acc = routines.retrain_models(
                args,
                [pair_models[aligned_model_index]],
                train_loader,
                test_loader,
                config,
                tensorboard_obj=tensorboard_obj,
                initial_acc=initial_acc,
            )
            args.retrain_geometric_best = -1
            args.retrain_naive_best = -1
            setattr(args, f"retrain_aligned_model{aligned_model_index}_best", best_retrain_acc[0])
        else:
            if naive_acc > 0:
                initial_acc = [*accuracies, geometric_acc, naive_acc, pair_accuracies[aligned_model_index]]
                _, best_retrain_acc = routines.retrain_models(
                    args,
                    [*models, geometric_model, naive_model, pair_models[aligned_model_index]],
                    train_loader,
                    test_loader,
                    config,
                    tensorboard_obj=tensorboard_obj,
                    initial_acc=initial_acc,
                )
                args.retrain_model0_best = best_retrain_acc[0]
                args.retrain_model1_best = best_retrain_acc[1]
                args.retrain_geometric_best = best_retrain_acc[2]
                args.retrain_naive_best = best_retrain_acc[3]
                setattr(args, f"retrain_aligned_model{aligned_model_index}_best", best_retrain_acc[-1])
            else:
                initial_acc = [*accuracies, geometric_acc, pair_accuracies[aligned_model_index]]
                _, best_retrain_acc = routines.retrain_models(
                    args,
                    [*models, geometric_model, pair_models[aligned_model_index]],
                    train_loader,
                    test_loader,
                    config,
                    tensorboard_obj=tensorboard_obj,
                    initial_acc=initial_acc,
                )
                args.retrain_model0_best = best_retrain_acc[0]
                args.retrain_model1_best = best_retrain_acc[1]
                args.retrain_geometric_best = best_retrain_acc[2]
                args.retrain_naive_best = best_retrain_acc[3]
                setattr(args, f"retrain_aligned_model{aligned_model_index}_best", best_retrain_acc[-1])

    if args.save_result_file != "":
        results_dic = {}
        results_dic["exp_name"] = args.exp_name

        for idx, acc in enumerate(accuracies):
            results_dic["model{}_acc".format(idx)] = acc

        for idx, local_acc in enumerate(local_accuracies):
            results_dic["model{}_local_acc".format(idx)] = local_acc

        results_dic["geometric_acc"] = geometric_acc
        results_dic["prediction_acc"] = prediction_acc
        results_dic["naive_acc"] = naive_acc

        # Additional statistics
        results_dic["geometric_gain"] = geometric_acc - max(accuracies)
        results_dic["geometric_gain_%"] = ((geometric_acc - max(accuracies)) * 100.0) / max(accuracies)
        results_dic["prediction_gain"] = prediction_acc - max(accuracies)
        results_dic["prediction_gain_%"] = ((prediction_acc - max(accuracies)) * 100.0) / max(accuracies)
        results_dic["relative_loss_wrt_prediction"] = (
            results_dic["prediction_gain_%"] - results_dic["geometric_gain_%"]
        )

        # if args.eval_aligned:
        #     results_dic['model0_aligned'] = args.model0_aligned_acc

        # Save retrain statistics!
        if args.retrain > 0:
            results_dic["retrain_geometric_best"] = args.retrain_geometric_best
            results_dic["retrain_naive_best"] = args.retrain_naive_best
            if not args.retrain_avg_only:
                results_dic[f"retrain_aligned_model{args.aligned_model_index}_best"] = getattr(
                    args, f"retrain_aligned_model{aligned_model_index}_best"
                )
                if not args.retrain_align_only:
                    results_dic["retrain_model0_best"] = args.retrain_model0_best
                    results_dic["retrain_model1_best"] = args.retrain_model1_best
            results_dic["retrain_epochs"] = args.retrain

        utils.save_results_params_csv(os.path.join(args.csv_dir, args.save_result_file), results_dic, args)

        logger.info("----- Saved results at {} ------".format(args.save_result_file))
        logger.info(results_dic)

    logger.info("FYI: the parameters were: \n{}".format(args))
    os.remove(TMP_DATETIME_FILE)
