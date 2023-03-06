import copy
import os
import sys
import time

import baseline
import numpy as np
import parameters
import routines
import torch
import utils
import wasserstein_ensemble
from align_layers import align_two_models
from basic_config import TMP_DATETIME_FILE
from log import logger
from model import get_model_from_name
from tensorboardX import SummaryWriter


if __name__ == "__main__":
    args = parameters.get_parameters()
    logger.info("The parameters are: \n {}".format(args))

    load_results = routines.initial_loading(args)
    args = load_results["args"]
    config = load_results["config"]
    models = load_results["models"]
    accuracies = load_results["accuracies"]
    epochs = load_results["epochs"]
    train_loader = load_results["train_loader"]
    test_loader = load_results["test_loader"]
    retrain_loader = load_results["retrain_loader"]

    # exit if train only
    if args.train_only or args.num_models == 1:
        os.remove(TMP_DATETIME_FILE)
        sys.exit(0)

    if not args.ensemble_only:
        fused_model = copy.deepcopy(models[0])
        fused_acc = accuracies[0]
        fused_epoch = epochs[0]
        fused_model_name = args.model_name_list[0]
        setattr(args, "fused_model_name", fused_model_name)

        if args.multiple_fusion_approach == "iterative":
            # iteratively fuse i-th model to the first model
            for iter in range(1, args.num_models):
                model1_config = None
                args.aligned_model_index = 1
                pair_models = [fused_model, copy.deepcopy(models[iter])]
                pair_accuracies = [fused_acc, accuracies[iter]]
                pair_epochs = [fused_epoch, epochs[iter]]
                pair_model_names = [fused_model_name, args.model_name_list[iter]]
                # align layers between models
                # if the number of layers of two models is different
                num_layer0 = len(list(pair_models[0].parameters()))
                num_layer1 = len(list(pair_models[1].parameters()))
                if num_layer0 != num_layer1:
                    # TODO: Support bias for heterogeneous models
                    assert args.disable_bias
                    logger.info(f"------- Align two models at iteration {iter} -------")
                    pair_models, pair_accuracies, args, model1_config = align_two_models(
                        args, pair_models, pair_accuracies, [num_layer0, num_layer1], pair_epochs, pair_model_names
                    )

                # reverse model order
                # because we want to align to the fused model
                pair_models = pair_models[::-1]
                # second_config is not needed here as well, since it's just used for the dataloader!
                activations = utils.get_model_activations(args, pair_models, config=config)

                # set seed for numpy based calculations
                NUMPY_SEED = 100
                np.random.seed(NUMPY_SEED)

                # run geometric aka wasserstein ensembling
                logger.info("------- Geometric Ensembling -------")
                st_time = time.perf_counter()
                if args.fusion_method == "ot":
                    geometric_acc, geometric_model = wasserstein_ensemble.geometric_ensembling_modularized(
                        args, pair_models, train_loader, test_loader, activations, model1_config
                    )
                elif args.fusion_method == "naive":
                    geometric_acc, geometric_model = baseline.naive_ensembling(args, pair_models, test_loader)
                end_time = time.perf_counter()
                setattr(args, "geometric_time", end_time - st_time)
                logger.info(f"Geometric ensembling time at iteration {iter}: {end_time - st_time}")
                fused_model = geometric_model
                fused_acc = geometric_acc
        elif args.multiple_fusion_approach == "many-to-one":
            # align every model to the first model
            # then all average model weights in layer-wise manner
            activations = [None, None]
            activations[1] = utils.get_model_activations(args, [fused_model], config=config)[0]
            params_list = [list(fused_model.parameters())]
            # iteratively fuse 2 models
            for iter in range(1, args.num_models):
                pair_models = [fused_model, copy.deepcopy(models[iter])]
                pair_accuracies = [fused_acc, accuracies[iter]]
                pair_epochs = [fused_epoch, epochs[iter]]
                pair_model_names = [fused_model_name, args.model_name_list[iter]]
                # align layers between models
                # if the number of layers of two models is different
                num_layer0 = len(list(pair_models[0].parameters()))
                num_layer1 = len(list(pair_models[1].parameters()))
                if num_layer0 != num_layer1:
                    # TODO: Support bias for heterogeneous models
                    assert args.disable_bias
                    logger.info(f"------- Align two models at iteration {iter} -------")
                    pair_models, pair_accuracies, args, _ = align_two_models(
                        args, pair_models, pair_accuracies, [num_layer0, num_layer1], pair_epochs, pair_model_names
                    )

                # reverse model order
                # because we want to align to the first model
                pair_models = pair_models[::-1]
                # second_config is not needed here as well, since it's just used for the dataloader!
                activations[0] = utils.get_model_activations(args, pair_models[0:1], config=config)[0]
                # set seed for numpy based calculations
                NUMPY_SEED = 100
                np.random.seed(NUMPY_SEED)

                # run geometric aka wasserstein ensembling
                logger.info("------- Geometric Ensembling -------")
                st_time = time.perf_counter()
                if args.fusion_method == "ot":
                    if args.geom_ensemble_type == "wts":
                        geometric_weights = wasserstein_ensemble.get_wassersteinized_layers_modularized(
                            args, pair_models, activations, test_loader=test_loader, not_avg=True
                        )
                    elif args.geom_ensemble_type == "acts":
                        geometric_weights = wasserstein_ensemble.get_acts_wassersteinized_layers_modularized(
                            args,
                            pair_models,
                            activations,
                            train_loader=train_loader,
                            test_loader=test_loader,
                            not_avg=True,
                        )
                elif args.fusion_method == "naive":
                    geometric_weights = list(pair_models[0].parameters())
                end_time = time.perf_counter()
                setattr(args, "geometric_time", end_time - st_time)
                logger.info(f"Geometric ensembling time at iteration {iter}: {end_time - st_time}")
                params_list.append(geometric_weights)

            # simply average the weights in networks
            weights = [1 / args.num_models for i in range(args.num_models)]
            avg_pars = []

            for par_group in zip(*params_list):
                avg_par = torch.mean(torch.stack(par_group), dim=0)
                avg_pars.append(avg_par)

            geometric_model = get_model_from_name(args, idx=0)
            # put on GPU
            if args.gpu_id != -1:
                geometric_model = geometric_model.cuda(args.gpu_id)
            # set the weights of the ensembled network
            for idx, (name, _) in enumerate(geometric_model.state_dict().items()):
                geometric_model.state_dict()[name].copy_(avg_pars[idx].data)

            # check the test performance
            log_dict = {}
            log_dict["test_losses"] = []
            geometric_acc = routines.test(args, geometric_model, test_loader, log_dict)

            if args.dump_final_models:
                routines.save_final_model(args, f"{args.fusion_method}_merge", geometric_model, geometric_acc)

        args.params_geometric = utils.get_model_size(geometric_model)
    else:
        geometric_model = None
        geometric_acc = -1
        setattr(args, "geometric_time", -1)

    # run prediction ensemble
    if args.skip_ensemble:
        prediction_acc = -1
    else:
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

        if args.retrain_geometric_only:
            initial_acc = [geometric_acc]
            nicks = ["geometric"]
            _, best_retrain_acc = routines.retrain_models(
                args,
                [geometric_model],
                retrain_loader,
                test_loader,
                config,
                tensorboard_obj=tensorboard_obj,
                initial_acc=initial_acc,
                nicks=nicks,
            )
            args.retrain_geometric_best = best_retrain_acc[0]
            args.retrain_naive_best = -1
            for idx in range(args.num_models):
                setattr(args, f"retrain_model{idx}_best", -1)
        elif args.retrain_avg_only:
            initial_acc = [naive_acc]
            nicks = ["naive_averaging"]
            _, best_retrain_acc = routines.retrain_models(
                args,
                [naive_model],
                retrain_loader,
                test_loader,
                config,
                tensorboard_obj=tensorboard_obj,
                initial_acc=initial_acc,
                nicks=nicks,
            )
            args.retrain_naive_best = best_retrain_acc[0]
            args.retrain_geometric_best = -1
            for idx in range(args.num_models):
                setattr(args, f"retrain_model{idx}_best", -1)
        elif args.retrain_origin_only:
            initial_acc = accuracies
            nicks = [f"model_{idx}" for idx in range(args.num_models)]
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
            for idx in range(args.num_models):
                setattr(args, f"retrain_model{idx}_best", best_retrain_acc[idx])
            args.retrain_naive_best = -1
            args.retrain_geometric_best = -1
        else:
            # retrain all models
            original_models = models
            original_nicks = [f"model_{idx}" for idx in range(args.num_models)]
            original_accuracies = accuracies

            if naive_acc < 0:
                # this happens in case the two models have different layer sizes
                nicks = original_nicks + ["geometric"]
                initial_acc = original_accuracies + [geometric_acc]
                _, best_retrain_acc = routines.retrain_models(
                    args,
                    [*original_models, geometric_model],
                    retrain_loader,
                    test_loader,
                    config,
                    tensorboard_obj=tensorboard_obj,
                    initial_acc=initial_acc,
                    nicks=nicks,
                )
                args.retrain_naive_best = -1
            else:
                nicks = original_nicks + ["geometric", "naive_averaging"]
                initial_acc = [*original_accuracies, geometric_acc, naive_acc]
                _, best_retrain_acc = routines.retrain_models(
                    args,
                    [*original_models, geometric_model, naive_model],
                    retrain_loader,
                    test_loader,
                    config,
                    tensorboard_obj=tensorboard_obj,
                    initial_acc=initial_acc,
                    nicks=nicks,
                )
                args.retrain_naive_best = best_retrain_acc[-1]

            for idx in range(args.num_models):
                setattr(args, f"retrain_model{idx}_best", best_retrain_acc[idx])

            args.retrain_geometric_best = best_retrain_acc[len(original_models)]

    if args.save_result_file != "":
        results_dic = {}
        results_dic["exp_name"] = args.exp_name

        for idx, acc in enumerate(accuracies):
            results_dic["model{}_acc".format(idx)] = acc

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

        results_dic["geometric_time"] = args.geometric_time
        # Save retrain statistics!
        if args.retrain > 0:
            results_dic["retrain_geometric_best"] = args.retrain_geometric_best
            results_dic["retrain_naive_best"] = args.retrain_naive_best
            for idx in range(args.num_models):
                results_dic[f"retrain_model{idx}_best"] = getattr(args, f"retrain_model{idx}_best")
            results_dic["retrain_model1_best"] = args.retrain_model1_best
            results_dic["retrain_epochs"] = args.retrain

        utils.save_results_params_csv(os.path.join(args.csv_dir, args.save_result_file), results_dic, args)

        logger.info("----- Saved results at {} ------".format(args.save_result_file))
        logger.info(results_dic)

    logger.info("FYI: the parameters were: \n{}".format(args))
    os.remove(TMP_DATETIME_FILE)
