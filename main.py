import copy
import os
import sys
import time

import baseline
import numpy as np
import parameters
import routines
import utils
import wasserstein_ensemble
from align_layers import align_two_models
from basic_config import TMP_DATETIME_FILE
from log import logger
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

    model1_config = None
    args.aligned_model_index = 1

    if not args.ensemble_only:
        # align layers between models
        # if the number of layers of two models is different
        num_layer0 = utils.get_number_of_layers(models[0])
        num_layer1 = utils.get_number_of_layers(models[1])
        if num_layer0 != num_layer1:
            # TODO: Support bias for heterogeneous models
            assert args.disable_bias
            logger.info("------- Align two models -------")
            pair_models, pair_accuracies, args, model1_config = align_two_models(
                args, copy.deepcopy(models), accuracies, [num_layer0, num_layer1], epochs
            )
        else:
            pair_models = copy.deepcopy(models)
            pair_accuracies = accuracies

        setattr(
            args,
            f"params_aligned_model_{args.aligned_model_index}",
            utils.get_model_size(pair_models[args.aligned_model_index]),
        )
        # exit if align only
        if args.align_only:
            os.remove(TMP_DATETIME_FILE)
            sys.exit(0)

        # second_config is not needed here as well, since it's just used for the dataloader!
        st_time = time.perf_counter()
        activations = utils.get_model_activations(args, pair_models, config=config)
        end_time = time.perf_counter()
        setattr(args, "activation_time", end_time - st_time)
        logger.info(f"Activation time: {end_time - st_time}")

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
        logger.info(f"Geometric ensembling time : {end_time - st_time}")
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
        aligned_model_index = args.aligned_model_index
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
            args.retrain_model0_best = -1
            args.retrain_model1_best = -1
            setattr(args, f"retrain_aligned_model{aligned_model_index}_best", -1)
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
            args.retrain_model0_best = -1
            args.retrain_model1_best = -1
            setattr(args, f"retrain_aligned_model{aligned_model_index}_best", -1)
        elif args.retrain_align_only:
            initial_acc = [pair_accuracies[aligned_model_index]]
            nicks = [f"aligned_model{aligned_model_index}"]
            _, best_retrain_acc = routines.retrain_models(
                args,
                [pair_models[aligned_model_index]],
                retrain_loader,
                test_loader,
                config,
                tensorboard_obj=tensorboard_obj,
                initial_acc=initial_acc,
                nicks=nicks,
            )
            setattr(args, f"retrain_aligned_model{aligned_model_index}_best", best_retrain_acc[0])
            args.retrain_naive_best = -1
            args.retrain_geometric_best = -1
            args.retrain_model0_best = -1
            args.retrain_model1_best = -1
        elif args.retrain_origin_only:
            initial_acc = accuracies
            nicks = ["model_0", "model_1"]
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
            args.retrain_model0_best = best_retrain_acc[0]
            args.retrain_model1_best = best_retrain_acc[1]
            args.retrain_naive_best = -1
            args.retrain_geometric_best = -1
            setattr(args, f"retrain_aligned_model{aligned_model_index}_best", -1)
        else:
            # retrain all models
            if args.skip_retrain == 0:
                original_models = [models[1]]
                original_nicks = ["model_1"]
                original_accuracies = [accuracies[1]]
            elif args.skip_retrain == 1:
                original_models = [models[0]]
                original_nicks = ["model_0"]
                original_accuracies = [accuracies[0]]
            elif args.skip_retrain < 0:
                original_models = models
                original_nicks = ["model_0", "model_1"]
                original_accuracies = accuracies
            else:
                raise NotImplementedError

            if naive_acc < 0:
                # this happens in case the two models have different layer sizes
                nicks = original_nicks + ["geometric", f"aligned_model{aligned_model_index}"]
                initial_acc = original_accuracies + [geometric_acc, pair_accuracies[aligned_model_index]]
                _, best_retrain_acc = routines.retrain_models(
                    args,
                    [*original_models, geometric_model, pair_models[aligned_model_index]],
                    retrain_loader,
                    test_loader,
                    config,
                    tensorboard_obj=tensorboard_obj,
                    initial_acc=initial_acc,
                    nicks=nicks,
                )
                args.retrain_naive_best = -1
            else:
                nicks = original_nicks + ["geometric", "naive_averaging", f"aligned_model{aligned_model_index}"]
                initial_acc = [*original_accuracies, geometric_acc, naive_acc, pair_accuracies[aligned_model_index]]
                _, best_retrain_acc = routines.retrain_models(
                    args,
                    [*original_models, geometric_model, naive_model, pair_models[aligned_model_index]],
                    retrain_loader,
                    test_loader,
                    config,
                    tensorboard_obj=tensorboard_obj,
                    initial_acc=initial_acc,
                    nicks=nicks,
                )
                args.retrain_naive_best = best_retrain_acc[-2]

            if args.skip_retrain == 0:
                args.retrain_model0_best = -1
                args.retrain_model1_best = best_retrain_acc[0]
            elif args.skip_retrain == 1:
                args.retrain_model0_best = best_retrain_acc[0]
                args.retrain_model1_best = -1
            elif args.skip_retrain < 0:
                args.retrain_model0_best = best_retrain_acc[0]
                args.retrain_model1_best = best_retrain_acc[1]

            args.retrain_geometric_best = best_retrain_acc[len(original_models)]
            setattr(args, f"retrain_aligned_model{aligned_model_index}_best", best_retrain_acc[-1])

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
            results_dic[f"retrain_aligned_model{args.aligned_model_index}_best"] = getattr(
                args, f"retrain_aligned_model{aligned_model_index}_best"
            )
            results_dic["retrain_model0_best"] = args.retrain_model0_best
            results_dic["retrain_model1_best"] = args.retrain_model1_best
            results_dic["retrain_epochs"] = args.retrain

        utils.save_results_params_csv(os.path.join(args.csv_dir, args.save_result_file), results_dic, args)

        logger.info("----- Saved results at {} ------".format(args.save_result_file))
        logger.info(results_dic)

    logger.info("FYI: the parameters were: \n{}".format(args))
    os.remove(TMP_DATETIME_FILE)
