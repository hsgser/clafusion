import argparse
import copy
import json
import os

from log import get_first_timestamp, logger
from utils import dotdict, mkdir


# commenting this now
def get_deprecated_params_vgg_cifar():
    parameters = {
        "n_epochs": 1,
        "enable_dropout": False,
        "batch_size_train": 128,
        "batch_size_test": 1000,
        "learning_rate": 0.01,
        "momentum": 0.5,
        "log_interval": 100,
        "to_download": True,  # set to True if MNIST/dataset hasn't been downloaded,
        "disable_bias": True,  # no bias at all in fc or conv layers,
        "dataset": "Cifar10",
        "num_models": 2,
        "model_name": "vgg11_nobias",
        "num_hidden_nodes": 100,
        "num_hidden_nodes1": 400,
        "num_hidden_nodes2": 200,
        "num_hidden_nodes3": 100,
    }
    return dotdict(parameters)


def get_deprecated_params_mnist_act():
    parameters = {
        "n_epochs": 1,
        "enable_dropout": False,
        "batch_size_train": 64,
        "batch_size_test": 1000,
        "learning_rate": 0.01,
        "momentum": 0.5,
        "log_interval": 100,
        "to_download": True,  # set to True if MNIST/dataset hasn't been downloaded,
        "disable_bias": True,  # no bias at all in fc or conv layers,
        "dataset": "mnist",
        "num_models": 2,
        "model_name": "simplenet",
        "num_hidden_nodes": 400,
        "num_hidden_nodes1": 400,
        "num_hidden_nodes2": 200,
        "num_hidden_nodes3": 100,
        "gpu_id": 5,
        "skip_last_layer": False,
        "reg": 1e-2,
        "debug": False,
        "activation_histograms": True,
        "act_num_samples": 100,
        "softmax_temperature": 1,
    }
    return dotdict(parameters)


def dump_parameters(args):
    logger.info("dumping parameters at {}".format(args.config_dir))
    mkdir(args.config_dir)
    with open(os.path.join(args.config_dir, args.exp_name + ".json"), "w") as outfile:
        if not (type(args) is dict or type(args) is dotdict):
            json.dump(vars(args), outfile, sort_keys=True, indent=4)
        else:
            json.dump(args, outfile, sort_keys=True, indent=4)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-epochs", default=1, type=int, help="number of epochs")
    parser.add_argument("--batch-size-train", default=64, type=int, help="training batch size")
    parser.add_argument("--batch-size-test", default=1000, type=int, help="test batch size")
    parser.add_argument("--learning-rate", default=0.01, type=float, help="learning rate for SGD (default: 0.01)")
    parser.add_argument("--momentum", default=0.5, type=float, help="momentum for SGD (default: 0.5)")

    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="log progress every N batches (when progress bar is disabled)",
    )

    parser.add_argument("--to-download", action="store_true", help="download the dataset (typically mnist)")
    parser.add_argument("--disable-bias", action="store_false", help="disable bias in the neural network layers")
    parser.add_argument(
        "--dataset",
        default="mnist",
        type=str,
        choices=["mnist", "cifar10", "cifar100", "tinyimagenet", "esc50"],
        help="dataset to use for the task",
    )
    parser.add_argument(
        "--test-fold", default=1, type=int, choices=[1, 2, 3, 4, 5], help="data fold will be used for testing"
    )
    parser.add_argument("--num-models", default=2, type=int, help="number of models to ensemble")
    parser.add_argument(
        "--model-name",
        type=str,
        default="mlpnet",
        help="Type of neural network model (net|mlpnet|cifarmlpnet|vgg*|renset*)",
    )
    parser.add_argument("--config-file", type=str, default=None, help="config file path")
    parser.add_argument("--config-dir", type=str, default="./configurations", help="config dir")
    parser.add_argument(
        "--check-layer-sizes", action="store_true", help="check layer sizes when align neurons between two models"
    )
    parser.add_argument(
        "--net-config",
        type=lambda xs: [list(map(int, x.split())) for x in xs.split("; ")],
        default=[],
        help='configuration for training multiple MLPNETs, e.g. "400 200 100; 400 100"',
    )
    parser.add_argument(
        "--num-hidden-nodes", default=400, type=int, help="simplenet: number of hidden nodes in the only hidden layer"
    )
    parser.add_argument("--num-hidden-layers", default=3, type=int, help="mlpnet: number of hidden layers")
    parser.add_argument(
        "--num-hidden-nodes1", default=400, type=int, help="mlpnet: number of hidden nodes in the hidden layer 1"
    )
    parser.add_argument(
        "--num-hidden-nodes2", default=200, type=int, help="mlpnet: number of hidden nodes in the hidden layer 2"
    )
    parser.add_argument(
        "--num-hidden-nodes3", default=100, type=int, help="mlpnet: number of hidden nodes in the hidden layer 3"
    )
    parser.add_argument(
        "--num-hidden-nodes4", default=50, type=int, help="mlpnet: number of hidden nodes in the hidden layer 3"
    )

    parser.add_argument("--sweep-id", default=-1, type=int, help="sweep id ")

    parser.add_argument("--gpu-id", default=-1, type=int, help="GPU id to use")
    parser.add_argument(
        "--skip-last-layer", action="store_true", help="skip the last layer in calculating optimal transport"
    )
    parser.add_argument(
        "--skip-last-layer-type",
        type=str,
        default="average",
        choices=["second", "average"],
        help="how to average the parameters for the last layer",
    )

    parser.add_argument("--debug", action="store_true", help="print debug statements")
    parser.add_argument("--cifar-style-data", action="store_true", help="use data loader in cifar style")
    parser.add_argument("--activation-histograms", action="store_true", help="utilize activation histograms")
    parser.add_argument("--act-num-samples", default=100, type=int, help="num of samples to compute activation stats")
    parser.add_argument(
        "--softmax-temperature", default=1, type=float, help="softmax temperature for activation weights (default: 1)"
    )
    parser.add_argument(
        "--activation-mode",
        type=str,
        default=None,
        choices=["mean", "std", "meanstd", "raw"],
        help="mode that chooses how the importance of a neuron is calculated.",
    )

    parser.add_argument(
        "--options-type", type=str, default="generic", choices=["generic"], help="the type of options to load"
    )
    parser.add_argument(
        "--deprecated",
        type=str,
        default=None,
        choices=["vgg_cifar", "mnist_act"],
        help="loaded parameters in deprecated style. ",
    )

    parser.add_argument(
        "--save-result-file", type=str, default="default.csv", help="path of csv file to save things to"
    )
    parser.add_argument("--sweep-name", type=str, default=None, help="name of sweep experiment")

    parser.add_argument("--reg", default=1e-2, type=float, help="regularization strength for sinkhorn (default: 1e-2)")
    parser.add_argument(
        "--reg-m",
        default=1e-3,
        type=float,
        help="regularization strength for marginals in unbalanced sinkhorn (default: 1e-3)",
    )
    parser.add_argument(
        "--ground-metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "cosine"],
        help="ground metric for OT calculations, only works in free support v2 and soon with Ground Metric class in all! .",
    )
    parser.add_argument(
        "--ground-metric-normalize",
        type=str,
        default="log",
        choices=["log", "max", "none", "median", "mean"],
        help="ground metric normalization to consider! ",
    )
    parser.add_argument("--not-squared", action="store_true", help="dont square the ground metric")
    parser.add_argument("--clip-gm", action="store_true", help="to clip ground metric")
    parser.add_argument("--clip-min", action="store", type=float, default=0, help="Value for clip-min for gm")
    parser.add_argument("--clip-max", action="store", type=float, default=5, help="Value for clip-max for gm")
    parser.add_argument("--tmap-stats", action="store_true", help="print tmap stats")
    parser.add_argument(
        "--ensemble-step", type=float, default=0.5, action="store", help="rate of adjustment towards the second model"
    )

    parser.add_argument(
        "--ground-metric-eff", action="store_true", help="memory efficient calculation of ground metric"
    )

    parser.add_argument(
        "--retrain",
        type=int,
        default=0,
        action="store",
        help="number of epochs to retrain all the models & their avgs",
    )
    parser.add_argument(
        "--retrain-lr-decay",
        type=float,
        default=-1,
        action="store",
        help="amount by which to reduce the initial lr while retraining the model avgs",
    )
    parser.add_argument(
        "--retrain-lr-decay-factor",
        type=float,
        default=None,
        action="store",
        help="lr decay factor when the LR is gradually decreased by Step LR",
    )
    parser.add_argument(
        "--retrain-lr-decay-epochs",
        type=str,
        default=None,
        action="store",
        help="epochs at which retrain lr decay factor should be applied. underscore separated! ",
    )
    parser.add_argument("--retrain-avg-only", action="store_true", help="retraining the model avgs only")
    parser.add_argument(
        "--keep-pretrained-weights", action="store_true", help="keep pretrain weights after transferring"
    )
    parser.add_argument("--retrain-align-only", action="store_true", help="retraining the aligned model only")
    parser.add_argument("--retrain-geometric-only", action="store_true", help="retraining the model geometric only")
    parser.add_argument("--retrain-origin-only", action="store_true", help="retraining the original models only")

    parser.add_argument(
        "--load-models", type=str, default="", help="path/name of directory from where to load the models"
    )
    parser.add_argument(
        "--load-personalized-datasets", type=str, default="", help="path/name of directory from where to load datasets"
    )
    parser.add_argument(
        "--ckpt-type", type=str, default="best", choices=["best", "final"], help="which checkpoint to load"
    )

    parser.add_argument("--recheck-cifar", action="store_true", help="recheck cifar accuracies")
    parser.add_argument(
        "--recheck-acc", action="store_true", help="recheck model accuracies (recheck-cifar is legacy/deprecated)"
    )
    parser.add_argument("--eval-aligned", action="store_true", help="evaluate the accuracy of the aligned model 0")

    parser.add_argument("--enable-dropout", action="store_true", help="enable dropout in neural networks")
    parser.add_argument("--dump-model", action="store_true", help="dump model checkpoints")
    parser.add_argument("--dump-final-models", action="store_true", help="dump final trained model checkpoints")
    parser.add_argument("--correction", action="store_true", help="scaling correction for OT")
    parser.add_argument("--train-seed", type=int, default=0, action="store", help="seed for training models")
    parser.add_argument(
        "--activation-seed", type=int, default=42, action="store", help="seed for computing activations"
    )

    parser.add_argument("--weight-stats", action="store_true", help="log neuron-wise weight vector stats.")
    parser.add_argument(
        "--sinkhorn-type",
        type=str,
        default="normal",
        choices=["normal", "stabilized", "epsilon", "gpu"],
        help="Type of sinkhorn algorithm to consider.",
    )
    parser.add_argument(
        "--geom-ensemble-type",
        type=str,
        default="wts",
        choices=["wts", "acts"],
        help="Ensemble based on weights (wts) or activations (acts).",
    )
    parser.add_argument(
        "--act-bug", action="store_true", help="simulate the bug in ground metric calc for act based averaging"
    )
    parser.add_argument(
        "--standardize-acts",
        action="store_true",
        help="subtract mean and divide by standard deviation across the samples for use in act based alignment",
    )
    parser.add_argument(
        "--transform-acts",
        action="store_true",
        help="transform activations by transport map for later use in bi_avg mode ",
    )
    parser.add_argument(
        "--center-acts",
        action="store_true",
        help="subtract mean only across the samples for use in act based alignment",
    )
    parser.add_argument(
        "--prelu-acts", action="store_true", help="do activation based alignment based on pre-relu acts"
    )
    parser.add_argument("--pool-acts", action="store_true", help="do activation based alignment based on pooling acts")
    parser.add_argument("--pool-relu", action="store_true", help="do relu first before pooling acts")
    parser.add_argument("--normalize-acts", action="store_true", help="normalize the vector of activations")
    parser.add_argument("--normalize-wts", action="store_true", help="normalize the vector of weights")
    parser.add_argument("--gromov", action="store_true", help="use gromov wasserstein distance and barycenters")
    parser.add_argument(
        "--gromov-loss",
        type=str,
        default="square_loss",
        action="store",
        choices=["square_loss", "kl_loss"],
        help="choice of loss function for gromov wasserstein computations",
    )
    parser.add_argument(
        "--tensorboard-root",
        action="store",
        default="./tensorboard",
        type=str,
        help="Root directory of tensorboard logs",
    )
    parser.add_argument("--tensorboard", action="store_true", help="Use tensorboard to plot the loss values")

    parser.add_argument(
        "--same-model", action="store", type=int, default=-1, help="Index of the same model to average with itself"
    )
    parser.add_argument("--dist-normalize", action="store_true", help="normalize distances by act num samples")
    parser.add_argument("--update-acts", action="store_true", help="update acts during the alignment of model0")
    parser.add_argument(
        "--past-correction",
        action="store_true",
        help="use the current weights aligned by multiplying with past transport map",
    )
    parser.add_argument(
        "--partial-reshape", action="store_true", help="partially reshape the conv layers in ground metric calculation"
    )
    parser.add_argument(
        "--choice", type=str, default="0 2 4 6 8", action="store", help="choice of how to partition the labels"
    )
    parser.add_argument(
        "--diff-init", action="store_true", help="different initialization for models in data separated mode"
    )

    parser.add_argument(
        "--partition-type",
        type=str,
        default="labels",
        action="store",
        choices=["labels", "personalized", "small_big"],
        help="type of partitioning of training set to carry out",
    )
    parser.add_argument(
        "--personal-class-idx", type=int, default=9, action="store", help="class index for personal data"
    )
    parser.add_argument(
        "--partition-dataloader",
        type=int,
        default=-1,
        action="store",
        help="data loader to use in data partitioned setting",
    )
    parser.add_argument(
        "--partition-train-seed",
        type=int,
        default=0,
        action="store",
        help="random seed for training data-separated models",
    )
    parser.add_argument("--dump-datasets", action="store_true", help="dump personalized datasets")
    parser.add_argument(
        "--personal-split-frac",
        type=float,
        default=0.1,
        action="store",
        help="split fraction of rest of examples for personal data",
    )
    parser.add_argument("--exact", action="store_true", help="compute exact optimal transport")
    parser.add_argument("--skip-personal-idx", action="store_true", help="skip personal data")
    parser.add_argument(
        "--prediction-wts", action="store_true", help="use wts given by ensemble step for prediction ensembling"
    )
    parser.add_argument(
        "--width-ratio",
        type=float,
        default=1,
        action="store",
        help="ratio of the widths of the hidden layers between the two models",
    )
    parser.add_argument(
        "--proper-marginals", action="store_true", help="consider the marginals of transport map properly"
    )
    parser.add_argument(
        "--retrain-seed", type=int, default=-1, action="store", help="if reseed computations again in retrain"
    )
    parser.add_argument(
        "--no-random-trainloaders",
        action="store_true",
        help="get train loaders without any random transforms to ensure consistency",
    )
    parser.add_argument(
        "--reinit-trainloaders",
        action="store_true",
        help="reinit train loader when starting retraining of each model!",
    )
    parser.add_argument("--second-model-name", type=str, default=None, action="store", help="name of second model!")
    parser.add_argument("--print-distances", action="store_true", help="print OT distances for every layer")
    parser.add_argument("--deterministic", action="store_true", help="do retrain in deterministic mode!")
    parser.add_argument(
        "--skip-retrain", type=int, default=-1, action="store", help="which of the original models to skip retraining"
    )
    parser.add_argument(
        "--importance",
        type=str,
        default=None,
        action="store",
        help="importance measure to use for building probab mass! (options, l1, l2, l11, l12)",
    )
    parser.add_argument("--unbalanced", action="store_true", help="use unbalanced OT")
    parser.add_argument("--temperature", default=20, type=float, help="distillation temperature for (default: 20)")
    parser.add_argument("--alpha", default=0.7, type=float, help="weight towards distillation loss (default: 0.7)")
    parser.add_argument("--dist-epochs", default=60, type=int, help="number of distillation epochs")
    parser.add_argument("--not-dist-geometric", action="store_true", help="skip geometric model when distillation")
    parser.add_argument(
        "--handle-skips", action="store_true", help="handle shortcut skips in resnet which decrease dimension"
    )
    # heterogeneous model fusion
    parser.add_argument(
        "--run-naive-ensemble",
        action="store_true",
        help="run the naive ensembling of weights if two model architecture are identical",
    )
    parser.add_argument("--train-only", action="store_true", help="only train models without running model fusion")
    parser.add_argument("--ensemble-only", action="store_true", help="only run ensemble learning")
    parser.add_argument("--skip-ensemble", action="store_true", help="skip ensemble learning")
    parser.add_argument("--align-only", action="store_true", help="do not run layer-wise fusion")
    parser.add_argument(
        "--start-layer-idx", default=2, type=int, action="store", help="which hidden layers to start alignment"
    )
    parser.add_argument(
        "--free-end-layers", default=1, type=int, action="store", help="match last hidden layers of two models"
    )
    parser.add_argument(
        "--layer-measure",
        type=str,
        default="neuron",
        choices=["neuron", "activation", "index"],
        help="type of layer measure",
    )
    parser.add_argument(
        "--layer-metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "cka", "cca", "wd"],
        help="cost function between layer measures of two models",
    )
    parser.add_argument(
        "--balance-method",
        type=str,
        default="add",
        choices=["add", "merge"],
        help="method to balance the number of layers",
    )
    parser.add_argument(
        "--relu-approx-method",
        type=str,
        default="sum",
        choices=["sum", "majority", "avg"],
        help="method to approximate ReLU",
    )
    # heterogeneous model transfer
    parser.add_argument(
        "--transfer-method",
        type=str,
        default="transfer_only",
        choices=["transfer_only", "transfer_avg", "transfer_ot", "transfer_add"],
        help="model transfer method",
    )
    parser.add_argument(
        "--mapping-type", type=str, default="cla", choices=["chain", "random", "cla"], help="type of transfer mapping"
    )
    parser.add_argument(
        "--fusion-method", type=str, default="ot", choices=["ot", "naive"], help="layer-wise model fusion method"
    )
    # multiple models
    parser.add_argument(
        "--model-name-list",
        type=lambda xs: xs.split("; "),
        default=[],
        help='list of model names, e.g. "mlpnet; mlpnet"; "vgg11_nobias; vgg13_nobias"',
    )
    parser.add_argument(
        "--multiple-fusion-approach",
        type=str,
        default="iterative",
        choices=["iterative", "many-to-one"],
        help="the fusion approach for multiple neural networks",
    )
    parser.add_argument(
        "--longest-model-index",
        default=0,
        type=int,
        help="the position of the longest model to which other models are aligned to",
    )
    return parser


def get_parameters():
    logger.info("------- Setting up parameters -------")
    parser = get_parser()
    base_args = parser.parse_args()

    # handle configs for MLPNETs
    num_configs = len(base_args.net_config)
    if num_configs > 0:
        setattr(base_args, "parse_config", True)
        assert num_configs == base_args.num_models
    else:
        setattr(base_args, "parse_config", False)

    for i in range(num_configs):
        setattr(base_args, f"model{i}_config", base_args.net_config[i])

    # check model name list
    if len(base_args.model_name_list) > 0:
        assert base_args.num_models == len(base_args.model_name_list)
        base_args.fused_model_name = base_args.model_name_list[0]
    else:
        if base_args.second_model_name is None:
            base_args.second_model_name = base_args.model_name
        base_args.model_name_list = [base_args.model_name, base_args.second_model_name]
        base_args.fused_model_name = base_args.model_name

    if base_args.options_type != "generic":
        # This allows adding specific arguments that might be needed for a particular task
        raise NotImplementedError

    if base_args.deprecated is not None:
        # This enables passing parameters in dictionaries (dotdicts)
        if base_args.deprecated == "vgg_cifar":
            args = get_deprecated_params_vgg_cifar()
            args.deprecated = base_args.deprecated
        elif base_args.deprecated == "mnist_act":
            args = get_deprecated_params_mnist_act()
            args.deprecated = base_args.deprecated
        else:
            raise NotImplementedError
    else:
        # Here we utilize config files to setup the parameters
        if base_args.config_file:
            args = copy.deepcopy(base_args)
            logger.info("Reading parameters from {}, but CLI arguments".format(args.config_file))
            with open(os.path.join(base_args.config_dir, base_args.config_file), "r") as f:
                file_params = dotdict(json.load(f))
                for param, value in file_params.items():
                    if not hasattr(args, param):
                        # If it doesn't contain, then set from config
                        setattr(args, param, value)
                    elif getattr(args, param) == parser.get_default(param):
                        # Or when it has , but is the default, then override from config
                        setattr(args, param, value)

                        # When it has and is not default, keep it

            # Remove the set flag of deprecated if the config was in that mode
            args.deprecated = None
        else:
            # these remain unmodified from what was default or passed in via CLI
            args = base_args

    # Setup a timestamp for the experiment and save it in args
    if hasattr(args, "timestamp"):
        # the config file contained the timestamp parameter from the last experiment
        # (which say is being reproduced) so save it as well
        args.previous_timestamp = args.timestamp
    args.timestamp = get_first_timestamp()

    # Set rootdir and other dump directories for the experiment
    args.rootdir = os.getcwd()
    args.baseroot = args.rootdir
    all_model_idx = [str(idx) for idx in range(args.num_models)]
    if args.load_models != "":
        exp_folder_name = "merge_" + args.load_models.split("/")[1] + "_" + "_".join(all_model_idx)
    elif args.second_model_name is None:
        exp_folder_name = "merge_" + args.model_name + "_" + "_".join(all_model_idx)
    else:
        exp_folder_name = "merge_" + args.model_name + "_" + args.second_model_name
    args.rootdir = os.path.join(args.rootdir, exp_folder_name)
    args.config_dir = os.path.join(args.rootdir, "configurations")
    args.result_dir = os.path.join(args.rootdir, "results")
    args.exp_name = "exp_" + args.timestamp
    args.exp_path = os.path.join(args.result_dir, args.exp_name)
    os.makedirs(args.exp_path, exist_ok=True)
    args.csv_dir = os.path.join(args.rootdir, "csv")
    mkdir(args.config_dir)
    mkdir(args.result_dir)
    mkdir(args.csv_dir)
    if not hasattr(args, "save_result_file") or args.save_result_file is None:
        args.save_result_file = "default.csv"

    # Dump these parameters for reproducibility.
    # These should be inside a different directory than the results,
    # because then you have to open each directory separately to see what it contained!
    dump_parameters(args)
    return args
