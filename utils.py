import collections
import csv
import os
import pickle
import sys
from itertools import chain

import partition
import torch
import train as cifar_train
import vgg
from basic_config import PATH_TO_CIFAR, PATH_TO_VGG
from log import logger


sys.path.append(PATH_TO_CIFAR)
sys.path.append(PATH_TO_VGG)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def mkdir(path):
    os.makedirs(path, exist_ok=True)
    # if not os.path.exists(path):
    #     os.makedirs(path)


def pickle_obj(obj, path, mode="wb", protocol=pickle.HIGHEST_PROTOCOL):
    """
    Pickle object 'obj' and dump at 'path' using specified
    'mode' and 'protocol'
    Returns time taken to pickle
    """

    import time

    st_time = time.perf_counter()
    pkl_file = open(path, mode)
    pickle.dump(obj, pkl_file, protocol=protocol)
    end_time = time.perf_counter()
    return end_time - st_time


def dict_union(*args):
    return dict(chain.from_iterable(d.items() for d in args))


def save_results_params_csv(path, results_dic, args, ordered=True):
    if os.path.exists(path):
        add_header = False
    else:
        mkdir(os.path.dirname(path))
        add_header = True

    with open(path, mode="a") as csv_file:
        if args.deprecated is not None:
            params = args
        else:
            params = vars(args)

        # Merge with params dic
        if ordered:
            # sort the parameters by name before saving
            params = collections.OrderedDict(sorted(params.items()))

        results_and_params_dic = dict_union(results_dic, params)

        writer = csv.DictWriter(csv_file, fieldnames=results_and_params_dic.keys())

        # Add key header if file doesn't exist
        if add_header:
            writer.writeheader()

        # Add results and params record
        writer.writerow(results_and_params_dic)


def save_datasets(args, personal_trainset, personal_testset, other_trainset, other_testset):
    torch.save(personal_trainset, os.path.join(args.exp_path, "personal_train.pth"))
    torch.save(personal_testset, os.path.join(args.exp_path, "personal_test.pth"))
    torch.save(other_trainset, os.path.join(args.exp_path, "other_train.pth"))
    torch.save(other_testset, os.path.join(args.exp_path, "other_test.pth"))


def isnan(x):
    return x != x


def get_model_activations(args, models, config=None, layer_name=None, selective=False, personal_dataset=None):
    import compute_activations
    from data import get_dataloader

    if args.activation_histograms and args.act_num_samples > 0:
        if args.dataset == "mnist":
            unit_batch_train_loader, _ = get_dataloader(args, unit_batch=True)
        else:
            if config is None:
                config = args.config  # just use the config in arg
            unit_batch_train_loader, _ = cifar_train.get_dataset(config, unit_batch_train=True)

        if args.activation_mode is None:
            activations = compute_activations.compute_activations_across_models(
                args, models, unit_batch_train_loader, args.act_num_samples
            )
        else:
            if selective and args.update_acts:
                activations = compute_activations.compute_selective_activation(
                    args, models, layer_name, unit_batch_train_loader, args.act_num_samples
                )
            else:
                if personal_dataset is not None:
                    # personal training set is passed which consists of (inp, tgts)
                    logger.info("using the one from partition")
                    loader = partition.to_dataloader_from_tens(personal_dataset[0], personal_dataset[1], 1)
                else:
                    loader = unit_batch_train_loader

                activations = compute_activations.compute_activations_across_models_v1(
                    args, models, loader, args.act_num_samples, mode=args.activation_mode
                )

    else:
        activations = None

    return activations


def get_number_of_layers(model):
    num_layers = 0

    for layer_name, _ in model.named_parameters():
        # shortcut 1 is also batchnorm
        if (
            ("bias" in layer_name)
            or ("bn" in layer_name)
            or ("shortcut.1" in layer_name)
            or ("BatchNorm2d" in layer_name)
        ):
            continue

        num_layers += 1

    return num_layers


def get_model_layers_cfg(model_name):
    logger.info("model_name is %s", model_name)
    if model_name == "mlpnet" or model_name[-7:] == "encoder":
        return None
    elif model_name[0:3].lower() == "vgg":
        cfg_key = model_name[0:5].upper()
    elif model_name[0:6].lower() == "resnet":
        return None
    return vgg.cfg[cfg_key]


def to_first_position(arr, idx):
    tmp = arr[0]
    arr[0] = arr[idx]
    arr[idx] = tmp

    return arr


def _get_config(args):
    import importlib

    config = None
    second_config = None
    if args.dataset.lower()[0:7] == "cifar10":
        config_file_tail = "_cifar10_baseline"
    elif args.dataset.lower() == "tinyimagenet":
        config_file_tail = "_tinyimagenet_baseline"
    elif args.dataset.lower() == "esc50":
        config_file_tail = "_esc50_baseline"

    if args.dataset.lower()[0:7] != "mnist":
        if len(args.model_name_list) == 0:
            try:
                config_file = importlib.import_module("hyperparameters." + args.model_name + config_file_tail)
                config = config_file.config
                config["dataset"] = args.dataset
            except ImportError:
                raise NotImplementedError

            if args.second_model_name is not None:
                try:
                    config_file = importlib.import_module(
                        "hyperparameters." + args.second_model_name + config_file_tail
                    )
                    second_config = config_file.config
                    second_config["dataset"] = args.dataset
                except ImportError:
                    raise NotImplementedError
            else:
                second_config = config

            return config, second_config

        else:
            config_list = []
            for idx in range(args.num_models):
                try:
                    model_name = args.model_name_list[idx]
                    config_file = importlib.import_module("hyperparameters." + model_name + config_file_tail)
                    config_dict = config_file.config
                    config_dict["dataset"] = args.dataset
                    config_list.append(config_dict)
                except ImportError:
                    raise NotImplementedError

            return config_list

    return [config, second_config]


def get_model_size(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
