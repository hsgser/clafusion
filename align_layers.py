import random
import sys
import time

import numpy as np
import ot
import routines
import torch
import torch.nn.functional as F
import train as cifar_train
import utils
from basic_config import PATH_TO_CIFAR
from data import get_dataloader
from ground_metric import GroundMetric
from layer_similarity import cca, cka, gram_linear
from log import logger
from torch.autograd import Variable
from wasserstein_ensemble import get_network_from_param_list


sys.path.append(PATH_TO_CIFAR)


vgg_cfg = {
    "vgg8": [64, "M", 128, "M", 256, "M", 512, "M", 512, "M"],
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg11_quad": [64, "M", 512, "M", 1024, 1024, "M", 2048, 2048, "M", 2048, 512, "M"],
    "vgg11_doub": [64, "M", 256, "M", 512, 512, "M", 1024, 1024, "M", 1024, 512, "M"],
    "vgg11_half": [64, "M", 64, "M", 128, 128, "M", 256, 256, "M", 256, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13_quad": [64, 256, "M", 512, 512, "M", 1024, 1024, "M", 2048, 2048, "M", 2048, 512, "M"],
    "vgg13_doub": [64, 128, "M", 256, 256, "M", 512, 512, "M", 1024, 1024, "M", 1024, 512, "M"],
    "vgg13_half": [64, 32, "M", 64, 64, "M", 128, 128, "M", 256, 256, "M", 256, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

resnet_cfg = {"resnet18": [2, 2, 2, 2], "resnet34": [3, 4, 6, 3]}


def get_number_of_neurons(network):
    """
    Get number of neurons of each hidden layer in MLPNet

    :param netwokrs: a network
    """
    n_neurons = []

    for _, layer_weight in network.named_parameters():
        n_neurons.append(layer_weight.size(0))

    return np.array(n_neurons)[:-1]


def get_activation_matrices(args, networks, personal_dataset=None, config=None, is_wd=False):
    """
    Get activation matrix for each layer of each network

    :param args: config parameters
    :param networks: list of networks
    :param personal_dataset: personalized dataset
    :param config: hyperparameters for CNNs, default = None
    :param is_wd: whether the cost is Wassersten distance
    :return: list of activation matrices for each model
    """
    activations = utils.get_model_activations(args, networks, personal_dataset=personal_dataset, config=config)
    list_act = []

    for _, model_dict in activations.items():
        model_act = []

        for _, layer_act in model_dict.items():
            if is_wd:
                reorder_dim = [l for l in range(2, len(layer_act.shape))]
                reorder_dim.extend([0, 1])
                layer_act = layer_act.permute(*reorder_dim).contiguous()
            layer_act = layer_act.view(layer_act.size(0), -1)
            model_act.append(layer_act)

        # exclude the activation of output layer
        list_act.append(model_act[:-1])

    return list_act


def get_wasserstein_distance(a, b, args):
    mu = np.ones(len(a)) / len(a)
    nu = np.ones(len(b)) / len(b)
    ground_metric_object = GroundMetric(args)
    logger.info(f"{a.size()}, {b.size()}")
    M = ground_metric_object.process(a, b)
    M_cpu = M.data.cpu().numpy()

    return ot.emd2(mu, nu, M_cpu)


def get_cost(a, b, args, cost="euclidean"):
    if cost == "euclidean":
        return (a - b) ** 2
    elif cost == "cca":
        return 1 - cca(a, b)
    elif cost == "cka":
        return 1 - cka(gram_linear(a), gram_linear(b))
    elif cost == "wd":
        return get_wasserstein_distance(a, b, args)
    elif cost == "cosine":
        return F.normalize(a, dim=0) @ F.normalize(b, dim=0).T
    else:
        raise NotImplementedError


def get_cost_matrix(x, y, args):
    """
    Compute the cost matrix between two measures.

    :param x: list of measures, size m
    :param y: list of measures, size n
    :param args: config parameters
    :return: cost matrix, size m x n
    """

    cost = args.layer_metric
    m, n = len(x), len(y)
    if m * n == 0:
        return []
    C = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            C[i][j] = get_cost(x[i], y[j], args, cost=cost)

    return C


def align(C, start_layer_idx=2, free_end_layers=0, mapping_type="cla"):
    """
    Compute the optimal map between hidden layers of two models using dynamic programming.

    :param C: cost matrix
    :param start_layer_idx: the layer index to start aligning, default = 2,
        i.e., start from the second hidden layer
    :param free_end_layers: match last free_end_layers hidden layers
        of two models, default = 0
    :return: list of layer indices of the large model
    """
    m, n = C.shape
    assert m >= n

    if mapping_type == "random":
        return random_align(C, start_layer_idx=start_layer_idx, free_end_layers=0)
    elif mapping_type == "chain":
        return np.arange(1, n + 1)

    if n == 1:
        return np.array([1.0])

    F = np.zeros((n + 1, m + 1))

    # compute the diagonal of F
    sum = 0
    for k in range(1, n + 1):
        sum += C[k - 1, k - 1]
        F[k, k] = sum

    # the first start_layer_idx hidden layer of two models match
    if start_layer_idx > 1:
        for l in range(start_layer_idx, m + 1 - free_end_layers):
            F[start_layer_idx - 1, l] = F[start_layer_idx - 1, start_layer_idx - 1]

    # forward recursion
    for k in range(start_layer_idx, n + 1 - free_end_layers):
        for l in range(k + 1, m + 1 - free_end_layers):
            F[k, l] = min(F[k, l - 1], F[k - 1, l - 1] + C[l - 1, k - 1])

    # backward recursion
    A = np.ones(n + 1)
    k, l = n - free_end_layers, m - free_end_layers

    for idx in range(1, start_layer_idx):
        A[idx] = idx

    if free_end_layers > 0:
        for idx in range(free_end_layers):
            A[n - idx] = m - idx

    while k >= start_layer_idx:
        while (l >= k + 1) and (F[k, l] == F[k, l - 1]):
            l -= 1

        A[k] = l
        k -= 1
        l -= 1

    # because the first hidden layer is layer at index 1
    return A[1:]


def random_align(C, start_layer_idx=2, free_end_layers=0):
    m, n = C.shape
    assert m >= n

    A = np.ones(n)

    for idx in range(1, start_layer_idx):
        A[idx] = idx

    if free_end_layers > 0:
        for idx in range(free_end_layers):
            A[n - idx - 1] = m - idx

    num_remained_layers = (n - free_end_layers) - (start_layer_idx - 1)
    list_remained_layers = range(start_layer_idx, m - free_end_layers + 1)
    A[start_layer_idx - 1 : n - free_end_layers] = sorted(random.sample(list_remained_layers, num_remained_layers))

    return A


def align_conv_layers(x, y, model_names, args):
    """
    Align the Covolution parts of two VGG models.
    Divide into chunks separated by max pooling layer
    then align each pair of chunks separately.

    :param C: cost matrix
    :param x: layer measure of the first model
    :param y: layer measure of the second model
    :param args: config parameters
    :param model_names: list of model name
    :return: list of layer indices of the large model
    """
    assert "vgg" in model_names[1]
    groups = []

    for name in model_names:
        layer_idx = []
        model_config = vgg_cfg[name.split("_")[0]]
        idx = 0
        for layer_size in model_config:
            if layer_size == "M":
                layer_idx.append(idx)
            else:
                idx += 1
        groups.append([0] + layer_idx)

    logger.info("Groups of model 0: {}".format(groups[0]))
    logger.info("Groups of model 1: {}".format(groups[1]))
    # check two models have the same number of max pooling layers
    assert len(groups[0]) == len(groups[1])
    A = []

    for idx in range(len(groups[0]) - 1):
        C = get_cost_matrix(x[groups[0][idx] : groups[0][idx + 1]], y[groups[1][idx] : groups[1][idx + 1]], args)
        logger.info(
            "Cost matrix between layers {}-{} of model 0 and layers {}-{} of model 1 is \n{}".format(
                groups[0][idx] + 1, groups[0][idx + 1], groups[1][idx] + 1, groups[1][idx + 1], C
            )
        )
        A_chunk = align(
            C,
            start_layer_idx=args.start_layer_idx,
            free_end_layers=args.free_end_layers,
            mapping_type=args.mapping_type,
        )
        A.append(A_chunk + groups[0][idx])

    return np.concatenate(A)


def align_resnet_block(x, y, model_names, args):
    """
    Align the Covolution parts of two RESNET models.
    Divide into stages then align each pair of stages separately.
    In each stage, align blocks instead of layers.

    :param C: cost matrix
    :param x: layer measure of the first model
    :param y: layer measure of the second model
    :param args: config parameters
    :param model_names: list of model name
    :return: list of layer indices of the large model
    """
    assert ("resnet18" in model_names[0]) or ("resnet34" in model_names[0])
    assert ("resnet18" in model_names[1]) or ("resnet34" in model_names[1])
    num_layers_per_block = 2
    groups = []

    for name in model_names:
        blocks = []
        model_config = resnet_cfg[name.split("_")[0]]
        start_idx = 1
        end_idx = 1
        for i, num_blocks in enumerate(model_config):
            layer_idx = []
            for j in range(num_blocks):
                end_idx += num_layers_per_block
                if j == 0 and i != 0:
                    end_idx += 1
                layer_idx.append(list(range(start_idx, end_idx)))
                start_idx = end_idx
            blocks.append(layer_idx)
        groups.append(blocks)
    logger.info("Groups of model 0: {}".format(groups[0]))
    logger.info("Groups of model 1: {}".format(groups[1]))
    # check two models have the same number of groups
    assert len(groups[0]) == len(groups[1])
    A = [0]
    for idx in range(len(groups[0])):
        blocks0 = groups[0][idx]
        blocks1 = groups[1][idx]
        block_end_idx0 = [block_idx[-1] for block_idx in blocks0]
        block_end_idx1 = [block_idx[-1] for block_idx in blocks1]
        block_measure0 = [x[i] for i in block_end_idx0]
        block_measure1 = [y[i] for i in block_end_idx1]
        C = get_cost_matrix(block_measure0, block_measure1, args)
        logger.info("Cost matrix between group {} of models 0 and 1 is \n{}".format(idx, C))
        for block_idx in align(
            C,
            start_layer_idx=args.start_layer_idx,
            free_end_layers=args.free_end_layers,
            mapping_type=args.mapping_type,
        ):
            A.extend(blocks0[int(block_idx) - 1])

    return np.array(A) + 1


def align_to_transfer_map(mapping, model_type):
    """Assume that the last layer of vgg or resnet is a FC layer."""
    if model_type == "vgg":
        return mapping[:-1]
    new_mapping = []

    for idx in mapping:
        if idx in [9, 18, 31]:
            continue
        elif idx > 31:
            new_mapping.append(idx - 3)
        elif idx > 18:
            new_mapping.append(idx - 2)
        elif idx > 9:
            new_mapping.append(idx - 1)
        else:
            new_mapping.append(idx)

    return new_mapping


def transfer_to_align_map(mapping, model_type):
    """Align map index starts from 1"""
    assert mapping == sorted(mapping)
    if model_type != "resnet":
        return [idx + 1 for idx in mapping]

    new_mapping = []

    for idx in mapping:
        if idx > 28:
            new_mapping.append(idx + 4)
        elif idx > 16:
            new_mapping.append(idx + 3)
        elif idx > 8:
            new_mapping.append(idx + 2)
        else:
            new_mapping.append(idx + 1)

    new_mapping.extend([10, 19, 32])

    return sorted(new_mapping)


def make_identity_mat(n_neurons, gpu_id=-1):
    """
    A utility function to create an identity weight matrix.
    Assumption: kernel_size = 3, padding = 1, stride = 1
    """
    weight_mat = torch.eye(n_neurons)
    if gpu_id != -1:
        weight_mat = weight_mat.cuda(gpu_id)
    return Variable(weight_mat)


def make_identity_conv(n_channels, model_name, gpu_id=-1):
    """
    A utility function to create an identity convolution layer.
    Assumption: kernel_size = 3, padding = 1, stride = 1
    """
    kernels = torch.zeros(n_channels, n_channels, 3, 3)
    if "vgg" in model_name:
        for idx in range(n_channels):
            kernels[idx, idx, 1, 1] = 1
    if gpu_id != -1:
        kernels = kernels.cuda(gpu_id)
    return Variable(kernels)


def add_layers_into_smaller_network(args, network0, network1, num_layer0, num_layer1, A, model_names):
    """
    Add layers into the smaller model.

    :param args: config parameters
    :param network0: the large model
    :param network1: the small model
    :param num_layer0: the number of layers of the large model
    :param num_layer1: the number of layers of the small model
    :param A: optimal map between layers of two models
    :param model_names: list of model names
    :return: list of weight matrix of the new model and
            the updated args
    """
    new_weight = []
    m = num_layer0 - 1
    n = num_layer1 - 1
    l0 = 1
    is_conv = False
    A = np.append(A, m + 1)

    for l1, (layer_name, layer_weight) in enumerate(network1.named_parameters()):
        if len(layer_weight.shape) > 2:
            is_conv = True
        else:
            is_conv = False
        logger.info("Layer {} is {} layer".format(layer_name, "conv" if is_conv else "fc"))
        if l1 == 0:
            # if the first hidden layer of the small model does not
            # map to the first hidden layer of the large model
            while l0 < A[0]:
                logger.info("Add identity weight at hidden layer {}".format(l0))
                if is_conv:
                    # here assume that kernel_size = 3, padding = 1, stride = 1
                    out_channels, in_channels, height, width = layer_weight.size()
                    new_weight.append(make_identity_conv(in_channels, model_names[1], gpu_id=args.gpu_id))
                else:
                    new_weight.append(make_identity_mat(layer_weight.shape[1], gpu_id=args.gpu_id))
                setattr(args, "num_hidden_nodes" + str(l0), layer_weight.shape[1])
                l0 += 1

        # add the current layer weight
        new_weight.append(layer_weight.data)
        setattr(args, "num_hidden_nodes" + str(l0), layer_weight.shape[0])
        l0 += 1

        if l1 <= n - 1:
            # fill the gap between two consecutive maps
            while l0 < A[l1 + 1]:
                logger.info("Add identity weight at hidden layer {}".format(l0))
                if is_conv:
                    # here assume that kernel_size = 3, padding = 1, stride = 1
                    out_channels, in_channels, height, width = layer_weight.size()
                    new_weight.append(make_identity_conv(out_channels, model_names[1], gpu_id=args.gpu_id))
                else:
                    new_weight.append(make_identity_mat(layer_weight.shape[0], gpu_id=args.gpu_id))
                setattr(args, "num_hidden_nodes" + str(l0), layer_weight.shape[0])
                l0 += 1

    setattr(args, "num_hidden_layers", m)
    # change model config
    model_config = None
    if "vgg" in model_names[0]:
        model_config = vgg_cfg[model_names[0].split("_")[0]]
        layer_idx = 1
        for config_index in range(len(model_config)):
            if model_config[config_index] != "M":
                model_config[config_index] = getattr(args, "num_hidden_nodes" + str(layer_idx))
                layer_idx += 1
        logger.info("New VGG model config {}".format(model_config))

    logger.info("Fused model config is {}".format(model_config))
    setattr(args, "model_config", model_config)

    return new_weight, args, model_config


def add_layers_into_smaller_network_v2(args, network0, network1, num_layer0, num_layer1, A, model_names):
    """
    Add layers into the smaller model. The difference from the previous version
    is that the identity weight matrix is added after the original weight matix.

    :param args: config parameters
    :param network0: the large model
    :param network1: the small model
    :param num_layer0: the number of layers of the large model
    :param num_layer1: the number of layers of the small model
    :param A: optimal map between layers of two models
    :param model_names: list of model names
    :return: list of weight matrix of the new model and
            the updated args
    """
    new_weight = []
    m = num_layer0 - 1
    n = num_layer1 - 1
    l0 = 0
    is_conv = False
    A = np.append(A, m + 1)

    for l1, (layer_name, layer_weight) in enumerate(network1.named_parameters()):
        if len(layer_weight.shape) > 2:
            is_conv = True
        else:
            is_conv = False
        logger.info("Layer {} is {} layer".format(layer_name, "conv" if is_conv else "fc"))

        # add the current layer weight
        new_weight.append(layer_weight.data)
        l0 += 1
        setattr(args, "num_hidden_nodes" + str(l0), layer_weight.shape[0])

        # fill the gap between two consecutive maps
        while l0 < A[l1]:
            logger.info("Add identity weight at hidden layer {}".format(l0))
            if is_conv:
                # here assume that kernel_size = 3, padding = 1, stride = 1
                out_channels, in_channels, height, width = layer_weight.size()
                new_weight.append(make_identity_conv(out_channels, gpu_id=args.gpu_id))
            else:
                new_weight.append(make_identity_mat(layer_weight.shape[0], gpu_id=args.gpu_id))
            l0 += 1
            setattr(args, "num_hidden_nodes" + str(l0), layer_weight.shape[0])

    setattr(args, "num_hidden_layers", m)
    # change model config
    model_config = None
    if "vgg" in model_names[0]:
        model_config = []
        for idx in range(m):
            model_config.append(getattr(args, "num_hidden_nodes" + str(idx + 1)))
        for idx in range(1, n):
            model_config.insert(int(A[n - idx]) - 1, "M")
        model_config.append("M")
        logger.info("New VGG model config {}".format(model_config))

    return new_weight, args, model_config


def approximate_relu(act_mat, num_columns, args, method="sum"):
    """
    Approximate ReLU activation function by a diagonal matrix

    :param act_mat: the pre-activation matrix, i.e. before applying ReLU
    :param num_columns: the number of nodes in the previous layer
    :param args: config parameters
    :param method: method to approximate the sign of activation ["sum", "majority", "avg"], default = "sum"
    :return: a matrix in which each row has the same value
    """
    if method == "sum":
        act_vec = act_mat.sum(axis=0) >= 0
    elif method == "majority":
        act_vec = (act_mat > 0).mean(axis=0) >= 0.5
    elif method == "avg":
        act_vec = ((act_mat > 0) * 1.0).mean(axis=0)
    else:
        raise NotImplementedError

    if isinstance(act_vec, torch.Tensor):
        return act_vec.unsqueeze(0).repeat(num_columns, 1).T
    else:
        return np.tile(act_vec, (num_columns, 1)).T


# Do not support two CNNs
def merge_layers_in_larger_network(args, network0, network1, num_layer0, num_layer1, acts, A, method="sum"):
    """
    Merge consecutive layers in the larger model.

    :param args: config parameters
    :param network0: the large model
    :param network1: the small model
    :param num_layer0: the number of layers of the large model
    :param num_layer1: the number of layers of the small model
    :param acts: list of activation matrices for hidden layers
    :param A: optimal map between layers of two models
    :param method: method to approximate the sign of activation ["sum", "majority"], default = "sum"
    :return: list of weight matrix of the new model and
            the updated args
    """
    new_weight = []
    m = num_layer0 - 1
    n = num_layer1 - 1
    l1 = 0
    if args.dataset == "mnist":
        input_dim = 784
    elif args.dataset == "cifar10":
        input_dim = 3072
    else:
        raise ValueError
    pre_weight = torch.eye(input_dim).cuda(args.gpu_id)

    for l0, (_, layer_weight) in enumerate(network0.named_parameters()):
        if l1 < n:
            if l0 + 1 < A[l1]:
                # if the current hidden layer does not map
                # to any hidden layer in the smaller model
                # then merge it into the previous matched one
                # suppose the activation function is ReLU
                logger.info(
                    "Approximate ReLU at hidden layer {} with activation of shape {}".format(l0 + 1, acts[l0].shape)
                )
                act_vec = approximate_relu(acts[l0], layer_weight.shape[1], args, method=method)
                assert act_vec.shape == layer_weight.shape
                if not isinstance(act_vec, torch.Tensor):
                    act_vec = torch.from_numpy(act_vec).cuda(args.gpu_id)
                layer_weight = layer_weight * act_vec
                pre_weight = layer_weight @ pre_weight
            else:
                pre_weight = layer_weight @ pre_weight
                setattr(args, "num_hidden_nodes" + str(l1 + 1), layer_weight.shape[0])
                l1 += 1
                new_weight.append(pre_weight)
                pre_weight = torch.eye(layer_weight.shape[0]).cuda(args.gpu_id)
        elif l0 < m:
            # if the last hidden layer of the small model does not
            # map to the last hidden layer of the large model
            logger.info(
                "Approximate ReLU at hidden layer {} with activation of shape {}".format(l0 + 1, acts[l0].shape)
            )
            act_vec = approximate_relu(acts[l0], layer_weight.shape[1], args, method=method)
            assert act_vec.shape == layer_weight.shape
            if not isinstance(act_vec, torch.Tensor):
                act_vec = torch.from_numpy(act_vec).cuda(args.gpu_id)
            layer_weight = layer_weight * act_vec
            pre_weight = layer_weight @ pre_weight
        else:
            # the last hidden layer of the large model
            pre_weight = layer_weight @ pre_weight
            new_weight.append(pre_weight)

        # args["num_hidden_layers"] = n
        setattr(args, "num_hidden_layers", n)

    return new_weight, args


def get_alignment_map(args, networks, num_layers, model_names, personal_dataset=None):
    for idx in range(2):
        logger.info(f"Model {idx} has {num_layers[idx]-1} hidden layers")

    logger.info(f"Using layer measure = {args.layer_measure} and metric = {args.layer_metric}")
    # measure time
    act_time = 0
    align_time = 0
    align_st_time = time.perf_counter()
    if args.layer_measure == "index":
        x = np.arange(1, num_layers[0] - 1)
        y = np.arange(1, num_layers[1] - 1)
        assert args.layer_metric == "euclidean"
    elif args.layer_measure == "neuron":
        x = get_number_of_neurons(networks[0])
        y = get_number_of_neurons(networks[1])
        assert args.layer_metric == "euclidean"
    elif args.layer_measure == "activation":
        act_st_time = time.perf_counter()
        is_wd = args.layer_metric == "wd"
        x, y = get_activation_matrices(
            args, networks, personal_dataset=personal_dataset, config=args.config, is_wd=is_wd
        )
        act_end_time = time.perf_counter()
        act_time = act_end_time - act_st_time
        assert args.layer_metric in ["cka", "cca", "wd"]

    # get alignment map
    classifier_idx = [None, None]
    for i in range(2):
        for idx, (_, layer_weight) in enumerate(networks[i].named_parameters()):
            if len(layer_weight.shape) == 2:
                break
        classifier_idx[i] = idx
        logger.info(f"FC layers of model {i} start from {idx}")
    if classifier_idx[0] * classifier_idx[1] > 0:
        if "vgg" in model_names[0]:
            A1 = align_conv_layers(x[: classifier_idx[0]], y[: classifier_idx[1]], model_names, args)
        elif "resnet" in model_names[0]:
            A1 = align_resnet_block(x[: classifier_idx[0]], y[: classifier_idx[1]], model_names, args)
        else:
            raise NotImplementedError
        if classifier_idx[0] < len(x):
            C = get_cost_matrix(x[classifier_idx[0] :], y[classifier_idx[1] :], args)
            logger.info(
                "Cost matrix between layers {}-{} of model 0 and layers {}-{} of model 1 is \n{}".format(
                    classifier_idx[0] + 1, len(x), classifier_idx[1] + 1, len(y), C
                )
            )
            A2 = align(
                C,
                start_layer_idx=args.start_layer_idx,
                free_end_layers=args.free_end_layers,
                mapping_type=args.mapping_type,
            )
        else:
            A2 = np.array([])
    else:
        A1 = []
        C = get_cost_matrix(x[classifier_idx[0] :], y[classifier_idx[1] :], args)
        logger.info(
            "Cost matrix between layers {}-{} of model 0 and layers {}-{} of model 1 is \n{}".format(
                classifier_idx[0] + 1, len(x), classifier_idx[1] + 1, len(y), C
            )
        )
        A2 = align(
            C,
            start_layer_idx=args.start_layer_idx,
            free_end_layers=args.free_end_layers,
            mapping_type=args.mapping_type,
        )
    A = np.concatenate([A1, A2 + classifier_idx[0]])
    logger.info("Optimal map from model 1 to model 0 is {}".format(A))
    setattr(args, "optimal_layer_alignment", list(A))
    align_end_time = time.perf_counter()
    align_time = align_end_time - align_st_time
    setattr(args, "align_layers_time", align_time)
    logger.info(f"Align layers time: {align_time}")

    return A, act_time, args, x


def balance_number_of_layers(args, networks, num_layers, model_names, A, act_time, x, personal_dataset=None):
    balance_time = 0
    logger.info("------- Balance the number of layers -------")
    model_configs = [None, None]
    if args.balance_method == "add":
        model_index = 1
        args.aligned_model_index = model_index
        logger.info(f"Add layers into model {model_index}")
        balance_st_time = time.perf_counter()
        new_weights, args, model_configs[model_index] = add_layers_into_smaller_network(
            args, networks[0], networks[1], num_layers[0], num_layers[1], A, model_names
        )
        balance_end_time = time.perf_counter()
        balance_time = balance_end_time - balance_st_time
        if "resnet" not in model_names[1]:
            args.fused_model_name = model_names[0]
        else:
            # resnetxx_nobias_nobn, resnetxx_half_nobias_nobn, resnetxx_doub_nobias_nobn
            model0_sub_type = model_names[0].split("_")
            model1_sub_type = model_names[1].split("_")
            if len(model1_sub_type) in [2, 3]:
                args.fused_model_name = model_names[0]
            elif len(model1_sub_type) == 4:
                if len(model0_sub_type) == 4:
                    model0_sub_type[1] = model1_sub_type[1]
                elif len(model0_sub_type) == 3:
                    model0_sub_type.insert(1, model1_sub_type[1])
                else:
                    raise NotImplementedError
                args.fused_model_name = "_".join(model0_sub_type)
            else:
                raise NotImplementedError
    elif args.balance_method == "merge":
        assert "vgg" not in args.model_name
        model_index = 0
        args.aligned_model_index = model_index
        logger.info(f"Merge layers in model {model_index}")
        logger.info(f"Approximate ReLU using method {args.relu_approx_method}")
        # get activation matrices
        if args.layer_measure == "activation" and args.layer_metric != "wd":
            act0 = x
        else:
            act_st_time = time.perf_counter()
            act0, _ = get_activation_matrices(args, networks, personal_dataset=personal_dataset, config=args.config)
            act_end_time = time.perf_counter()
            act_time = act_end_time - act_st_time
        balance_st_time = time.perf_counter()
        new_weights, args = merge_layers_in_larger_network(
            args, networks[0], networks[1], num_layers[0], num_layers[1], act0, A, method=args.relu_approx_method
        )
        balance_end_time = time.perf_counter()
        balance_time = balance_end_time - balance_st_time + act_time
        args.fused_model_name = model_names[1]

    setattr(args, "balance_layers_time", balance_time)
    logger.info(f"Balance layers time: {balance_time}")
    logger.info("Obtain test dataloaders")
    if args.dataset == "mnist":
        _, test_loader = get_dataloader(args)
    else:
        _, test_loader = cifar_train.get_dataset(args.config, to_download=args.to_download)

    if args.parse_config:
        logger.info(
            "Change configuration from list of hidden_layer_sizes to num_hidden_layers/num_hidden_nodes style."
        )
        setattr(args, "parse_config", False)
    logger.info("Get new model from param list")
    new_acc, new_network = get_network_from_param_list(
        args, new_weights, test_loader, model_name=args.fused_model_name, model_config=model_configs[model_index]
    )

    return new_acc, new_network, args, model_configs, model_index


def print_model_info(networks, accuracies, epochs):
    """
    A utility functions to print model info.

    :param: list of networks
    :param: list of accuracies
    :param: list of epochs
    """
    for i, network in enumerate(networks):
        logger.info("Model {} has accuracy of {} at epoch {}".format(i, accuracies[i], epochs[i]))
        # for weight in network.parameters():
        #     logger.info(weight.size())


def align_two_models(args, networks, accuracies, num_layers, epochs, model_names=None):
    """
    Align layers of two models.

    :param args: config parameters
    :param networks: list of models
    :param accuracies: list of accuracies
    :param num_layers: list of number of layers
    :param epochs: list of epochs
    :param model_names: list of model names
    :return: list of updated models, accuracies and config parameters
    """
    logger.info("------- Align layers of two models -------")
    if model_names is None:
        model_names = args.model_name_list
    if num_layers[0] < num_layers[1]:
        logger.info("Shuffle two models so that model 0 has more layers than model 1")
        networks = networks[::-1]
        accuracies = accuracies[::-1]
        num_layers = num_layers[::-1]
        epochs = epochs[::-1]
        model_names = model_names[::-1]

    A, act_time, args, x = get_alignment_map(args, networks, num_layers, model_names)
    logger.info("------- Before balancing number of layers -------")
    print_model_info(networks, accuracies, epochs)
    new_acc, new_network, args, model_configs, model_index = balance_number_of_layers(
        args, networks, num_layers, model_names, A, act_time, x
    )

    logger.info("------- After balancing number of layers -------")
    networks[model_index] = new_network
    accuracies[model_index] = new_acc
    print_model_info(networks, accuracies, epochs)

    if args.dump_final_models:
        logger.info(f"Dump new model {model_index}")
        routines.save_final_model(args, f"aligned_model{model_index}", new_network, new_acc)

    logger.info("Change config parameters to match model 1")

    for idx, weight in enumerate(networks[1].parameters()):
        if (idx != 0) and (len(weight.shape) > 1):
            setattr(args, "num_hidden_nodes" + str(idx), weight.size(1))

    return networks, accuracies, args, model_configs[1]


def print_data_separated_model_info(networks, accuracies, local_accuracies, choices, epochs):
    """
    A utility functions to print model info

    :param: list of networks
    :param: list of accuracies
    :param: list of local accuracies
    :param: list of choices
    :param: list of epochs
    """
    for i, network in enumerate(networks):
        logger.info(
            "Model {} has local accuracy of {} and overall accuracy of {} for choice {} at epoch {}".format(
                i, local_accuracies[i], accuracies[i], choices[i], epochs[i]
            )
        )

        # for weight in network.parameters():
        #     logger.info(weight.size())


def align_two_data_separated_models(
    args, networks, accuracies, local_acccuracies, num_layers, personal_dataset, local_test_loaders, choices, epochs
):
    """
    Align layers of two models.

    :param args: config parameters
    :param networks: list of models
    :param accuracies: list of accuracies
    :param local_accuraies: list of local accuracies
    :param num_layers: list of number of layers
    :param personal_dataset: personalized dataset
    :param local_test_loaders: list of local test loaders for each model
    :param choices: list of choices
    :param epochs: list of epochs
    :return: list of updated models, accuracies and config parameters
    """
    logger.info("------- Align layers of two models -------")
    if num_layers[0] < num_layers[1]:
        logger.info("Shuffle two models so that model 0 has more layers than model 1")
        networks = networks[::-1]
        accuracies = accuracies[::-1]
        local_acccuracies = local_acccuracies[::-1]
        num_layers = num_layers[::-1]
        local_test_loaders = local_test_loaders[::-1]
        choices = choices[::-1]
        epochs = epochs[::-1]
        model_names = args.model_name_list[::-1]
    else:
        model_names = args.model_name_list

    A, act_time, args, x = get_alignment_map(
        args, networks, num_layers, model_names, personal_dataset=personal_dataset
    )

    logger.info("------- Before balancing number of layers -------")
    print_data_separated_model_info(networks, accuracies, local_acccuracies, choices, epochs)
    new_acc, new_network, args, model_configs, model_index = balance_number_of_layers(
        args, networks, num_layers, model_names, A, act_time, x, personal_dataset=personal_dataset
    )
    logger.info("Get local accuracy of new model")
    log_dict = {}
    log_dict["train_losses"] = []
    log_dict["train_counter"] = []
    log_dict["local_test_losses"] = []
    log_dict["test_losses"] = []
    if local_test_loaders[model_index]:
        new_local_acc = routines.test(args, new_network, local_test_loaders[model_index], log_dict, is_local=True)
    else:
        # if local dataset is unavailable
        new_local_acc = new_acc

    logger.info("------- After balancing number of layers -------")
    networks[model_index] = new_network
    accuracies[model_index] = new_acc
    local_acccuracies[model_index] = new_local_acc
    print_data_separated_model_info(networks, accuracies, local_acccuracies, choices, epochs)

    if args.dump_final_models:
        logger.info(f"Dump new model {model_index}")
        setattr(args, "n_epochs", epochs[model_index])
        routines.save_final_data_separated_model(
            args, f"aligned_model{model_index}", new_network, new_local_acc, new_acc, choices[model_index]
        )

    logger.info("Change config parameters to match model 1")

    for idx, weight in enumerate(networks[1].parameters()):
        if idx != 0:
            setattr(args, "num_hidden_nodes" + str(idx), weight.size(1))

    return networks, accuracies, args, model_configs[1]
