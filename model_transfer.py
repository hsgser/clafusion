"""
This file is adapted from https://anonymous.4open.science/r/6ab184dc-3c64-4fdd-ba6d-1e5097623dfd/a_hetero_model_transfer.py
"""

from __future__ import division, print_function

import torch
import torch.nn as nn
from log import logger
from torch.nn import functional as F
from tqdm import tqdm


def circle_copy(arr1, arr2, arr_dims):
    c1 = arr1.size(0)
    c2 = arr2.size(0)

    for i in range(c1):
        j = i % c2
        if arr_dims == 1:
            arr1[i] = arr2[j]
        else:
            arr1[i, :, :] = arr2[j, :, :]
    return arr1


def copy_and_padding(arr1, arr2, arr_dims):
    c1 = arr1.size(0)
    c2 = arr2.size(0)

    for i in range(c1):
        if arr_dims == 1:
            if i >= c2:
                arr1[i] = torch.zeros(1, dtype=torch.FloatTensor).cuda()
            else:
                arr1[i] = arr2[i]
        else:
            _, h, w = arr2.size()
            if i >= c2:
                arr1[i, :, :] = torch.zeros((h, w), dtype=torch.FloatTensor).cuda()
            else:
                arr1[i, :, :] = arr2[i, :, :]
    return arr1


def get_name_and_params(model, model_type="resnet"):
    names, params = [], []
    if model_type == "mlp":
        key_layer = "fc"
    elif model_type == "vgg":
        key_layer = "features"
    elif model_type == "resnet":
        key_layer = "conv"

    for n, p in model.named_parameters():
        if key_layer in n.lower():
            names.append(n)
            params.append(p)

    return names, params


# model 1 is a target model, model 2 is a pre-trained model
def transfer_from_hetero_model(model1, model2, model_type="resnet", mapping=None, mapping_for=1):
    if isinstance(model1, nn.DataParallel):
        model1 = model1.module
    if isinstance(model2, nn.DataParallel):
        model2 = model2.module

    name1, model1_params = get_name_and_params(model1, model_type=model_type)
    name2, model2_params = get_name_and_params(model2, model_type=model_type)

    logger.info(f"name1: {name1}")
    logger.info(f"name2: {name2}")
    if mapping is not None:
        # indexes = [3,1,0,7,4,6,2]          # for random permutation
        # indexes = [0,2,4,6]            # for continuous verification
        indexes = mapping
    else:
        indexes = list(range(len(model1_params)))
        # random.shuffle(indexes)
    logger.info(f"Indexes: {indexes}")
    if mapping_for == 1:
        model1_params = [model1_params[i] for i in indexes]
    else:
        model2_params = [model2_params[i] for i in indexes]

    transferable_layer_num = min((len(model1_params), len(model2_params)))

    # transfer
    for i in tqdm(range(transferable_layer_num)):
        p1 = model1_params[i]
        p2 = model2_params[i]
        # logger.info(f'name1: {name1[i]}')
        # logger.info(f'name2: {name2[i]}')
        # logger.info(f'p1 size: {p1.size()}')
        # logger.info(f'p2 size: {p2.size()}')

        if len(p1.data.size()) == 1 and len(p2.data.size()) == 1:
            c_out1 = p1.size()[0]
            c_out2 = p2.size()[0]
            p1.data = circle_copy(p1.data, p2.data, 1)
        elif len(p1.data.size()) == 2 and len(p2.data.size()) == 2:
            c_out1 = p1.size()[0]
            c_out2 = p2.size()[0]

            # cyclic stack
            for j in range(c_out1):
                k = j % c_out2
                p1.data[j, :] = circle_copy(p1.data[j, :], p2.data[k, :], 1)
        else:
            c_out1, c_in1, k_h1, c_w1 = p1.size()
            c_out2, c_in2, k_h2, c_w2 = p2.size()
            # check square filter
            assert k_h1 == c_w1
            assert k_h2 == c_w2

            # step 1: filter interpolation
            if k_h1 != k_h2:
                # mode={nearest, linear, bilinear}, align_corners=True/False
                p2_resized = F.interpolate(p2.data, size=[k_h1, c_w1], mode="bilinear")
            else:
                p2_resized = p2.data

            # step 2: cyclic stack
            for j in range(c_out1):
                k = j % c_out2
                p1.data[j, :, :, :] = circle_copy(p1.data[j, :, :, :], p2_resized[k, :, :, :], 3)

        # p1.data = p1.data*torch.numel(p2.data)/torch.numel(p1.data)
        # logger.info(f'after p1 size: {p1.size()}')
    return model1
