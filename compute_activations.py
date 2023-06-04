import os
import sys
from basic_config import PATH_TO_CIFAR
sys.path.append(PATH_TO_CIFAR)

import torch
import utils as myutils
from log import logger
from tqdm import tqdm


ensemble_root_dir = "./cifar_models/"
ensemble_experiment = "exp_2019-08-24_02-20-26"
ensemble_dir = ensemble_root_dir + ensemble_experiment

activation_root_dir = "./activations/"
checkpoint_type = "final"  # which checkpoint to use for ensembling (either of 'best' or 'final)


def save_activations(idx, activation, dump_path):
    myutils.mkdir(dump_path)
    myutils.pickle_obj(activation, os.path.join(dump_path, "model_{}_activations".format(idx)))


def compute_activations_across_models(args, models, train_loader, num_samples, dump_activations=False, dump_path=None):
    # hook that computes the mean activations across data samples
    def get_activation(activation, name):
        def hook(model, input, output):
            if name not in activation:
                activation[name] = output.detach()
            else:
                activation[name] = (num_samples_processed * activation[name] + output.detach()) / (
                    num_samples_processed + 1
                )

        return hook

    # Prepare all the models
    activations = {}

    for idx, model in enumerate(models):
        # Initialize the activation dictionary for each model
        activations[idx] = {}

        # Set forward hooks for all layers inside a model
        for name, layer in model.named_modules():
            if name == "":
                logger.info("excluded")
                continue
            layer.register_forward_hook(get_activation(activations[idx], name))
            logger.info("set forward hook for layer named: %s", name)

        # Set the model in train mode
        model.train()

    # Run the same data samples ('num_samples' many) across all the models
    num_samples_processed = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.gpu_id != -1:
            data = data.cuda(args.gpu_id)
        for idx, model in enumerate(models):
            model(data)
        num_samples_processed += 1
        if num_samples_processed == num_samples:
            break

    # Dump the activations for all models onto disk
    if dump_activations and dump_path is not None:
        for idx in range(len(models)):
            save_activations(idx, activations[idx], dump_path)

    return activations


def normalize_tensor(tens):
    tens_shape = tens.shape
    assert tens_shape[1] == 1
    tens = tens.view(tens_shape[0], 1, -1)
    norms = tens.norm(dim=-1)
    norms.clamp_(min=1e-3)
    ntens = tens / norms.view(-1, 1, 1)
    ntens = ntens.view(tens_shape)
    return ntens


def compute_activations_across_models_v1(
    args, models, train_loader, num_samples, mode="mean", dump_activations=False, dump_path=None
):
    torch.manual_seed(args.activation_seed)
    is_audio = args.dataset == "esc50"

    # hook that computes the mean activations across data samples
    def get_activation(activation, name, is_audio=False):
        def hook(model, input, output):
            if name not in activation:
                activation[name] = []
            if is_audio and len(output.size()) == 4:
                # output size: batch size x n_channels x n_mels x t
                activation[name].append(output.mean(3).detach())
            else:
                activation[name].append(output.detach())

        return hook

    # Prepare all the models
    activations = {}
    forward_hooks = []

    # assert args.disable_bias
    # handle below for bias later on!
    param_names = [
        [tupl[0].replace(".weight", "") for tupl in models[i].named_parameters()] for i in range(len(models))
    ]
    for idx, model in enumerate(models):
        # Initialize the activation dictionary for each model
        activations[idx] = {}
        layer_hooks = []
        # Set forward hooks for all layers inside a model
        for name, layer in model.named_modules():
            if (name == "") or ("bias" in name) or ("bn" in name) or ("shortcut.1" in name) or ("BatchNorm2d" in name):
                logger.info(f"excluded {name}")
                continue
            elif args.dataset != "mnist" and name not in param_names[idx]:
                logger.info(f"this was continued, {name}")
                continue
            layer_hooks.append(layer.register_forward_hook(get_activation(activations[idx], name, is_audio)))
            logger.info(f"set forward hook for layer named: {name}")

        forward_hooks.append(layer_hooks)
        # Set the model in train mode
        model.train()

    # Run the same data samples ('num_samples' many) across all the models
    num_samples_processed = 0
    num_personal_idx = 0
    for _, (data, target) in tqdm(enumerate(train_loader)):
        if num_samples_processed == num_samples:
            break
        if args.gpu_id != -1:
            data = data.cuda(args.gpu_id)

        if args.skip_personal_idx and int(target.item()) == args.personal_class_idx:
            continue

        if int(target.item()) == args.personal_class_idx:
            num_personal_idx += 1

        for idx, model in enumerate(models):
            model(data)

        num_samples_processed += 1

    logger.info("num_personal_idx %d", num_personal_idx)
    setattr(args, "num_personal_idx", num_personal_idx)

    relu = torch.nn.ReLU()
    maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    avgpool = torch.nn.AvgPool2d(kernel_size=1, stride=1)

    # Combine the activations generated across the number of samples to form importance scores
    # The importance calculated is based on the 'mode' flag: which is either of 'mean', 'std', 'meanstd'

    model_cfg = myutils.get_model_layers_cfg(args.model_name)
    for idx in range(len(models)):
        cfg_idx = 0
        for lnum, layer in enumerate(activations[idx]):
            # logger.info('***********')
            activations[idx][layer] = torch.stack(activations[idx][layer])
            # logger.info("min of act: {}, max: {}, mean: {}".format(
            #     torch.min(activations[idx][layer]),
            #     torch.max(activations[idx][layer]),
            #     torch.mean(activations[idx][layer])))
            # assert (activations[idx][layer] >= 0).all()

            if not args.prelu_acts and not lnum == (len(activations[idx]) - 1):
                # logger.info("applying relu ---------------")
                activations[idx][layer] = relu(activations[idx][layer])
                # logger.info("after RELU: min of act: {}, max: {}, mean: {}".format(
                #     torch.min(activations[idx][layer]),
                #     torch.max(activations[idx][layer]),
                #     torch.mean(activations[idx][layer])))

            elif args.model_name == "vgg11_nobias" and args.pool_acts and len(activations[idx][layer].shape) > 3:
                if args.pool_relu:
                    # logger.info("applying relu ---------------")
                    activations[idx][layer] = relu(activations[idx][layer])

                activations[idx][layer] = activations[idx][layer].squeeze(1)

                # apply maxpool wherever the next thing in config list is 'M'
                if (cfg_idx + 1) < len(model_cfg):
                    if model_cfg[cfg_idx + 1] == "M":
                        # logger.info("applying maxpool ---------------")
                        activations[idx][layer] = maxpool(activations[idx][layer])
                        cfg_idx += 2
                    else:
                        cfg_idx += 1

                # apply avgpool only for the last layer
                if cfg_idx == len(model_cfg):
                    # logger.info("applying avgpool ---------------")
                    activations[idx][layer] = avgpool(activations[idx][layer])

                # unsqueeze back at axis 1
                activations[idx][layer] = activations[idx][layer].unsqueeze(1)

                # logger.info("checking stats after pooling")
                # logger.info("min of act: {}, max: {}, mean: {}".format(
                #     torch.min(activations[idx][layer]),
                #     torch.max(activations[idx][layer]),
                #     torch.mean(activations[idx][layer])))

            if mode == "mean":
                activations[idx][layer] = activations[idx][layer].mean(dim=0)
            elif mode == "std":
                activations[idx][layer] = activations[idx][layer].std(dim=0)
            elif mode == "meanstd":
                activations[idx][layer] = activations[idx][layer].mean(dim=0) * activations[idx][layer].std(dim=0)

            if args.standardize_acts:
                mean_acts = activations[idx][layer].mean(dim=0)
                std_acts = activations[idx][layer].std(dim=0)
                # logger.info("shape of mean, std, and usual acts are: {} {} {}".format(
                #     mean_acts.shape,
                #     std_acts.shape,
                #     activations[idx][layer].shape))
                activations[idx][layer] = (activations[idx][layer] - mean_acts) / (std_acts + 1e-9)
            elif args.center_acts:
                mean_acts = activations[idx][layer].mean(dim=0)
                # logger.info("shape of mean and usual acts are: {} {}".format(
                #     mean_acts.shape,
                #     activations[idx][layer].shape))
                activations[idx][layer] = activations[idx][layer] - mean_acts
            elif args.normalize_acts:
                # logger.info("normalizing the activation vectors")
                activations[idx][layer] = normalize_tensor(activations[idx][layer])
                # logger.info("min of act: {}, max: {}, mean: {}".format(
                #     torch.min(activations[idx][layer]),
                #     torch.max(activations[idx][layer]),
                #     torch.mean(activations[idx][layer])))

            # logger.info("activations for idx {} at layer {} have the following shape {}".format(
            #     idx,
            #     layer,
            #     activations[idx][layer].shape))
            # logger.info('-----------')

    # Dump the activations for all models onto disk
    if dump_activations and dump_path is not None:
        for idx in range(len(models)):
            save_activations(idx, activations[idx], dump_path)

    # Remove the hooks (as this was intefering with prediction ensembling)
    for idx in range(len(forward_hooks)):
        for hook in forward_hooks[idx]:
            hook.remove()

    return activations


def compute_selective_activation(
    args, models, layer_name, train_loader, num_samples, dump_activations=False, dump_path=None
):
    torch.manual_seed(args.activation_seed)

    # hook that computes the mean activations across data samples
    def get_activation(activation, name):
        def hook(model, input, output):
            if name not in activation:
                activation[name] = []

            activation[name].append(output.detach())

        return hook

    # Prepare all the models
    activations = {}
    forward_hooks = []

    # assert args.disable_bias
    # handle below for bias later on!
    param_names = [
        [tupl[0].replace(".weight", "") for tupl in models[i].named_parameters()] for i in range(len(models))
    ]

    for idx, model in enumerate(models):
        # Initialize the activation dictionary for each model
        activations[idx] = {}
        layer_hooks = []
        # Set forward hooks for all layers inside a model
        for name, layer in model.named_modules():
            if (name == "") or ("bias" in name):
                logger.info("excluded")
            elif args.dataset != "mnist" and name not in param_names[idx]:
                logger.info("this was continued, %s", name)
            else:
                layer_hooks.append(layer.register_forward_hook(get_activation(activations[idx], name)))
                logger.info("set forward hook for layer named: %s", name)

        forward_hooks.append(layer_hooks)
        # Set the model in train mode
        model.train()

    # Run the same data samples ('num_samples' many) across all the models
    num_samples_processed = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if num_samples_processed == num_samples:
            break
        if args.gpu_id != -1:
            data = data.cuda(args.gpu_id)
        for idx, model in enumerate(models):
            model(data)
        num_samples_processed += 1

    relu = torch.nn.ReLU()
    for idx in range(len(models)):
        for lnum, layer in enumerate(activations[idx]):
            logger.info("***********")
            activations[idx][layer] = torch.stack(activations[idx][layer])
            logger.info(
                "min of act: {}, max: {}, mean: {}".format(
                    torch.min(activations[idx][layer]),
                    torch.max(activations[idx][layer]),
                    torch.mean(activations[idx][layer]),
                )
            )
            # assert (activations[idx][layer] >= 0).all()

            if not args.prelu_acts and not lnum == (len(activations[idx]) - 1):
                logger.info("applying relu ---------------")
                activations[idx][layer] = relu(activations[idx][layer])
                logger.info(
                    "after RELU: min of act: {}, max: {}, mean: {}".format(
                        torch.min(activations[idx][layer]),
                        torch.max(activations[idx][layer]),
                        torch.mean(activations[idx][layer]),
                    )
                )
            if args.standardize_acts:
                mean_acts = activations[idx][layer].mean(dim=0)
                std_acts = activations[idx][layer].std(dim=0)
                logger.info(
                    "shape of mean, std, and usual acts are: {} {} {}".format(
                        mean_acts.shape, std_acts.shape, activations[idx][layer].shape
                    )
                )
                activations[idx][layer] = (activations[idx][layer] - mean_acts) / (std_acts + 1e-9)
            elif args.center_acts:
                mean_acts = activations[idx][layer].mean(dim=0)
                logger.info(
                    "shape of mean and usual acts are: {} {}".format(mean_acts.shape, activations[idx][layer].shape)
                )
                activations[idx][layer] = activations[idx][layer] - mean_acts

            logger.info(
                "activations for idx {} at layer {} have the following shape {}".format(
                    idx, layer, activations[idx][layer].shape
                )
            )
            logger.info("-----------")
    # Dump the activations for all models onto disk
    if dump_activations and dump_path is not None:
        for idx in range(len(models)):
            save_activations(idx, activations[idx], dump_path)

    # Remove the hooks (as this was intefering with prediction ensembling)
    for idx in range(len(forward_hooks)):
        for hook in forward_hooks[idx]:
            hook.remove()

    return activations
