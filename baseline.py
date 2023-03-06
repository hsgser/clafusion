import copy

import routines
import torch
import torch.nn.functional as F
import utils
import wasserstein_ensemble
from align_layers import add_layers_into_smaller_network, transfer_to_align_map
from log import logger
from model import get_model_from_name
from model_transfer import transfer_from_hetero_model
from tqdm import tqdm


def get_avg_parameters(networks, weights=None):
    logger.info("Average parameters!")
    avg_pars = []
    for par_group in zip(*[net.parameters() for net in networks]):
        # logger.info([par.shape for par in par_group])
        if weights is not None:
            weighted_par_group = [par * weights[i] for i, par in enumerate(par_group)]
            avg_par = torch.sum(torch.stack(weighted_par_group), dim=0)
        else:
            avg_par = torch.mean(torch.stack(par_group), dim=0)
        # logger.info(avg_par.shape)
        avg_pars.append(avg_par)
    return avg_pars


def naive_ensembling(args, networks, test_loader, network_idx=0):
    # simply average the weights in networks
    if args.width_ratio != 1:
        logger.info("Unfortunately naive ensembling can't work if models are not of same shape!")
        return -1, None
    if args.num_models == 2:
        weights = [(1 - args.ensemble_step), args.ensemble_step]
    else:
        weights = [1 / args.num_models for i in range(args.num_models)]
    avg_pars = get_avg_parameters(networks, weights)
    ensemble_network = get_model_from_name(args, idx=network_idx)
    # put on GPU
    if args.gpu_id != -1:
        ensemble_network = ensemble_network.cuda(args.gpu_id)

    # check the test performance of the method before
    log_dict = {}
    log_dict["test_losses"] = []
    routines.test(args, ensemble_network, test_loader, log_dict)

    # set the weights of the ensembled network
    for idx, (name, param) in enumerate(ensemble_network.state_dict().items()):
        ensemble_network.state_dict()[name].copy_(avg_pars[idx].data)

    # check the test performance of the method after ensembling
    log_dict = {}
    log_dict["test_losses"] = []
    acc = routines.test(args, ensemble_network, test_loader, log_dict)
    if args.dump_final_models:
        routines.save_final_model(args, "OT_merge", ensemble_network, acc)

    return acc, ensemble_network


def prediction_ensembling(args, networks, test_loader):
    log_dict = {}
    log_dict["test_losses"] = []

    if args.dataset.lower() != "mnist":
        criterion = torch.nn.CrossEntropyLoss()

    # set all the networks in eval mode
    for net in networks:
        net.eval()
    test_loss = 0
    correct = 0

    if (len(networks) == 2) and (args.prediction_wts):
        wts = [(1 - args.ensemble_step), args.ensemble_step]
    else:
        wts = [1 / len(networks) for _ in range(len(networks))]

    for data, target in tqdm(test_loader):
        if args.gpu_id != -1:
            data = data.cuda(args.gpu_id)
            target = target.cuda(args.gpu_id)
        outputs = []
        # average the outputs of all nets
        for idx, net in enumerate(networks):
            outputs.append(wts[idx] * net(data))

        output = torch.sum(torch.stack(outputs), dim=0)

        #  check loss of this ensembled prediction
        if args.dataset.lower() == "mnist":
            test_loss += F.nll_loss(output, target, reduction="sum").item()
        else:
            # mnist models return log_softmax outputs, while cifar ones return raw values!
            test_loss += criterion(output, target).item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    log_dict["test_losses"].append(test_loss)

    logger.info(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

    return (float(correct) * 100.0) / len(test_loader.dataset)


def transfer_networks(args, networks, test_loader, model_type="resnet", keep_weights=True, mapping=None):
    """
    keep_weights: If True, keep weights of the larger pretrained network after
        transfering. If False, randomly initialize the weights that are not transfered.
    """
    transfer_accuracies = []
    transfer_models = []

    if keep_weights:
        for i in range(1):
            log_dict = {}
            log_dict["test_losses"] = []
            transfer_models.append(
                transfer_from_hetero_model(
                    copy.deepcopy(networks[i]),
                    networks[1 - i],
                    model_type=model_type,
                    mapping=mapping,
                    mapping_for=1 + i,
                )
            )
            transfer_accuracies.append(routines.test(args, transfer_models[i], test_loader, log_dict))
    else:
        log_dict = {}
        log_dict["test_losses"] = []
        random_net = get_model_from_name(args, idx=0)
        transfer_models.append(
            transfer_from_hetero_model(random_net, networks[1], model_type=model_type, mapping=mapping)
        )
        transfer_accuracies.append(routines.test(args, transfer_models[0], test_loader, log_dict))

    return transfer_accuracies, transfer_models


def transfer_networks_and_naive_ensembling(args, networks, test_loader, model_type="resnet", mapping=None):
    transfer_accuracies = []
    transfer_models = []

    for i in range(1):
        log_dict = {}
        log_dict["test_losses"] = []
        trans_model = transfer_from_hetero_model(
            copy.deepcopy(networks[i]), networks[1 - i], model_type=model_type, mapping=mapping, mapping_for=1 + i
        )
        trans_acc, trans_model = naive_ensembling(args, [trans_model, networks[i]], test_loader, network_idx=i)
        transfer_models.append(trans_model)
        transfer_accuracies.append(trans_acc)

    return transfer_accuracies, transfer_models


def transfer_networks_and_otfusion(args, networks, train_loader, test_loader, model_type="resnet", mapping=None):
    transfer_accuracies = []
    transfer_models = []

    for i in range(1):
        log_dict = {}
        log_dict["test_losses"] = []
        trans_model = transfer_from_hetero_model(
            copy.deepcopy(networks[i]), networks[1 - i], model_type=model_type, mapping=mapping, mapping_for=1 + i
        )
        activations = utils.get_model_activations(args, [trans_model, networks[i]], config=args.config)
        trans_acc, trans_model = wasserstein_ensemble.geometric_ensembling_modularized(
            args, [trans_model, networks[i]], train_loader, test_loader, activations, idx=i
        )
        transfer_models.append(trans_model)
        transfer_accuracies.append(trans_acc)

    return transfer_accuracies, transfer_models


def transfer_networks_and_add_layers(
    args, networks, train_loader, test_loader, num_layers, model_names, model_type="resnet", mapping=None
):
    """
    networks[0] is larger than networks[1]
    """
    transfer_accuracies = []
    transfer_models = []
    if mapping is not None:
        if model_type == "mlp":
            mapping = mapping[: num_layers[1]]
        else:
            mapping = mapping[: num_layers[1] - 1]

    trans_model = transfer_from_hetero_model(
        copy.deepcopy(networks[1]), networks[0], model_type=model_type, mapping=mapping, mapping_for=2
    )
    log_dict = {}
    log_dict["test_losses"] = []
    routines.test(args, trans_model, test_loader, log_dict)

    if mapping is None:
        mapping = list(range(num_layers[1] - 1))
    else:
        mapping = transfer_to_align_map(mapping, model_type=model_type)

    if model_type != "mlp":
        # last linear classifier layers are aligned to each other
        mapping.append(num_layers[0])

    logger.info(f"Alignment map is {mapping}")
    new_weights, args, model_config = add_layers_into_smaller_network(
        args, networks[0], trans_model, num_layers[0], num_layers[1], mapping, model_names
    )

    trans_acc, trans_model = wasserstein_ensemble.get_network_from_param_list(
        args, new_weights, test_loader, model_config=model_config
    )
    transfer_models.append(trans_model)
    transfer_accuracies.append(trans_acc)

    return transfer_accuracies, transfer_models
