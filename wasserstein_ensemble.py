import math

import numpy as np
import ot
import routines
import torch
import utils
from ground_metric import GroundMetric
from log import logger
from model import get_model_from_name


def cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    return c


def get_histogram(args, idx, cardinality, layer_name, activations=None, return_numpy=True, float64=False):
    if activations is None:
        # returns a uniform measure
        if not args.unbalanced:
            logger.info("returns a uniform measure of cardinality: %d", cardinality)
            return np.ones(cardinality) / cardinality
        else:
            return np.ones(cardinality)
    else:
        # return softmax over the activations raised to a temperature
        # layer_name is like 'fc1.weight', while activations only contains 'fc1'
        logger.info(activations[idx].keys())
        unnormalized_weights = activations[idx][layer_name.split(".")[0]]
        logger.info(
            "For layer {},  shape of unnormalized weights is {}".format(layer_name, unnormalized_weights.shape)
        )
        unnormalized_weights = unnormalized_weights.squeeze()
        assert unnormalized_weights.shape[0] == cardinality

        if return_numpy:
            if float64:
                return (
                    torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0)
                    .data.cpu()
                    .numpy()
                    .astype(np.float64)
                )
            else:
                return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0).data.cpu().numpy()
        else:
            return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0)


def get_wassersteinized_layers_modularized(
    args, networks, activations=None, eps=1e-7, test_loader=None, not_avg=False
):
    """
    Two neural networks that have to be averaged in geometric manner (i.e. layerwise).
    The 1st network is aligned with respect to the other via wasserstein distance.
    Also this assumes that all the layers are either fully connected or convolutional *(with no bias)*

    :param networks: list of networks
    :param activations: If not None, use it to build the activation histograms.
    Otherwise assumes uniform distribution over neurons in a layer.
    :return: list of layer weights 'wassersteinized'
    """
    avg_aligned_layers = []
    # cumulative_T_var = None
    T_var = None
    if args.handle_skips:
        skip_T_var = None
        skip_T_var_idx = -1
        residual_T_var = None
        residual_T_var_idx = -1
    previous_layer_shape = None
    ground_metric_object = GroundMetric(args)

    if args.eval_aligned:
        model0_aligned_layers = []

    if args.gpu_id == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(args.gpu_id))

    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    networks_named_params = list(zip(networks[0].named_parameters(), networks[1].named_parameters()))
    idx = 0
    while idx < num_layers:
        logger.info("\n--------------- At layer index {} ------------- \n ".format(idx))
        ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) = networks_named_params[idx]
        # check input dim in layer 0
        # for other layers, aligned_wt and fc_layer1_weight
        # has the same number of columns
        if idx == 0:
            assert fc_layer0_weight.shape[1] == fc_layer1_weight.shape[1]
        logger.info("Previous layer shape is {}".format(previous_layer_shape))
        previous_layer_shape = fc_layer1_weight.shape
        mu_cardinality = int(fc_layer0_weight.shape[0])
        nu_cardinality = int(fc_layer1_weight.shape[0])

        layer_shape = fc_layer0_weight.shape
        if len(layer_shape) > 2:
            is_conv = True
            # For convolutional layers, it is (#out_channels, #in_channels, height, width)
            fc_layer0_weight_data = fc_layer0_weight.data.view(
                fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1
            )
            fc_layer1_weight_data = fc_layer1_weight.data.view(
                fc_layer1_weight.shape[0], fc_layer1_weight.shape[1], -1
            )
        else:
            is_conv = False
            fc_layer0_weight_data = fc_layer0_weight.data
            fc_layer1_weight_data = fc_layer1_weight.data

        if idx == 0:
            if is_conv:
                M = ground_metric_object.process(
                    fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
                    fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1),
                )
            else:
                M = ground_metric_object.process(fc_layer0_weight_data, fc_layer1_weight_data)

            aligned_wt = fc_layer0_weight_data
        elif len(layer_shape) == 1:
            logger.info("shape of layer: model 0 {}".format(fc_layer0_weight_data.shape))
            logger.info("shape of layer: model 1 {}".format(fc_layer1_weight_data.shape))
            logger.info("shape of previous transport map {}".format(T_var.shape))
            # to handle bias and batch norm weights
            # simply use the previous transportation map
            # to align weights and then take average
            aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)
            if args.ensemble_step != 0.5:
                logger.info("taking baby steps! ")
                geometric_fc = (1 - args.ensemble_step) * aligned_wt + args.ensemble_step * fc_layer1_weight_data
            else:
                geometric_fc = (aligned_wt + fc_layer1_weight_data) / 2
            avg_aligned_layers.append(geometric_fc)
            idx += 1
            fc_layer0_weight_data = None
            fc_layer1_weight_data = None
            continue
        else:
            logger.info("shape of layer: model 0 {}".format(fc_layer0_weight_data.shape))
            logger.info("shape of layer: model 1 {}".format(fc_layer1_weight_data.shape))
            logger.info("shape of previous transport map {}".format(T_var.shape))

            # aligned_wt = None, this caches the tensor and causes OOM
            if is_conv:
                if args.handle_skips:
                    assert len(layer_shape) == 4
                    # save skip_level transport map if there is block ahead
                    if layer_shape[1] != layer_shape[0]:
                        if not (layer_shape[2] == 1 and layer_shape[3] == 1):
                            logger.info(f"saved skip T_var at layer {idx} with shape {layer_shape}")
                            skip_T_var = T_var.clone()
                            skip_T_var_idx = idx
                        else:
                            logger.info(
                                f"utilizing skip T_var saved from layer layer {skip_T_var_idx} with shape {skip_T_var.shape}"
                            )
                            # if it's a shortcut (128, 64, 1, 1)
                            residual_T_var = T_var.clone()
                            residual_T_var_idx = idx  # use this after the skip
                            T_var = skip_T_var
                        logger.info("shape of previous transport map now is {}".format(T_var.shape))
                    else:
                        if residual_T_var is not None and (residual_T_var_idx == (idx - 1)):
                            T_var = (T_var + residual_T_var) / 2
                            logger.info("averaging multiple T_var's")
                        else:
                            logger.info("doing nothing for skips")
                T_var_conv = T_var.unsqueeze(0).repeat(fc_layer0_weight_data.shape[2], 1, 1)
                aligned_wt = torch.bmm(fc_layer0_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)

                M = ground_metric_object.process(
                    aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                    fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1),
                )
            else:
                if fc_layer0_weight.data.shape[1] != T_var.shape[0]:
                    # Handles the switch from convolutional layers to fc layers
                    fc_layer0_unflattened = fc_layer0_weight.data.view(
                        fc_layer0_weight.shape[0], T_var.shape[0], -1
                    ).permute(2, 0, 1)
                    aligned_wt = torch.bmm(
                        fc_layer0_unflattened, T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
                    ).permute(1, 2, 0)
                    aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                else:
                    aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)

                M = ground_metric_object.process(aligned_wt, fc_layer1_weight)
                # logger.info("ground metric is {}".format(M))
            if args.skip_last_layer and idx == (num_layers - 1):
                logger.info("Simple averaging of last layer weights. NO transport map needs to be computed")
                if not_avg:
                    avg_aligned_layers.append(aligned_wt)
                else:
                    if args.ensemble_step != 0.5:
                        avg_aligned_layers.append(
                            (1 - args.ensemble_step) * aligned_wt + args.ensemble_step * fc_layer1_weight
                        )
                    else:
                        avg_aligned_layers.append((aligned_wt + fc_layer1_weight) / 2)
                return avg_aligned_layers

        if args.importance is None or (idx == num_layers - 1):
            mu = get_histogram(args, 0, mu_cardinality, layer0_name)
            nu = get_histogram(args, 1, nu_cardinality, layer1_name)
        else:
            # mu = _get_neuron_importance_histogram(args, aligned_wt, is_conv)
            mu = _get_neuron_importance_histogram(args, fc_layer0_weight_data, is_conv)
            nu = _get_neuron_importance_histogram(args, fc_layer1_weight_data, is_conv)
            logger.info("{}, {}".format(mu, nu))
            assert args.proper_marginals

        cpuM = M.data.cpu().numpy()
        if args.exact:
            T = ot.emd(mu, nu, cpuM)
        else:
            T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=args.reg)
        # T = ot.emd(mu, nu, log_cpuM)

        if args.gpu_id != -1:
            T_var = torch.from_numpy(T).cuda(args.gpu_id).float()
        else:
            T_var = torch.from_numpy(T).float()

        # logger.info("the transport map is {}".format(T_var))

        if args.correction:
            if not args.proper_marginals:
                # think of it as m x 1, scaling weights for m linear combinations of points in X
                if args.gpu_id != -1:
                    # marginals = torch.mv(T_var.t(), torch.ones(T_var.shape[0]).cuda(args.gpu_id))  # T.t().shape[1] = T.shape[0]
                    marginals = torch.ones(T_var.shape[1]).cuda(args.gpu_id) / T_var.shape[1]
                else:
                    # marginals = torch.mv(T_var.t(),
                    #                      torch.ones(T_var.shape[0]))  # T.t().shape[1] = T.shape[0]
                    marginals = torch.ones(T_var.shape[1]) / T_var.shape[1]
                marginals = torch.diag(1.0 / (marginals + eps))  # take inverse
                T_var = torch.matmul(T_var, marginals)
            else:
                # marginals_alpha = T_var @ torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)
                marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype).to(device)

                marginals = 1 / (marginals_beta + eps)
                logger.info("shape of inverse marginals beta is {}".format(marginals_beta.shape))
                logger.info("inverse marginals beta is {}".format(marginals_beta))

                T_var = T_var * marginals
                # i.e., how a neuron of 2nd model is constituted by the neurons of 1st model
                # this should all be ones, and number equal to number of neurons in 2nd model
                logger.info(T_var.sum(dim=0))
                # assert (T_var.sum(dim=0) == torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)).all()

        if args.debug:
            if idx == (num_layers - 1):
                logger.info("there goes the last transport map: \n {}".format(T_var))
            else:
                logger.info("there goes the transport map at layer {}: \n {}".format(idx, T_var))

            logger.info("Ratio of trace to the matrix sum: %f", torch.trace(T_var) / torch.sum(T_var))

        logger.info("Ratio of trace to the matrix sum: %f", torch.trace(T_var) / torch.sum(T_var))
        logger.info("Here, trace is {} and matrix sum is {} ".format(torch.trace(T_var), torch.sum(T_var)))
        setattr(args, "trace_sum_ratio_{}".format(layer0_name), (torch.trace(T_var) / torch.sum(T_var)).item())

        if args.past_correction:
            logger.info("this is past correction for weight mode")
            logger.info("Shape of aligned wt is {}".format(aligned_wt.shape))
            logger.info("Shape of fc_layer0_weight_data is {}".format(fc_layer0_weight_data.shape))
            t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
        else:
            t_fc0_model = torch.matmul(T_var.t(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1))

        # Average the weights of aligned first layers
        if not_avg:
            geometric_fc = t_fc0_model
        else:
            if args.ensemble_step != 0.5:
                geometric_fc = (
                    1 - args.ensemble_step
                ) * t_fc0_model + args.ensemble_step * fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
            else:
                geometric_fc = (t_fc0_model + fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)) / 2
        if is_conv and layer_shape != geometric_fc.shape:
            geometric_fc = geometric_fc.view(layer_shape)
        avg_aligned_layers.append(geometric_fc)

        # get the performance of the model 0 aligned with respect to the model 1
        if args.eval_aligned:
            if is_conv and layer_shape != t_fc0_model.shape:
                t_fc0_model = t_fc0_model.view(layer_shape)
            model0_aligned_layers.append(t_fc0_model)
            _, acc = update_model(args, networks[0], model0_aligned_layers, test=True, test_loader=test_loader, idx=0)
            logger.info("For layer idx {}, accuracy of the updated model is {}".format(idx, acc))
            setattr(args, "model0_aligned_acc_layer_{}".format(str(idx)), acc)
            if idx == (num_layers - 1):
                setattr(args, "model0_aligned_acc", acc)

        mu = None
        nu = None
        fc_layer0_weight_data = None
        fc_layer1_weight_data = None
        M = None
        idx += 1

    return avg_aligned_layers


def print_stats(arr, nick=""):
    logger.info(nick)
    logger.info(
        "summary stats are: \n max: {}, mean: {}, min: {}, median: {}, std: {} \n".format(
            arr.max(), arr.mean(), arr.min(), np.median(arr), arr.std()
        )
    )


def get_activation_distance_stats(activations_0, activations_1, layer_name=""):
    if layer_name != "":
        logger.info("In layer {}: getting activation distance statistics".format(layer_name))
    # logger.info("Activation 0 shape {}".format(activations_0.shape))
    # logger.info("Activation 1 shape {}".format(activations_1.shape))
    M = cost_matrix(activations_0, activations_1) ** (1 / 2)
    mean_dists = torch.mean(M, dim=-1)
    max_dists = torch.max(M, dim=-1)[0]
    min_dists = torch.min(M, dim=-1)[0]
    std_dists = torch.std(M, dim=-1)

    logger.info("Statistics of the distance from neurons of layer 1 (averaged across nodes of layer 0): \n")
    logger.info(
        "Max : {}, Mean : {}, Min : {}, Std: {}".format(
            torch.mean(max_dists), torch.mean(mean_dists), torch.mean(min_dists), torch.mean(std_dists)
        )
    )


def update_model(args, model, new_params, test=False, test_loader=None, reversed=False, idx=-1):
    updated_model = get_model_from_name(args, idx=idx)
    if args.gpu_id != -1:
        updated_model = updated_model.cuda(args.gpu_id)

    layer_idx = 0
    model_state_dict = model.state_dict()

    logger.info("len of model_state_dict is %d", len(model_state_dict.items()))
    logger.info("len of new_params is %d", len(new_params))

    for key, value in model_state_dict.items():
        logger.info("updated parameters for layer %s", key)
        model_state_dict[key] = new_params[layer_idx]
        layer_idx += 1
        if layer_idx == len(new_params):
            break

    updated_model.load_state_dict(model_state_dict)

    if test:
        log_dict = {}
        log_dict["test_losses"] = []
        final_acc = routines.test(args, updated_model, test_loader, log_dict)
        logger.info("accuracy after update is %f", final_acc)
    else:
        final_acc = None

    return updated_model, final_acc


def _check_activation_sizes(args, acts0, acts1):
    if args.width_ratio == 1:
        return acts0.shape == acts1.shape
    else:
        return acts0.shape[-1] / acts1.shape[-1] == args.width_ratio


def process_activations(args, activations, layer0_name, layer1_name):
    try:
        activations_0 = activations[0][layer0_name.replace("." + layer0_name.split(".")[-1], "")].squeeze(1)
        activations_1 = activations[1][layer1_name.replace("." + layer1_name.split(".")[-1], "")].squeeze(1)
    except KeyError:
        return None, None

    # assert activations_0.shape == activations_1.shape
    _check_activation_sizes(args, activations_0, activations_1)

    if args.same_model != -1:
        # sanity check when averaging the same model (with value being the model index)
        assert (activations_0 == activations_1).all()
        logger.info("Are the activations the same? {}".format((activations_0 == activations_1).all()))

    if len(activations_0.shape) == 2:
        activations_0 = activations_0.t()
        activations_1 = activations_1.t()
    elif len(activations_0.shape) > 2:
        reorder_dim = [l for l in range(1, len(activations_0.shape))]
        reorder_dim.append(0)
        logger.info("reorder_dim is {}".format(reorder_dim))
        activations_0 = activations_0.permute(*reorder_dim).contiguous()
        activations_1 = activations_1.permute(*reorder_dim).contiguous()

    return activations_0, activations_1


def _reduce_layer_name(layer_name):
    # previous way assumed only one dot, so now I replace the stuff after last dot
    return layer_name.replace("." + layer_name.split(".")[-1], "")


def _get_layer_weights(layer_weight, is_conv):
    if is_conv:
        # For convolutional layers, it is (#out_channels, #in_channels, height, width)
        layer_weight_data = layer_weight.data.view(layer_weight.shape[0], layer_weight.shape[1], -1)
    else:
        layer_weight_data = layer_weight.data

    return layer_weight_data


def _process_ground_metric_from_acts(args, is_conv, ground_metric_object, activations):
    logger.info("inside refactored")
    if is_conv:
        if not args.gromov:
            M0 = ground_metric_object.process(
                activations[0].view(activations[0].shape[0], -1), activations[1].view(activations[1].shape[0], -1)
            )
        else:
            M0 = ground_metric_object.process(
                activations[0].view(activations[0].shape[0], -1), activations[0].view(activations[0].shape[0], -1)
            )
            M1 = ground_metric_object.process(
                activations[1].view(activations[1].shape[0], -1), activations[1].view(activations[1].shape[0], -1)
            )

        logger.info("# of ground metric features is %d", (activations[0].view(activations[0].shape[0], -1)).shape[1])
    else:
        if not args.gromov:
            M0 = ground_metric_object.process(activations[0], activations[1])
        else:
            M0 = ground_metric_object.process(activations[0], activations[0])
            M1 = ground_metric_object.process(activations[1], activations[1])

    if args.gromov:
        return M0, M1
    else:
        return M0, None


def _custom_sinkhorn(args, mu, nu, cpuM):
    if not args.unbalanced:
        if args.sinkhorn_type == "normal":
            T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=args.reg)
        elif args.sinkhorn_type == "stabilized":
            T = ot.bregman.sinkhorn_stabilized(mu, nu, cpuM, reg=args.reg)
        elif args.sinkhorn_type == "epsilon":
            T = ot.bregman.sinkhorn_epsilon_scaling(mu, nu, cpuM, reg=args.reg)
        elif args.sinkhorn_type == "gpu":
            T, _ = utils.sinkhorn_loss(cpuM, mu, nu, gpu_id=args.gpu_id, epsilon=args.reg, return_tmap=True)
        else:
            raise NotImplementedError
    else:
        T = ot.unbalanced.sinkhorn_knopp_unbalanced(mu, nu, cpuM, reg=args.reg, reg_m=args.reg_m)
    return T


def _sanity_check_tmap(T):
    if not math.isclose(np.sum(T), 1.0, abs_tol=1e-7):
        logger.info("Sum of transport map is {}".format(np.sum(T)))
        raise Exception("NAN inside Transport MAP. Most likely due to large ground metric values")


def _get_updated_acts_v0(args, layer_shape, aligned_wt, model0_aligned_layers, networks, test_loader, layer_names):
    """
    Return the updated activations of the 0th model with respect to the other one.

    :param args:
    :param layer_shape:
    :param aligned_wt:
    :param model0_aligned_layers:
    :param networks:
    :param test_loader:
    :param layer_names:
    :return:
    """
    if layer_shape != aligned_wt.shape:
        updated_aligned_wt = aligned_wt.view(layer_shape)
    else:
        updated_aligned_wt = aligned_wt

    updated_model0, _ = update_model(
        args, networks[0], model0_aligned_layers + [updated_aligned_wt], test=True, test_loader=test_loader, idx=0
    )
    updated_activations = utils.get_model_activations(
        args,
        [updated_model0, networks[1]],
        config=args.config,
        layer_name=_reduce_layer_name(layer_names[0]),
        selective=True,
    )

    updated_activations_0, updated_activations_1 = process_activations(
        args, updated_activations, layer_names[0], layer_names[1]
    )
    return updated_activations_0, updated_activations_1


def _check_layer_sizes(args, layer_idx, shape1, shape2, num_layers):
    if args.width_ratio == 1:
        return shape1 == shape2
    else:
        if args.dataset == "mnist":
            if layer_idx == 0:
                return shape1[-1] == shape2[-1] and (shape1[0] / shape2[0]) == args.width_ratio
            elif layer_idx == (num_layers - 1):
                return (shape1[-1] / shape2[-1]) == args.width_ratio and shape1[0] == shape2[0]
            else:
                ans = True
                for ix in range(len(shape1)):
                    ans = ans and shape1[ix] / shape2[ix] == args.width_ratio
                return ans
        elif args.dataset[0:7] == "Cifar10":
            assert args.second_model_name is not None
            if layer_idx == 0 or layer_idx == (num_layers - 1):
                return shape1 == shape2
            else:
                if (not args.reverse and layer_idx == (num_layers - 2)) or (args.reverse and layer_idx == 1):
                    return (shape1[1] / shape2[1]) == args.width_ratio
                else:
                    return (shape1[0] / shape2[0]) == args.width_ratio


def _compute_marginals(args, T_var, device, eps=1e-7):
    if args.correction:
        if not args.proper_marginals:
            # think of it as m x 1, scaling weights for m linear combinations of points in X
            marginals = torch.ones(T_var.shape[0], T_var.shape[0])
            if args.gpu_id != -1:
                marginals = marginals.cuda(args.gpu_id)

            marginals = torch.matmul(marginals, T_var)
            marginals = 1 / (marginals + eps)
            # logger.info("marginals are {}".format(marginals))

            T_var = T_var * marginals

        else:
            # marginals_alpha = T_var @ torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)
            marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype).to(device)

            marginals = 1 / (marginals_beta + eps)
            logger.info("shape of inverse marginals beta is {}".format(marginals_beta.shape))
            logger.info("inverse marginals beta is {}".format(marginals_beta))

            T_var = T_var * marginals
            # i.e., how a neuron of 2nd model is constituted by the neurons of 1st model
            # this should all be ones, and number equal to number of neurons in 2nd model
            logger.info(T_var.sum(dim=0))
            # assert (T_var.sum(dim=0) == torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)).all()

        # logger.info("T_var after correction {}".format(T_var))
        logger.info(
            "T_var stats: max {}, min {}, mean {}, std {} ".format(T_var.max(), T_var.min(), T_var.mean(), T_var.std())
        )
    else:
        marginals = None

    return T_var, marginals


def _get_current_layer_transport_map(args, mu, nu, M0, M1, idx, layer_shape, eps=1e-7, layer_name=None):
    if not args.gromov:
        cpuM = M0.data.cpu().numpy()
        if args.exact:
            T = ot.emd(mu, nu, cpuM)
        else:
            T = _custom_sinkhorn(args, mu, nu, cpuM)

        if args.print_distances:
            ot_cost = np.multiply(T, cpuM).sum()
            logger.info(f"At layer idx {idx} and shape {layer_shape}, the OT cost is %f", ot_cost)
            if layer_name is not None:
                setattr(args, f"{layer_name}_layer_{idx}_cost", ot_cost)
            else:
                setattr(args, f"layer_{idx}_cost", ot_cost)
    else:
        cpuM0 = M0.data.cpu().numpy()
        cpuM1 = M1.data.cpu().numpy()

        assert not args.exact
        T = ot.gromov.entropic_gromov_wasserstein(cpuM0, cpuM1, mu, nu, loss_fun=args.gromov_loss, epsilon=args.reg)

    if not args.unbalanced:
        _sanity_check_tmap(T)

    if args.gpu_id != -1:
        T_var = torch.from_numpy(T).cuda(args.gpu_id).float()
    else:
        T_var = torch.from_numpy(T).float()

    if args.tmap_stats:
        logger.info(
            "Tmap stats (before correction) \n: For layer {}, frobenius norm from the joe's transport map is {}".format(
                layer_name, torch.norm(T_var - torch.ones_like(T_var) / torch.numel(T_var), p="fro")
            )
        )

    logger.info("shape of T_var is {}".format(T_var.shape))
    # logger.info("T_var before correction {}".format(T_var))

    return T_var


def _get_neuron_importance_histogram(args, layer_weight, is_conv, eps=1e-9):
    logger.info("shape of layer_weight is {}".format(layer_weight.shape))
    if is_conv:
        layer = layer_weight.contiguous().view(layer_weight.shape[0], -1).cpu().numpy()
    else:
        layer = layer_weight.cpu().numpy()

    if args.importance == "l1":
        importance_hist = np.linalg.norm(layer, ord=1, axis=-1).astype(np.float64) + eps
    elif args.importance == "l2":
        importance_hist = np.linalg.norm(layer, ord=2, axis=-1).astype(np.float64) + eps
    else:
        raise NotImplementedError

    if not args.unbalanced:
        importance_hist = importance_hist / importance_hist.sum()
        logger.info("sum of importance hist is %f", importance_hist.sum())
    # assert importance_hist.sum() == 1.0
    return importance_hist


def get_acts_wassersteinized_layers_modularized(
    args, networks, activations, eps=1e-7, train_loader=None, test_loader=None, not_avg=False
):
    """
    Average based on the activation vector over data samples. Obtain the transport map,
    and then based on which align the nodes and average the weights!
    Like before: two neural networks that have to be averaged in geometric manner (i.e. layerwise).
    The 1st network is aligned with respect to the other via wasserstein distance.
    Also this assumes that all the layers are either fully connected or convolutional *(with no bias)*
    :param networks: list of networks
    :param activations: If not None, use it to build the activation histograms.
    Otherwise assumes uniform distribution over neurons in a layer.
    :return: list of layer weights 'wassersteinized'
    """
    avg_aligned_layers = []
    T_var = None
    if args.handle_skips:
        skip_T_var = None
        skip_T_var_idx = -1
        residual_T_var = None
        residual_T_var_idx = -1

    previous_layer_shape = None
    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    ground_metric_object = GroundMetric(args)

    if args.update_acts or args.eval_aligned:
        model0_aligned_layers = []

    if args.gpu_id == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(args.gpu_id))

    networks_named_params = list(zip(networks[0].named_parameters(), networks[1].named_parameters()))
    idx = 0
    incoming_layer_aligned = True  # for input
    while idx < num_layers:
        logger.info("\n--------------- At layer index {} ------------- \n ".format(idx))
        ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) = networks_named_params[idx]
        # layer shape is out x in
        # check layer sizes
        if args.check_layer_sizes:
            assert _check_layer_sizes(args, idx, fc_layer0_weight.shape, fc_layer1_weight.shape, num_layers)
        logger.info("Previous layer shape is {}".format(previous_layer_shape))
        previous_layer_shape = fc_layer1_weight.shape

        # will have shape layer_size x act_num_samples
        layer0_name_reduced = _reduce_layer_name(layer0_name)

        logger.info(
            "let's see the difference in layer names %s %s",
            layer0_name.replace("." + layer0_name.split(".")[-1], ""),
            layer0_name_reduced,
        )
        # for conv layer I need to make the act_num_samples dimension the last one, but it has the intermediate dimensions for
        # height and width of channels, so that won't work.
        # So convert (num_samples, layer_size, ht, wt) -> (layer_size, ht, wt, num_samples)

        activations_0, activations_1 = process_activations(args, activations, layer0_name, layer1_name)

        if activations_0 is not None:
            assert activations_0.shape[0] == fc_layer0_weight.shape[0]
            assert activations_1.shape[0] == fc_layer1_weight.shape[0]
            # get_activation_distance_stats(activations_0, activations_1, layer0_name)

        mu_cardinality = int(fc_layer0_weight.shape[0])
        nu_cardinality = int(fc_layer1_weight.shape[0])

        layer0_shape = fc_layer0_weight.shape
        layer_shape = fc_layer1_weight.shape
        if len(layer_shape) > 2:
            is_conv = True
        else:
            is_conv = False

        fc_layer0_weight_data = _get_layer_weights(fc_layer0_weight, is_conv)
        fc_layer1_weight_data = _get_layer_weights(fc_layer1_weight, is_conv)

        if idx == 0 or incoming_layer_aligned:
            aligned_wt = fc_layer0_weight_data
        elif len(layer_shape) == 1:
            logger.info("shape of layer: model 0 {}".format(fc_layer0_weight_data.shape))
            logger.info("shape of layer: model 1 {}".format(fc_layer1_weight_data.shape))
            logger.info("shape of previous transport map {}".format(T_var.shape))
            # to handle bias and batch norm weights
            # simply use the previous transportation map
            # to align weights and then take average
            aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)
            if args.ensemble_step != 0.5:
                logger.info("taking baby steps! ")
                geometric_fc = (1 - args.ensemble_step) * aligned_wt + args.ensemble_step * fc_layer1_weight_data
            else:
                geometric_fc = (aligned_wt + fc_layer1_weight_data) / 2
            avg_aligned_layers.append(geometric_fc)
            idx += 1
            activations_0 = None
            activations_1 = None
            fc_layer0_weight_data = None
            fc_layer1_weight_data = None
            continue
        else:
            logger.info("shape of layer: model 0 {}".format(fc_layer0_weight_data.shape))
            logger.info("shape of layer: model 1 {}".format(fc_layer1_weight_data.shape))
            logger.info("shape of activations: model 0 {}".format(activations_0.shape))
            logger.info("shape of activations: model 1 {}".format(activations_1.shape))
            logger.info("shape of previous transport map {}".format(T_var.shape))

            # aligned_wt = None, this caches the tensor and causes OOM
            if is_conv:
                if args.handle_skips:
                    assert len(layer0_shape) == 4
                    # save skip_level transport map if there is block ahead
                    if layer0_shape[1] != layer0_shape[0]:
                        if not (layer0_shape[2] == 1 and layer0_shape[3] == 1):
                            logger.info(f"saved skip T_var at layer {idx} with shape {layer0_shape}")
                            skip_T_var = T_var.clone()
                            skip_T_var_idx = idx
                        else:
                            logger.info(
                                f"utilizing skip T_var saved from layer layer {skip_T_var_idx} with shape {skip_T_var.shape}"
                            )
                            # if it's a shortcut (128, 64, 1, 1)
                            residual_T_var = T_var.clone()
                            residual_T_var_idx = idx  # use this after the skip
                            T_var = skip_T_var
                        logger.info("shape of previous transport map now is {}".format(T_var.shape))
                    else:
                        if residual_T_var is not None and (residual_T_var_idx == (idx - 1)):
                            T_var = (T_var + residual_T_var) / 2
                            logger.info("averaging multiple T_var's")
                        else:
                            logger.info("doing nothing for skips")
                T_var_conv = T_var.unsqueeze(0).repeat(fc_layer0_weight_data.shape[2], 1, 1)
                aligned_wt = torch.bmm(fc_layer0_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)
            else:
                if fc_layer0_weight.data.shape[1] != T_var.shape[0]:
                    # Handles the switch from convolutional layers to fc layers
                    # checks if the input has been reshaped
                    fc_layer0_unflattened = fc_layer0_weight.data.view(
                        fc_layer0_weight.shape[0], T_var.shape[0], -1
                    ).permute(2, 0, 1)
                    aligned_wt = torch.bmm(
                        fc_layer0_unflattened, T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
                    ).permute(1, 2, 0)
                    aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                else:
                    aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)
            # Refactored #
            if args.update_acts:
                assert args.second_model_name is None
                activations_0, activations_1 = _get_updated_acts_v0(
                    args,
                    layer_shape,
                    aligned_wt,
                    model0_aligned_layers,
                    networks,
                    test_loader,
                    [layer0_name, layer1_name],
                )

        if args.importance is None or (idx == num_layers - 1):
            mu = get_histogram(args, 0, mu_cardinality, layer0_name)
            nu = get_histogram(args, 1, nu_cardinality, layer1_name)
        else:
            # mu = _get_neuron_importance_histogram(args, aligned_wt, is_conv)
            mu = _get_neuron_importance_histogram(args, fc_layer0_weight_data, is_conv)
            nu = _get_neuron_importance_histogram(args, fc_layer1_weight_data, is_conv)
            logger.info(f"{mu}, {nu}")
            assert args.proper_marginals

        if args.act_bug:
            # bug from before (didn't change the activation part)
            # only for reproducing results from previous version
            M0 = ground_metric_object.process(
                aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1),
            )
        else:
            # debugged part
            logger.info("Refactored ground metric calc")
            M0, M1 = _process_ground_metric_from_acts(
                args, is_conv, ground_metric_object, [activations_0, activations_1]
            )

            logger.info(
                "# of ground metric features in 0 is %d", (activations_0.view(activations_0.shape[0], -1)).shape[1]
            )
            logger.info(
                "# of ground metric features in 1 is %d", (activations_1.view(activations_1.shape[0], -1)).shape[1]
            )

        if args.debug and not args.gromov:
            # bug from before (didn't change the activation part)
            M_old = ground_metric_object.process(
                aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1),
            )
            logger.info(
                "Frobenius norm of old (i.e. bug involving wts) and new are {} {}".format(
                    torch.norm(M_old, "fro"), torch.norm(M0, "fro")
                )
            )
            logger.info(
                "Frobenius norm of difference between ground metric wrt old {}".format(
                    torch.norm(M0 - M_old, "fro") / torch.norm(M_old, "fro")
                )
            )

            logger.info("ground metric old (i.e. bug involving wts) is {}".format(M_old))
            logger.info("ground metric new is {}".format(M0))

        ####################

        if args.same_model != -1:
            logger.info("Checking ground metric matrix in case of same models")
            if not args.gromov:
                logger.info(M0)
            else:
                logger.info(f"{M0}, {M1}")

        if args.skip_last_layer and idx == (num_layers - 1):
            if args.skip_last_layer_type == "average":
                logger.info("Simple averaging of last layer weights. NO transport map needs to be computed")
                if not_avg:
                    avg_aligned_layers.append(aligned_wt)
                else:
                    if args.ensemble_step != 0.5:
                        logger.info("taking baby steps (even in skip) ! ")
                        avg_aligned_layers.append(
                            (1 - args.ensemble_step) * aligned_wt + args.ensemble_step * fc_layer1_weight
                        )
                    else:
                        avg_aligned_layers.append(((aligned_wt + fc_layer1_weight) / 2))
            elif args.skip_last_layer_type == "second":
                logger.info("Just giving the weights of the second model. NO transport map needs to be computed")
                avg_aligned_layers.append(fc_layer1_weight)

            return avg_aligned_layers

        # logger.info("ground metric (m0) is {}".format(M0))

        T_var = _get_current_layer_transport_map(
            args, mu, nu, M0, M1, idx=idx, layer_shape=layer_shape, eps=eps, layer_name=layer0_name
        )

        T_var, marginals = _compute_marginals(args, T_var, device, eps=eps)

        if args.debug:
            if idx == (num_layers - 1):
                logger.info("there goes the last transport map: \n {}".format(T_var))
                logger.info("and before marginals it is {}".format(T_var / marginals))
            else:
                logger.info("there goes the transport map at layer {}: \n {}".format(idx, T_var))

        logger.info("Ratio of trace to the matrix sum: %f", torch.trace(T_var) / torch.sum(T_var))
        logger.info("Here, trace is {} and matrix sum is {} ".format(torch.trace(T_var), torch.sum(T_var)))
        setattr(args, "trace_sum_ratio_{}".format(layer0_name), (torch.trace(T_var) / torch.sum(T_var)).item())

        if args.past_correction:
            logger.info("Shape of aligned wt is {}".format(aligned_wt.shape))
            logger.info("Shape of fc_layer0_weight_data is {}".format(fc_layer0_weight_data.shape))
            t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
        else:
            t_fc0_model = torch.matmul(T_var.t(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1))

        # Average the weights of aligned first layers
        if not_avg:
            geometric_fc = t_fc0_model
        else:
            if args.ensemble_step != 0.5:
                logger.info("taking baby steps! ")
                geometric_fc = (
                    1 - args.ensemble_step
                ) * t_fc0_model + args.ensemble_step * fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
            else:
                geometric_fc = (t_fc0_model + fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)) / 2
        if is_conv and layer_shape != geometric_fc.shape:
            geometric_fc = geometric_fc.view(layer_shape)
        avg_aligned_layers.append(geometric_fc)

        if args.update_acts or args.eval_aligned:
            assert args.second_model_name is None
            # the thing is that there might be conv layers or other more intricate layers
            # hence there is no point in having them here
            # so instead call the compute_activations script and pass it the model0 aligned layers
            # and also the aligned weight computed (which has been aligned via the prev T map, i.e. incoming edges).
            if is_conv and layer_shape != t_fc0_model.shape:
                t_fc0_model = t_fc0_model.view(layer_shape)
            model0_aligned_layers.append(t_fc0_model)
            _, acc = update_model(args, networks[0], model0_aligned_layers, test=True, test_loader=test_loader, idx=0)
            logger.info("For layer idx {}, accuracy of the updated model is {}".format(idx, acc))
            setattr(args, "model0_aligned_acc_layer_{}".format(str(idx)), acc)
            if idx == (num_layers - 1):
                setattr(args, "model0_aligned_acc", acc)

        incoming_layer_aligned = False

        # remove cached variables to prevent out of memory
        activations_0 = None
        activations_1 = None
        mu = None
        nu = None
        fc_layer0_weight_data = None
        fc_layer1_weight_data = None
        M0 = None
        M1 = None
        idx += 1

    return avg_aligned_layers


def get_network_from_param_list(
    args, param_list, test_loader, model_name=None, model_config=None, dump_final_models=False, idx=-1
):
    logger.info("using independent method")
    new_network = get_model_from_name(args, idx=idx, model_name=model_name, model_config=model_config)
    if args.gpu_id != -1:
        new_network = new_network.cuda(args.gpu_id)

    # check the test performance of the network before
    log_dict = {}
    log_dict["test_losses"] = []
    routines.test(args, new_network, test_loader, log_dict)

    # set the weights of the new network
    logger.info(
        "len of model parameters and avg aligned layers is %d %d", len(list(new_network.parameters())), len(param_list)
    )
    assert len(list(new_network.parameters())) == len(param_list)

    layer_idx = 0
    model_state_dict = new_network.state_dict()

    logger.info("len of model_state_dict is %d", len(model_state_dict.items()))
    logger.info("len of param_list is %d", len(param_list))

    for key, _ in model_state_dict.items():
        # exclude some batch norm states
        if ("running_mean" in key) or ("running_var" in key) or ("num_batches_tracked" in key):
            continue
        model_state_dict[key] = param_list[layer_idx]
        layer_idx += 1

    new_network.load_state_dict(model_state_dict)

    # check the test performance of the network after
    log_dict = {}
    log_dict["test_losses"] = []
    acc = routines.test(args, new_network, test_loader, log_dict)
    if dump_final_models:
        routines.save_final_model(args, "OT_merge", new_network, acc)
    return acc, new_network


def geometric_ensembling_modularized(
    args, networks, train_loader, test_loader, activations=None, model1_config=None, idx=0
):
    if args.geom_ensemble_type == "wts":
        avg_aligned_layers = get_wassersteinized_layers_modularized(
            args, networks, activations, test_loader=test_loader
        )
    elif args.geom_ensemble_type == "acts":
        avg_aligned_layers = get_acts_wassersteinized_layers_modularized(
            args, networks, activations, train_loader=train_loader, test_loader=test_loader
        )

    return get_network_from_param_list(
        args,
        avg_aligned_layers,
        test_loader,
        model_name=args.fused_model_name,
        model_config=model1_config,
        dump_final_models=args.dump_final_models,
        idx=idx,
    )
