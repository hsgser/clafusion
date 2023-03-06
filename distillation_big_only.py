import copy
import os
import sys

import baseline
import numpy as np
import parameters
import routines
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import train as cifar_train
import utils
import wasserstein_ensemble
from basic_config import PATH_TO_CIFAR, TMP_DATETIME_FILE
from data import get_dataloader
from log import logger
from model import get_model_from_name
from tqdm import tqdm


sys.path.append(PATH_TO_CIFAR)


def recheck_accuracy(args, models, test_loader):
    # Additional flag of recheck_acc to supplement the legacy flag recheck_cifar
    if args.recheck_cifar or args.recheck_acc:
        recheck_accuracies = []
        for model in models:
            log_dict = {}
            log_dict["test_losses"] = []
            recheck_accuracies.append(routines.test(args, model, test_loader, log_dict))
        logger.info(f"Rechecked accuracies are {recheck_accuracies}")


def get_dataloaders(args, config):
    if args.dataset == "mnist":
        train_loader, test_loader = get_dataloader(args)
        retrain_loader, _ = get_dataloader(args, no_randomness=args.no_random_trainloaders)
    else:
        assert config is not None
        args.cifar_init_lr = config["optimizer_learning_rate"]
        if args.second_model_name is not None:
            assert args.second_config is not None
            # also the below things should be fine as it is just dataloader loading!
        logger.info("loading {} dataloaders".format(args.dataset.lower()))
        train_loader, test_loader = cifar_train.get_dataset(config)
        retrain_loader, _ = cifar_train.get_dataset(config, no_randomness=args.no_random_trainloaders)

    return train_loader, test_loader, retrain_loader


def load_pretrained_models(args, config, second_config=None):
    logger.info("------- Loading pre-trained models -------")
    ensemble_experiment = args.load_models.split("/")
    if len(ensemble_experiment) > 1:
        # both the path and name of the experiment have been specified
        ensemble_dir = args.load_models
    elif len(ensemble_experiment) == 1:
        # otherwise append the directory before!
        ensemble_root_dir = "{}/{}_models/".format(args.baseroot, (args.dataset).lower())
        ensemble_dir = ensemble_root_dir + args.load_models

    models = []
    accuracies = []

    for idx in range(args.num_models):
        logger.info("loading model with idx {} and checkpoint_type is {}".format(idx, args.ckpt_type))

        if args.dataset.lower() != "mnist" and (
            args.model_name.lower()[0:5] == "vgg11" or args.model_name.lower()[0:6] == "resnet"
        ):
            if idx == 0:
                config_used = config
            elif idx == 1:
                config_used = second_config

            model, accuracy, _ = cifar_train.get_pretrained_model(
                config_used,
                os.path.join(ensemble_dir, "model_{}/{}.checkpoint".format(idx, args.ckpt_type)),
                args.gpu_id,
                relu_inplace=not args.prelu_acts,  # if you want pre-relu acts, set relu_inplace to False
            )
        else:
            model, accuracy, _ = routines.get_pretrained_model(
                args, os.path.join(ensemble_dir, "model_{}/{}.checkpoint".format(idx, args.ckpt_type)), idx=idx
            )

        models.append(model)

        accuracies.append(accuracy)
    logger.info("Done loading all the models")

    return models, accuracies


def test_model(args, model, test_loader):
    log_dict = {}
    log_dict["test_losses"] = []
    return routines.test(args, model, test_loader, log_dict)


def loss_fn_kd(outputs, labels, teacher_outputs, params):
    # Source: https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = params.alpha
    T = params.temperature
    KD_loss = nn.KLDivLoss(reduction="mean")(
        F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs / T, dim=1)
    ) * (alpha * T * T) + F.cross_entropy(outputs, labels) * (1.0 - alpha)

    return KD_loss


def distillation(args, teachers, student, train_loader, test_loader, device):
    # Inspiration: https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/evaluate.py

    for teacher in teachers:
        teacher.eval()

    optimizer = optim.SGD(student.parameters(), lr=args.learning_rate, momentum=args.momentum)

    log_dict = {}
    log_dict["train_losses"] = []
    log_dict["train_counter"] = []
    log_dict["test_losses"] = []

    accuracies = []
    accuracies.append(routines.test(args, student, test_loader, log_dict))
    for epoch_idx in range(0, args.dist_epochs):
        print("Epoch: " + str(epoch_idx))
        student.train()

        for batch_idx, (data_batch, labels_batch) in tqdm(enumerate(train_loader)):
            # move to GPU if available
            if args.gpu_id != -1:
                data_batch, labels_batch = data_batch.to(device), labels_batch.to(device)

            # compute mean teacher output
            teacher_outputs = []
            for teacher in teachers:
                teacher_outputs.append(teacher(data_batch, disable_logits=True))
            teacher_outputs = torch.stack(teacher_outputs)
            teacher_outputs = teacher_outputs.mean(dim=0)
            optimizer.zero_grad()
            # get student output
            student_output = student(data_batch, disable_logits=True)

            # knowledge distillation loss
            loss = loss_fn_kd(student_output, labels_batch, teacher_outputs, args)
            loss.backward()
            # update student
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch_idx,
                        batch_idx * len(data_batch),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
                log_dict["train_losses"].append(loss.item())
                log_dict["train_counter"].append((batch_idx * 64) + ((epoch_idx - 1) * len(train_loader.dataset)))

        accuracies.append(routines.test(args, student, test_loader, log_dict))

    return student, accuracies


def distillation_cnn(args, teachers, student, train_loader, test_loader, device):
    # Inspiration: https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/evaluate.py

    for teacher in teachers:
        teacher.eval()

    assert args.second_config is not None
    # update the parameters
    config = args.second_config
    config["num_epochs"] = args.retrain
    logger.info("number of epochs would be %d", config["num_epochs"])

    if args.retrain_lr_decay > 0:
        config["optimizer_learning_rate"] = args.cifar_init_lr / args.retrain_lr_decay
        logger.info("optimizer_learning_rate is %f", config["optimizer_learning_rate"])

    if args.retrain_lr_decay_factor is not None:
        config["optimizer_decay_with_factor"] = args.retrain_lr_decay_factor
        logger.info("optimizer lr decay factor is %f", config["optimizer_decay_with_factor"])

    if args.retrain_lr_decay_epochs is not None:
        config["optimizer_decay_at_epochs"] = [int(ep) for ep in args.retrain_lr_decay_epochs.split("_")]
        logger.info("optimizer lr decay epochs is {}".format(config["optimizer_decay_at_epochs"]))

    optimizer, scheduler = cifar_train.get_optimizer(config, student.parameters())

    log_dict = {}
    log_dict["train_losses"] = []
    log_dict["train_counter"] = []
    log_dict["test_losses"] = []

    accuracies = []
    accuracies.append(routines.test(args, student, test_loader, log_dict))
    for epoch_idx in range(0, config["num_epochs"]):
        print("Epoch: " + str(epoch_idx))
        student.train()

        for batch_idx, (data_batch, labels_batch) in tqdm(enumerate(train_loader)):
            # move to GPU if available
            if args.gpu_id != -1:
                data_batch, labels_batch = data_batch.to(device), labels_batch.to(device)

            # compute mean teacher output
            teacher_outputs = []
            for teacher in teachers:
                teacher_outputs.append(teacher(data_batch))
            teacher_outputs = torch.stack(teacher_outputs)
            teacher_outputs = teacher_outputs.mean(dim=0)
            optimizer.zero_grad()
            # get student output
            student_output = student(data_batch)

            # knowledge distillation loss
            loss = loss_fn_kd(student_output, labels_batch, teacher_outputs, args)
            loss.backward()
            # update student
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch_idx,
                        batch_idx * len(data_batch),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
                log_dict["train_losses"].append(loss.item())
                log_dict["train_counter"].append((batch_idx * 64) + ((epoch_idx - 1) * len(train_loader.dataset)))

        accuracies.append(routines.test(args, student, test_loader, log_dict))
        scheduler.step()

    return student, accuracies


if __name__ == "__main__":
    args = parameters.get_parameters()

    if args.width_ratio != 1:
        if not args.proper_marginals:
            logger.info("setting proper marginals to True (needed for width_ratio!=1 mode)")
            args.proper_marginals = True
        if args.eval_aligned:
            logger.info("setting eval aligned to False (needed for width_ratio!=1 mode)")
            args.eval_aligned = False

    logger.info("The parameters are: \n {}".format(args))

    config, second_config = utils._get_config(args)
    args.config = config
    args.second_config = second_config
    # Set the seed
    if "vgg" in args.model_name or "resnet" in args.model_name:
        logger.info(f"Training seed is {config['seed']}")
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        my_distillation = distillation_cnn
    else:
        if args.train_seed < 0:
            args.train_seed = 0
        logger.info(f"Training seed is {args.train_seed}")
        torch.manual_seed(args.train_seed)
        np.random.seed(args.train_seed)
        my_distillation = distillation

    setattr(args, "autoencoder", False)
    train_loader, test_loader, retrain_loader = get_dataloaders(args, config)

    models, accuracies = load_pretrained_models(args, config, second_config=second_config)

    recheck_accuracy(args, models, test_loader)

    for idx, model in enumerate(models):
        model_size = utils.get_model_size(model)
        logger.info("model {} size is {}".format(idx, model_size))
        setattr(args, f"params_model_{idx}", model_size)
        test_model(args, model, test_loader)

    if args.gpu_id == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(args.gpu_id))

    logger.info("------- Prediction based ensembling -------")
    prediction_acc = baseline.prediction_ensembling(args, models, test_loader)

    if args.not_dist_geometric:
        geometric_acc = -1
        args.params_geometric = -1
    else:
        logger.info("------- Geometric Ensembling -------")
        args.fused_model_name = args.second_model_name
        activations = utils.get_model_activations(args, models, config=config)
        geometric_acc, geometric_model = wasserstein_ensemble.geometric_ensembling_modularized(
            args, models, train_loader, test_loader, activations
        )
        args.params_geometric = utils.get_model_size(geometric_model)

    logger.info("------- Distillation!! -------")
    distilled_model = get_model_from_name(args, idx=1)
    distilled_model = distilled_model.to(device)
    args.params_distill = utils.get_model_size(distilled_model)

    distill_scratch_init_acc = test_model(args, distilled_model, test_loader)

    distillation_results = {}

    logger.info("------- Distilling Big to scratch -------")
    _, acc = my_distillation(args, [models[0]], copy.deepcopy(distilled_model), train_loader, test_loader, device)
    distillation_results["scratch_distill_from_big"] = acc

    if args.not_dist_geometric:
        distillation_results["geometric_distill_from_big"] = [-1]
    else:
        logger.info("------- Distilling Big to OT Avg. -------")
        _, acc = my_distillation(args, [models[0]], copy.deepcopy(geometric_model), train_loader, test_loader, device)
        distillation_results["geometric_distill_from_big"] = acc

    logger.info("------- Distilling Big to Model B -------")
    _, acc = my_distillation(args, [models[0]], copy.deepcopy(models[1]), train_loader, test_loader, device)
    distillation_results["model_b_distill_from_big"] = acc

    if args.save_result_file != "":
        results_dic = {}
        results_dic["exp_name"] = args.exp_name

        for idx, acc in enumerate(accuracies):
            results_dic["model{}_acc".format(idx)] = acc

        results_dic["geometric_acc"] = geometric_acc
        results_dic["prediction_acc"] = prediction_acc
        results_dic["distill_scratch_init_acc"] = distill_scratch_init_acc

        # distillation acc results
        for distill_name, acc in distillation_results.items():
            results_dic[f"best_{distill_name}"] = max(acc)
            results_dic[f"idx_{distill_name}"] = np.argmax(np.array(acc))
            results_dic[f"acc_{distill_name}"] = acc

        utils.save_results_params_csv(os.path.join(args.csv_dir, args.save_result_file), results_dic, args)

        logger.info("----- Saved results at {} ------".format(args.save_result_file))
        logger.info(results_dic)

    logger.info("FYI: the parameters were: \n {}".format(args))

    logger.info("------- ------- ------- ------- -------")
    os.remove(TMP_DATETIME_FILE)
