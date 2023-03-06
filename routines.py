import copy
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import train as cifar_train
from basic_config import PATH_TO_CIFAR
from data import get_dataloader
from log import logger
from model import get_model_from_name
from tqdm import tqdm
from utils import _get_config, get_model_size, to_first_position


sys.path.append(PATH_TO_CIFAR)


def get_trained_model(args, id, random_seed, train_loader, test_loader):
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    logger.info(f"Training seed is {random_seed}")
    network = get_model_from_name(args, idx=id)

    optimizer = optim.SGD(network.parameters(), lr=args.learning_rate, momentum=args.momentum)
    if args.gpu_id != -1:
        network = network.cuda(args.gpu_id)
    log_dict = {}
    log_dict["train_losses"] = []
    log_dict["train_counter"] = []
    log_dict["test_losses"] = []
    # log_dict['test_counter'] = [i * len(test_loader.dataset) for i in range(args.n_epochs + 1)]
    best_acc = test(args, network, test_loader, log_dict)
    best_model = network
    best_epoch = 0
    for epoch in range(1, args.n_epochs + 1):
        print("Epoch: " + str(epoch))
        train(args, network, optimizer, train_loader, log_dict, epoch, model_id=str(id))
        acc = test(args, network, test_loader, log_dict)
        if acc > best_acc:
            best_model = network
            best_acc = acc
            best_epoch = epoch
    logger.info(f"Model {id} has best accuracy of {best_acc} at epoch {best_epoch}.")
    logger.info(f"Model {id} has final accuracy of {acc} at last epoch {args.n_epochs}.")
    if args.ckpt_type == "final":
        return network, acc
    else:
        return best_model, best_acc


def check_freezed_params(model, frozen):
    flag = True
    for idx, param in enumerate(model.parameters()):
        if idx >= len(frozen):
            return flag

        flag = flag and (param.data == frozen[idx].data).all()

    return flag


def get_intmd_retrain_model(args, random_seed, network, aligned_wts, train_loader, test_loader):
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    logger.info(f"Training seed is {random_seed}")

    num_params_aligned = len(aligned_wts)
    for idx, param in enumerate(network.parameters()):
        if idx < num_params_aligned:
            param.requires_grad = False

    logger.info("number of layers that are intmd retrained %d", len(list(network.parameters())) - num_params_aligned)
    optimizer = optim.SGD(
        network.parameters(), lr=args.learning_rate * args.intmd_retrain_lrdec, momentum=args.momentum
    )
    log_dict = {}
    log_dict["train_losses"] = []
    log_dict["train_counter"] = []
    log_dict["test_losses"] = []
    # log_dict['test_counter'] = [i * len(test_loader.dataset) for i in range(args.n_epochs + 1)]
    acc = test(args, network, test_loader, log_dict)
    for epoch in range(1, args.intmd_retrain_epochs + 1):
        print("Epoch: " + str(epoch))
        train(args, network, optimizer, train_loader, log_dict, epoch, model_id=str(id))
        acc = test(args, network, test_loader, log_dict)

    logger.info(
        "Finally accuracy of model {} after intermediate retraining for {} epochs with lr decay {} is {}".format(
            random_seed, args.intmd_retrain_epochs, args.intmd_retrain_lrdec, acc
        )
    )

    assert check_freezed_params(network, aligned_wts)
    return network


def get_trained_data_separated_model(args, id, local_train_loader, local_test_loader, test_loader, base_net):
    torch.backends.cudnn.enabled = False
    network = copy.deepcopy(base_net)
    optimizer = optim.SGD(network.parameters(), lr=args.learning_rate, momentum=args.momentum)
    if args.gpu_id != -1:
        network = network.cuda(args.gpu_id)
    log_dict = {}
    log_dict["train_losses"] = []
    log_dict["train_counter"] = []
    log_dict["local_test_losses"] = []
    log_dict["test_losses"] = []
    # log_dict['test_counter'] = [i * len(test_loader.dataset) for i in range(args.n_epochs + 1)]
    acc = test(args, network, test_loader, log_dict)
    local_acc = test(args, network, local_test_loader, log_dict, is_local=True)
    for epoch in range(1, args.n_epochs + 1):
        print("Epoch: " + str(epoch))
        train(args, network, optimizer, local_train_loader, log_dict, epoch, model_id=str(id))
        acc = test(args, network, test_loader, log_dict)
        local_acc = test(args, network, local_test_loader, log_dict, is_local=True)
    return network, acc, local_acc


def get_retrained_model(args, train_loader, test_loader, old_network, tensorboard_obj=None, nick="", start_acc=-1):
    torch.backends.cudnn.enabled = False
    if args.retrain_lr_decay > 0:
        args.retrain_lr = args.learning_rate / args.retrain_lr_decay
        logger.info("optimizer_learning_rate is %f", args.retrain_lr)
    if args.retrain_seed != -1:
        torch.manual_seed(args.retrain_seed)
        logger.info(f"Training seed is {args.retrain_seed}")

    optimizer = optim.SGD(old_network.parameters(), lr=args.retrain_lr, momentum=args.momentum)
    log_dict = {}
    log_dict["train_losses"] = []
    log_dict["train_counter"] = []
    log_dict["test_losses"] = []
    # log_dict['test_counter'] = [i * len(train_loader.dataset) for i in range(args.n_epochs + 1)]

    acc = test(args, old_network, test_loader, log_dict)
    logger.info("check accuracy once again before retraining starts: %f", acc)

    if tensorboard_obj is not None and start_acc != -1:
        tensorboard_obj.add_scalars("test_accuracy_percent/", {nick: start_acc}, global_step=0)
        assert start_acc == acc

    best_acc = -1
    for epoch in range(1, args.retrain + 1):
        print("Epoch: " + str(epoch))
        train(args, old_network, optimizer, train_loader, log_dict, epoch)
        acc, loss = test(args, old_network, test_loader, log_dict, return_loss=True)

        if tensorboard_obj is not None:
            assert nick != ""
            tensorboard_obj.add_scalars("test_loss/", {nick: loss}, global_step=epoch)
            tensorboard_obj.add_scalars("test_accuracy_percent/", {nick: acc}, global_step=epoch)

        logger.info("At retrain epoch the accuracy is : %f", acc)
        best_acc = max(best_acc, acc)

    return old_network, best_acc


def get_pretrained_model(args, path, data_separated=False, idx=-1):
    if args.gpu_id != -1:
        state = torch.load(
            path,
            map_location=(lambda s, _: torch.serialization.default_restore_location(s, "cuda:" + str(args.gpu_id))),
        )
    else:
        state = torch.load(
            path,
            map_location=(lambda s, _: torch.serialization.default_restore_location(s, "cpu")),
        )

    # change the MlpNet config
    # the fused model has the same config as model 1
    if args.model_name == "mlpnet":
        load_args = state["args"]
        if ("parse_config" in load_args.keys()) and (load_args["parse_config"]):
            net_config = load_args["net_config"][idx]
            setattr(args, "num_hidden_layers", len(net_config))
            for layer_idx in range(1, args.num_hidden_layers + 1):
                param_name = "num_hidden_nodes" + str(layer_idx)
                setattr(args, param_name, net_config[layer_idx - 1])
        else:
            setattr(args, "num_hidden_layers", load_args["num_hidden_layers"])
            for layer_idx in range(1, args.num_hidden_layers + 1):
                param_name = "num_hidden_nodes" + str(layer_idx)
                setattr(args, param_name, load_args[param_name])

    model = get_model_from_name(args, idx=idx)

    model_state_dict = state["model_state_dict"]

    if "test_accuracy" not in state:
        state["test_accuracy"] = -1

    if "epoch" not in state:
        state["epoch"] = -1

    if not data_separated:
        logger.info(
            "Loading model at path {} which had accuracy {} and at epoch {}".format(
                path, state["test_accuracy"], state["epoch"]
            )
        )
    else:
        logger.info(
            "Loading model at path {} which had local accuracy {} and overall accuracy {} for choice {} at epoch {}".format(
                path, state["local_test_accuracy"], state["test_accuracy"], state["choice"], state["epoch"]
            )
        )

    model.load_state_dict(model_state_dict)

    if args.gpu_id != -1:
        model = model.cuda(args.gpu_id)

    if not data_separated:
        return model, state["test_accuracy"], state["epoch"]
    else:
        return model, state["test_accuracy"], state["local_test_accuracy"], state["choice"], state["epoch"]


def train(args, network, optimizer, train_loader, log_dict, epoch, model_id=-1):
    network.train()
    loss_func = F.nll_loss

    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        if args.gpu_id != -1:
            data = data.cuda(args.gpu_id)
            target = target.cuda(args.gpu_id)
        optimizer.zero_grad()
        output = network(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            log_dict["train_losses"].append(loss.item())
            log_dict["train_counter"].append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))

            assert args.exp_name == "exp_" + args.timestamp

            os.makedirs("{}/{}".format(args.result_dir, args.exp_name), exist_ok=True)
            if args.dump_model:
                assert model_id != -1
                torch.save(
                    network.state_dict(),
                    "{}/{}/model_{}_{}.pth".format(args.result_dir, args.exp_name, args.model_name, model_id),
                )
                torch.save(
                    optimizer.state_dict(),
                    "{}/{}/optimizer_{}_{}.pth".format(args.result_dir, args.exp_name, args.model_name, model_id),
                )


def test(args, network, test_loader, log_dict, debug=False, return_loss=False, is_local=False):
    network.eval()
    test_loss = 0
    correct = 0
    if is_local:
        logger.info("\n--------- Testing in local mode ---------")
    else:
        logger.info("\n--------- Testing in global mode ---------")

    if args.dataset.lower() != "mnist":
        loss_func = F.cross_entropy
        logger.info("Using CrossEntropyLoss")
    else:
        loss_func = F.nll_loss
        logger.info("Using negative log likelihood loss")

    for data, target in test_loader:
        if args.gpu_id != -1:
            data = data.cuda(args.gpu_id)
            target = target.cuda(args.gpu_id)

        output = network(data)
        if debug:
            logger.info("output is {}".format(output))

        test_loss += loss_func(output, target, reduction="sum").item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    logger.info("size of test_loader dataset: %d", len(test_loader.dataset))
    test_loss /= len(test_loader.dataset)
    if is_local:
        string_info = "local_test"
    else:
        string_info = "test"
    log_dict["{}_losses".format(string_info)].append(test_loss)
    logger.info(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

    ans = (float(correct) * 100.0) / len(test_loader.dataset)

    if not return_loss:
        return ans
    else:
        return ans, test_loss


def train_data_separated_models(
    args, local_train_loaders, local_test_loaders, test_loader, choices, model_configs=[None, None]
):
    if args.model_name == "mlpnet":
        networks = []
        local_accuracies = []
        accuracies = []
        base_nets = []
        base_net = get_model_from_name(args, idx=0)
        base_nets.append(base_net)
        if args.diff_init or args.width_ratio != 1 or args.parse_config:
            base_nets.append(get_model_from_name(args, idx=1))
        else:
            base_nets.append(base_net)

        for i in range(args.num_models):
            logger.info("\nTraining model {} on its separate data \n ".format(str(i)))
            network, acc, local_acc = get_trained_data_separated_model(
                args, i, local_train_loaders[i], local_test_loaders[i], test_loader, base_nets[i]
            )
            networks.append(network)
            accuracies.append(acc)
            local_accuracies.append(local_acc)
            if args.dump_final_models:
                save_final_data_separated_model(args, i, network, local_acc, acc, choices[i])
    elif "vgg" in args.model_name or "resnet" in args.model_name:
        networks = []
        local_accuracies = []
        accuracies = []

        for i in range(args.num_models):
            logger.info("\nTraining model {} on its separate data \n ".format(str(i)))
            model_output_dir = os.path.join(args.exp_path, "model_{}".format(i))
            logger.info("Model config {}".format(model_configs[i]))
            local_acc, acc, network = cifar_train.main(
                model_configs[i],
                model_output_dir,
                args.gpu_id,
                data_separated=True,
                pretrained_dataset=[local_train_loaders[i], local_test_loaders[i]],
                return_model=True,
            )
            networks.append(network)
            accuracies.append(acc)
            local_accuracies.append(local_acc)
    return networks, accuracies, local_accuracies


def train_models(args, train_loader, test_loader):
    networks = []
    accuracies = []
    for i in range(args.num_models):
        if args.train_seed >= 0:
            random_seed = args.train_seed
        else:
            random_seed = i
        network, acc = get_trained_model(args, i, random_seed, train_loader, test_loader)
        networks.append(network)
        accuracies.append(acc)
        if args.dump_final_models:
            save_final_model(args, i, network, acc)
    return networks, accuracies


def save_final_data_separated_model(args, idx, model, local_test_accuracy, test_accuracy, choice):
    path = os.path.join(args.result_dir, args.exp_name, "model_{}".format(idx))
    os.makedirs(path, exist_ok=True)
    import time

    # args.ckpt_type = 'final'
    time.sleep(1)  # workaround for RuntimeError('Unknown Error -1') https://github.com/pytorch/pytorch/issues/10577
    torch.save(
        {
            "args": vars(args),
            "epoch": args.n_epochs,
            "local_test_accuracy": local_test_accuracy,
            "test_accuracy": test_accuracy,
            "choice": str(choice),
            "model_state_dict": model.state_dict(),
        },
        os.path.join(path, "{}.checkpoint".format(args.ckpt_type)),
    )


def save_final_model(args, idx, model, test_accuracy):
    path = os.path.join(args.result_dir, args.exp_name, "model_{}".format(idx))
    os.makedirs(path, exist_ok=True)
    import time

    # args.ckpt_type = 'final'
    time.sleep(1)  # workaround for RuntimeError('Unknown Error -1') https://github.com/pytorch/pytorch/issues/10577
    torch.save(
        {
            "args": vars(args),
            "epoch": args.n_epochs,
            "test_accuracy": test_accuracy,
            "model_state_dict": model.state_dict(),
        },
        os.path.join(path, "{}.checkpoint".format(args.ckpt_type)),
    )


def retrain_models(
    args, old_networks, train_loader, test_loader, config, tensorboard_obj=None, initial_acc=None, nicks=None
):
    accuracies = []
    retrained_networks = []

    for i in range(len(old_networks)):
        nick = nicks[i]
        logger.info("Retraining model : %s", nick)

        # recheck accuracy
        log_dict = {}
        log_dict["test_losses"] = []
        logger.info("Rechecked accuracy is {}".format(test(args, old_networks[i], test_loader, log_dict)))

        if initial_acc is not None:
            start_acc = initial_acc[i]
        else:
            start_acc = -1
        if args.dataset.lower() != "mnist":
            if args.reinit_trainloaders:
                logger.info("reiniting trainloader")
                retrain_loader, _ = cifar_train.get_dataset(config, no_randomness=args.no_random_trainloaders)
            else:
                retrain_loader = train_loader

            output_root_dir = os.path.join(args.exp_path, f"retrain_{nick}")
            os.makedirs(output_root_dir, exist_ok=True)

            retrained_network, acc = cifar_train.get_retrained_model(
                args,
                retrain_loader,
                test_loader,
                old_networks[i],
                config,
                output_root_dir,
                tensorboard_obj=tensorboard_obj,
                nick=nick,
                start_acc=initial_acc[i],
            )

        else:
            if args.reinit_trainloaders:
                logger.info("reiniting trainloader")
                retrain_loader, _ = get_dataloader(args, no_randomness=args.no_random_trainloaders)
            else:
                retrain_loader = train_loader

            start_acc = initial_acc[i]
            retrained_network, acc = get_retrained_model(
                args,
                retrain_loader,
                test_loader,
                old_network=old_networks[i],
                tensorboard_obj=tensorboard_obj,
                nick=nick,
                start_acc=start_acc,
            )
        retrained_networks.append(retrained_network)
        accuracies.append(acc)
    return retrained_networks, accuracies


def intmd_retrain_models(
    args, old_networks, aligned_wts, train_loader, test_loader, config, tensorboard_obj=None, initial_acc=None
):
    accuracies = []
    retrained_networks = []
    # nicks = []

    # assert len(old_networks) >= 4

    for i in range(len(old_networks)):
        nick = "intmd_retrain_model_" + str(i)
        logger.info("Retraining model : %s", nick)

        if initial_acc is not None:
            start_acc = initial_acc[i]
        else:
            start_acc = -1
        if args.dataset.lower() != "mnist":
            output_root_dir = "{}/{}_models_ensembled/".format(args.baseroot, (args.dataset).lower())
            output_root_dir = os.path.join(output_root_dir, args.exp_name, nick)
            os.makedirs(output_root_dir, exist_ok=True)

            retrained_network, acc = cifar_train.get_retrained_model(
                args,
                train_loader,
                test_loader,
                old_networks[i],
                config,
                output_root_dir,
                tensorboard_obj=tensorboard_obj,
                nick=nick,
                start_acc=start_acc,
            )

        else:
            # start_acc = initial_acc[i]
            retrained_network, acc = get_intmd_retrain_model(
                args,
                train_loader,
                test_loader,
                old_network=old_networks[i],
                tensorboard_obj=tensorboard_obj,
                nick=nick,
                start_acc=start_acc,
            )
        retrained_networks.append(retrained_network)
        accuracies.append(acc)
    return retrained_networks, accuracies


def initial_loading(args):
    # loading configuration
    config_list = _get_config(args)
    # Move the longest model to the first position
    config_list = to_first_position(config_list, args.longest_model_index)
    args.model_name_list = to_first_position(args.model_name_list, args.longest_model_index)
    args.config = config_list[0]
    # Set the seed
    logger.info(f"Set seed is {args.config['seed']}")
    torch.manual_seed(args.config["seed"])
    torch.cuda.manual_seed(args.config["seed"])
    np.random.seed(args.config["seed"])
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # obtain trained models
    if args.load_models != "":
        logger.info("------- Loading pre-trained models -------")
        ensemble_experiment = args.load_models.split("/")
        if len(ensemble_experiment) > 1:
            # both the path and name of the experiment have been specified
            ensemble_dir = args.load_models
        elif len(ensemble_experiment) == 1:
            # otherwise append the directory before!
            ensemble_root_dir = "{}/{}_models/".format(args.baseroot, (args.dataset).lower())
            ensemble_dir = ensemble_root_dir + args.load_models

        # checkpoint_type = 'final'  # which checkpoint to use for ensembling (either of 'best' or 'final)

        if args.dataset == "mnist":
            train_loader, test_loader = get_dataloader(args)
            retrain_loader, _ = get_dataloader(args, no_randomness=args.no_random_trainloaders)
        else:
            if args.gpu_id == -1:
                num_workers = 0
            else:
                num_workers = 2
            args.cifar_init_lr = config_list[0]["optimizer_learning_rate"]
            logger.info("loading {} dataloaders".format(args.dataset.lower()))
            train_loader, test_loader = cifar_train.get_dataset(
                args.config, to_download=args.to_download, num_workers=num_workers
            )
            retrain_loader, _ = cifar_train.get_dataset(
                args.config,
                no_randomness=args.no_random_trainloaders,
                to_download=args.to_download,
                num_workers=num_workers,
            )

        models = []
        accuracies = []
        epochs = []

        for idx in range(args.num_models):
            # Move the longest model to the first position
            if idx == args.longest_model_index:
                idx = 0
            elif idx == 0:
                idx = args.longest_model_index
            logger.info("loading model with idx {} and checkpoint_type is {}".format(idx, args.ckpt_type))
            model_name = args.model_name_list[idx]
            if model_name.lower()[0:3] == "vgg" or model_name.lower()[0:6] == "resnet":
                config_used = config_list[idx]

                model, accuracy, epoch = cifar_train.get_pretrained_model(
                    config_used,
                    os.path.join(ensemble_dir, "model_{}/{}.checkpoint".format(idx, args.ckpt_type)),
                    args.gpu_id,
                    relu_inplace=not args.prelu_acts,  # if you want pre-relu acts, set relu_inplace to False
                )
            else:
                model, accuracy, epoch = get_pretrained_model(
                    args, os.path.join(ensemble_dir, "model_{}/{}.checkpoint".format(idx, args.ckpt_type)), idx=idx
                )

            models.append(model)
            accuracies.append(accuracy)
            epochs.append(epoch)
        logger.info("Done loading all the models")

        # Additional flag of recheck_acc to supplement the legacy flag recheck_cifar
        if args.recheck_cifar or args.recheck_acc:
            recheck_accuracies = []
            for model in models:
                log_dict = {}
                log_dict["test_losses"] = []
                recheck_accuracies.append(test(args, model, test_loader, log_dict))
            logger.info("Rechecked accuracies are {}".format(recheck_accuracies))

    else:
        epochs = [args.n_epochs, args.n_epochs]
        # get dataloaders
        logger.info("------- Obtain dataloaders -------")
        train_loader, test_loader = get_dataloader(args)
        retrain_loader, _ = get_dataloader(args, no_randomness=args.no_random_trainloaders)

        logger.info("------- Training independent models -------")
        models, accuracies = train_models(args, train_loader, test_loader)

    # get model size
    for idx, model in enumerate(models):
        setattr(args, f"params_model_{idx}", get_model_size(model))

    results = {
        "args": args,
        "config": args.config,
        "models": models,
        "accuracies": accuracies,
        "epochs": epochs,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "retrain_loader": retrain_loader,
    }

    return results
