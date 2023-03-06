#!/usr/bin/env python3
import copy
import logging as logger
import os
import sys
import time
import warnings

import cifar_utils.accumulators
import models
import numpy as np
import torch
import torchvision
from tqdm import tqdm


warnings.simplefilter(action="ignore", category=FutureWarning)
sys.path.insert(0, "..")


def main(
    config,
    output_dir,
    gpu_id,
    data_separated=False,
    pretrained_model=None,
    pretrained_dataset=None,
    tensorboard_obj=None,
    return_model=False,
):
    """
    Train a model
    You can either call this script directly (using the default parameters),
    or import it as a module, override config and run main()
    :return: scalar of the best accuracy
    """

    # Set the seed
    logger.info(f"Training seed is {config['seed']}")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(gpu_id)

    # Configure the dataset, model and the optimizer based on the global
    # `config` dictionary.
    if data_separated:
        assert pretrained_dataset is not None
        training_loader, local_test_loader = pretrained_dataset
        _, test_loader = get_dataset(config)
    else:
        training_loader, test_loader = get_dataset(config)

    if pretrained_model is not None:
        model = pretrained_model
    else:
        model = get_model(config, gpu_id)

    best_model = None
    optimizer, scheduler, need_metric = get_optimizer(config, model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    if tensorboard_obj is not None and config["start_acc"] != -1:
        assert config["nick"] != ""
        tensorboard_obj.add_scalars("test_accuracy_percent/", {config["nick"]: config["start_acc"]}, global_step=0)

    # We keep track of the best accuracy so far to store checkpoints
    best_local_accuracy_so_far = cifar_utils.accumulators.Max()
    best_accuracy_so_far = cifar_utils.accumulators.Max()

    logger.info("number of epochs would be %d", config["num_epochs"])
    for epoch in range(config["num_epochs"]):
        print("Epoch: " + str(epoch))
        logger.info("Epoch {:03d}".format(epoch))

        # Enable training mode (automatic differentiation + batch norm)
        model.train()

        # Keep track of statistics during training
        mean_train_accuracy = cifar_utils.accumulators.Mean()
        mean_train_loss = cifar_utils.accumulators.Mean()

        for batch_x, batch_y in tqdm(training_loader):
            batch_x, batch_y = batch_x.cuda(gpu_id), batch_y.cuda(gpu_id)

            # Compute gradients for the batch
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            acc = accuracy(prediction, batch_y)
            loss.backward()

            # Do an optimizer steps
            optimizer.step()

            # Store the statistics
            mean_train_loss.add(loss.item(), weight=len(batch_x))
            mean_train_accuracy.add(acc.item(), weight=len(batch_x))

        # Log training stats
        log_metric("accuracy", {"epoch": epoch, "value": mean_train_accuracy.value() * 100}, {"split": "train"})
        log_metric("cross_entropy", {"epoch": epoch, "value": mean_train_loss.value()}, {"split": "train"})

        # Evaluation
        model.eval()
        if data_separated:
            mean_local_test_accuracy = cifar_utils.accumulators.Mean()
            mean_local_test_loss = cifar_utils.accumulators.Mean()
            logger.info("\n--------- Testing in local mode ---------")
            for batch_x, batch_y in local_test_loader:
                batch_x, batch_y = batch_x.cuda(gpu_id), batch_y.cuda(gpu_id)
                prediction = model(batch_x)
                loss = criterion(prediction, batch_y)
                acc = accuracy(prediction, batch_y)
                mean_local_test_loss.add(loss.item(), weight=len(batch_x))
                mean_local_test_accuracy.add(acc.item(), weight=len(batch_x))

            # Log test stats
            log_metric(
                "accuracy", {"epoch": epoch, "value": mean_local_test_accuracy.value() * 100}, {"split": "local_test"}
            )
            log_metric(
                "cross_entropy", {"epoch": epoch, "value": mean_local_test_loss.value()}, {"split": "local_test"}
            )

        logger.info("\n--------- Testing in global mode ---------")
        mean_test_accuracy = cifar_utils.accumulators.Mean()
        mean_test_loss = cifar_utils.accumulators.Mean()
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.cuda(gpu_id), batch_y.cuda(gpu_id)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            acc = accuracy(prediction, batch_y)
            mean_test_loss.add(loss.item(), weight=len(batch_x))
            mean_test_accuracy.add(acc.item(), weight=len(batch_x))

        # Log test stats
        log_metric("accuracy", {"epoch": epoch, "value": mean_test_accuracy.value() * 100}, {"split": "test"})
        log_metric("cross_entropy", {"epoch": epoch, "value": mean_test_loss.value()}, {"split": "test"})
        # Tensorboard
        if tensorboard_obj is not None:
            assert config["nick"] != ""
            tensorboard_obj.add_scalars(
                "lr/", {config["nick"]: optimizer.param_groups[0]["lr"]}, global_step=(epoch + 1)
            )
            tensorboard_obj.add_scalars(
                "train_loss/", {config["nick"]: mean_train_loss.value()}, global_step=(epoch + 1)
            )
            tensorboard_obj.add_scalars(
                "train_accuracy_percent/", {config["nick"]: mean_train_accuracy.value() * 100}, global_step=(epoch + 1)
            )
            if data_separated:
                tensorboard_obj.add_scalars(
                    "local_test_loss/", {config["nick"]: mean_local_test_loss.value()}, global_step=(epoch + 1)
                )
                tensorboard_obj.add_scalars(
                    "local_test_accuracy_percent/",
                    {config["nick"]: mean_local_test_accuracy.value() * 100},
                    global_step=(epoch + 1),
                )
            tensorboard_obj.add_scalars(
                "test_loss/", {config["nick"]: mean_test_loss.value()}, global_step=(epoch + 1)
            )
            tensorboard_obj.add_scalars(
                "test_accuracy_percent/", {config["nick"]: mean_test_accuracy.value() * 100}, global_step=(epoch + 1)
            )

        # Store checkpoints for the best model so far
        if data_separated:
            is_best_local_so_far = best_local_accuracy_so_far.add(mean_local_test_accuracy.value())
            is_best_so_far = best_accuracy_so_far.add(mean_test_accuracy.value())
            if is_best_local_so_far:
                best_model = model
                store_data_separated_checkpoint(
                    output_dir,
                    "best.checkpoint",
                    model,
                    epoch,
                    mean_local_test_accuracy.value(),
                    mean_test_accuracy.value(),
                )
        else:
            is_best_so_far = best_accuracy_so_far.add(mean_test_accuracy.value())
            if is_best_so_far:
                best_model = model
                store_checkpoint(output_dir, "best.checkpoint", model, epoch, mean_test_accuracy.value())

        # Update the optimizer's learning rate
        if need_metric:
            scheduler.step(mean_test_loss.value())
        else:
            scheduler.step()

    # Store a final checkpoint
    if data_separated:
        store_data_separated_checkpoint(
            output_dir,
            "final.checkpoint",
            model,
            config["num_epochs"] - 1,
            mean_local_test_accuracy.value(),
            mean_test_accuracy.value(),
        )
        # Return the optimal accuracy, could be used for learning rate tuning
        if return_model:
            return best_local_accuracy_so_far.value(), best_accuracy_so_far.value(), best_model
        else:
            return best_local_accuracy_so_far.value(), best_accuracy_so_far.value()
    else:
        store_checkpoint(output_dir, "final.checkpoint", model, config["num_epochs"] - 1, mean_test_accuracy.value())
        # Return the optimal accuracy, could be used for learning rate tuning
        if return_model:
            return best_accuracy_so_far.value(), best_model
        else:
            return best_accuracy_so_far.value()


def accuracy(predicted_logits, reference):
    """Compute the ratio of correctly predicted labels"""
    labels = torch.argmax(predicted_logits, 1)
    correct_predictions = labels.eq(reference)
    return correct_predictions.sum().float() / correct_predictions.nelement()


def log_metric(name, values, tags):
    """
    Log timeseries data.
    Placeholder implementation.
    This function should be overwritten by any script that runs this as a module.
    """
    logger.info("{name}: {values} ({tags})".format(name=name, values=values, tags=tags))


def get_dataset(
    config,
    test_batch_size=100,
    shuffle_train=True,
    num_workers=2,
    data_root="./data",
    unit_batch_train=False,
    no_randomness=False,
    to_download=True,
):
    """
    Create dataset loaders for the chosen dataset
    :return: Tuple (training_loader, test_loader)
    """
    if config["dataset"].lower() == "cifar10":
        dataset = torchvision.datasets.CIFAR10
        data_mean = (0.4914, 0.4822, 0.4465)
        data_stddev = (0.2023, 0.1994, 0.2010)
    elif config["dataset"].lower() == "cifar100":
        dataset = torchvision.datasets.CIFAR100
        data_mean = (0.5071, 0.4867, 0.4408)
        data_stddev = (0.2675, 0.2565, 0.2761)
        # numbers taken from https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
    elif config["dataset"].lower() == "tinyimagenet":
        # no_randomness = True
        # data_mean = (0.485, 0.456, 0.406)
        # data_stddev = (0.229, 0.224, 0.225)
        # numbers taken from https://github.com/kennethleungty/PyTorch-Ignite-Tiny-ImageNet-Classification/blob/main/Tiny_ImageNet_Classification.ipynb
        from data_engine import DataEngine

        new_config = copy.deepcopy(config)

        if unit_batch_train or (num_workers == 0):
            new_config["batch_size"] = 1

        new_config["num_workers"] = num_workers
        new_config["train_data_path"] = os.path.join(data_root, "tiny-imagenet-200/train")
        new_config["test_data_path"] = os.path.join(data_root, "tiny-imagenet-200/val/images")
        data = DataEngine(new_config)

        return data.train_loader, data.test_loader
    elif config["dataset"].lower() == "esc50":
        from esc50 import get_train_test_dataloader

        if unit_batch_train or (num_workers == 0):
            train_batch_size = 1
        else:
            train_batch_size = config["batch_size"]
            num_workers = train_batch_size

        return get_train_test_dataloader(data_root, train_batch_size, num_workers, test_fold=config["test_fold"])
    else:
        raise ValueError("Unexpected value for config[dataset] {}".format(config["dataset"]))

    # TODO: I guess the randomness at random transforms is at play!
    # TODO: I think in retrain if I fix this, then the issue should be resolved
    if no_randomness:
        transform_train = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(data_mean, data_stddev),
            ]
        )
        shuffle_train = False
        logger.info("disabling shuffle train as well in no_randomness!")
    else:
        transform_train = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(data_mean, data_stddev),
            ]
        )

    transform_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(data_mean, data_stddev),
        ]
    )

    training_set = dataset(root=data_root, train=True, download=to_download, transform=transform_train)
    test_set = dataset(root=data_root, train=False, download=to_download, transform=transform_test)

    if unit_batch_train or (num_workers == 0):
        train_batch_size = 1
    else:
        train_batch_size = config["batch_size"]

    training_loader = torch.utils.data.DataLoader(
        training_set, batch_size=train_batch_size, shuffle=shuffle_train, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers
    )

    return training_loader, test_loader


def get_optimizer(config, model_parameters):
    """
    Create an optimizer for a given model
    :param model_parameters: a list of parameters to be trained
    :return: Tuple (optimizer, scheduler)
    """
    logger.info("lr is %f", config["optimizer_learning_rate"])
    if config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(
            model_parameters,
            lr=config["optimizer_learning_rate"],
            momentum=config["optimizer_momentum"],
            weight_decay=config["optimizer_weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config["optimizer_decay_at_epochs"],
            gamma=1.0 / config["optimizer_decay_with_factor"],
        )
        need_metric = False
    elif config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model_parameters, lr=config["optimizer_learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=5, verbose=True, min_lr=1e-3 * 1e-4, factor=0.33
        )
        need_metric = True
    else:
        raise ValueError("Unexpected value for optimizer")

    return optimizer, scheduler, need_metric


def get_model(config, device=-1, relu_inplace=True, model_config=None):
    """
    :param device: instance of torch.device
    :return: An instance of torch.nn.Module
    """
    if config["dataset"].lower() == "cifar10":
        num_classes = 10
    elif config["dataset"].lower() == "cifar100":
        num_classes = 100
    elif config["dataset"].lower() == "tinyimagenet":
        num_classes = 200
    elif config["dataset"].lower() == "esc50":
        num_classes = 50

    if model_config:
        assert "vgg" in config["model"]
        model = models.VGG(model_config, num_classes, batch_norm=False, bias=False, relu_inplace=relu_inplace)
    else:
        model = {
            "vgg11_nobias": lambda: models.VGG(
                models.cfg["VGG11"], num_classes, batch_norm=False, bias=False, relu_inplace=relu_inplace
            ),
            "vgg11_half_nobias": lambda: models.VGG(
                models.cfg["VGG11_half"], num_classes, batch_norm=False, bias=False, relu_inplace=relu_inplace
            ),
            "vgg11_doub_nobias": lambda: models.VGG(
                models.cfg["VGG11_doub"], num_classes, batch_norm=False, bias=False, relu_inplace=relu_inplace
            ),
            "vgg11_quad_nobias": lambda: models.VGG(
                models.cfg["VGG11_quad"], num_classes, batch_norm=False, bias=False, relu_inplace=relu_inplace
            ),
            "vgg11": lambda: models.VGG(models.cfg["VGG11"], num_classes, batch_norm=False, relu_inplace=relu_inplace),
            "vgg11_bn": lambda: models.VGG(
                models.cfg["VGG11"], num_classes, batch_norm=True, relu_inplace=relu_inplace
            ),
            "vgg8_nobias": lambda: models.VGG(
                models.cfg["VGG8"], num_classes, batch_norm=False, bias=False, relu_inplace=relu_inplace
            ),
            "vgg13_nobias": lambda: models.VGG(
                models.cfg["VGG13"], num_classes, batch_norm=False, bias=False, relu_inplace=relu_inplace
            ),
            "vgg13_student_nobias": lambda: models.VGG(
                models.cfg["VGG13_student"], num_classes, batch_norm=False, bias=False, relu_inplace=relu_inplace
            ),
            "vgg13_half_nobias": lambda: models.VGG(
                models.cfg["VGG13_half"], num_classes, batch_norm=False, bias=False, relu_inplace=relu_inplace
            ),
            "vgg13_doub_nobias": lambda: models.VGG(
                models.cfg["VGG13_doub"], num_classes, batch_norm=False, bias=False, relu_inplace=relu_inplace
            ),
            "vgg13_quad_nobias": lambda: models.VGG(
                models.cfg["VGG13_quad"], num_classes, batch_norm=False, bias=False, relu_inplace=relu_inplace
            ),
            "vgg16_nobias": lambda: models.VGG(
                models.cfg["VGG16"], num_classes, batch_norm=False, bias=False, relu_inplace=relu_inplace
            ),
            "vgg19_nobias": lambda: models.VGG(
                models.cfg["VGG19"], num_classes, batch_norm=False, bias=False, relu_inplace=relu_inplace
            ),
            "resnet18": lambda: models.ResNet18(num_classes=num_classes),
            "resnet18_nobias": lambda: models.ResNet18(num_classes=num_classes, linear_bias=False),
            "resnet18_nobn": lambda: models.ResNet18(num_classes=num_classes, use_batchnorm=False),
            "resnet18_nobias_nobn": lambda: models.ResNet18(
                num_classes=num_classes, use_batchnorm=False, linear_bias=False
            ),
            "resnet18_eighth_nobias_nobn": lambda: models.ResNet18(
                num_classes=num_classes, use_batchnorm=False, linear_bias=False, width_ratio=8
            ),
            "resnet18_fourth_nobias_nobn": lambda: models.ResNet18(
                num_classes=num_classes, use_batchnorm=False, linear_bias=False, width_ratio=4
            ),
            "resnet18_half_nobias_nobn": lambda: models.ResNet18(
                num_classes=num_classes, use_batchnorm=False, linear_bias=False, width_ratio=2
            ),
            "resnet18_doub_nobias_nobn": lambda: models.ResNet18(
                num_classes=num_classes, use_batchnorm=False, linear_bias=False, width_ratio=0.5
            ),
            "resnet34": lambda: models.ResNet34(num_classes=num_classes),
            "resnet34_nobias": lambda: models.ResNet34(num_classes=num_classes, linear_bias=False),
            "resnet34_nobn": lambda: models.ResNet34(num_classes=num_classes, use_batchnorm=False),
            "resnet34_nobias_nobn": lambda: models.ResNet34(
                num_classes=num_classes, use_batchnorm=False, linear_bias=False
            ),
            "resnet34_eighth_nobias_nobn": lambda: models.ResNet34(
                num_classes=num_classes, use_batchnorm=False, linear_bias=False, width_ratio=8
            ),
            "resnet34_fourth_nobias_nobn": lambda: models.ResNet34(
                num_classes=num_classes, use_batchnorm=False, linear_bias=False, width_ratio=4
            ),
            "resnet34_half_nobias_nobn": lambda: models.ResNet34(
                num_classes=num_classes, use_batchnorm=False, linear_bias=False, width_ratio=2
            ),
            "resnet34_doub_nobias_nobn": lambda: models.ResNet34(
                num_classes=num_classes, use_batchnorm=False, linear_bias=False, width_ratio=0.5
            ),
        }[config["model"]]()

    if device != -1:
        # model.to(device)
        model = model.cuda(device)
        # logger.info("model parameters are")
        # for param in model.parameters():
        #     logger.info(param.shape)
        if device == "cuda":
            model = torch.nn.DataParallel(model)
            torch.backends.cudnn.benchmark = True

    return model


def get_pretrained_model(config, path, device_id=-1, data_separated=False, relu_inplace=True):
    model = get_model(config, device_id, relu_inplace=relu_inplace)

    if device_id != -1:
        state = torch.load(
            path,
            map_location=(lambda s, _: torch.serialization.default_restore_location(s, "cuda:" + str(device_id))),
        )
    else:
        state = torch.load(
            path,
            map_location=(lambda s, _: torch.serialization.default_restore_location(s, "cpu")),
        )

    model.load_state_dict(state["model_state_dict"])

    if not data_separated:
        logger.info(
            "Loading model at path {} which had accuracy {} and at epoch {}".format(
                path, state["test_accuracy"], state["epoch"]
            )
        )
        return model, state["test_accuracy"], state["epoch"]
    else:
        # currently only support small_big split
        logger.info(
            "Loading model at path {} which had local accuracy {} and overall accuracy {} for choice {} at epoch {}".format(
                path, state["local_test_accuracy"], state["test_accuracy"], None, state["epoch"]
            )
        )
        return model, state["test_accuracy"], state["epoch"], state["local_test_accuracy"], None


def get_retrained_model(
    args, train_loader, test_loader, old_network, config, output_dir, tensorboard_obj=None, nick="", start_acc=-1
):
    # update the parameters
    config["num_epochs"] = args.retrain
    if nick == "geometric":
        nick += "_" + str(args.activation_seed)
    config["nick"] = nick
    config["start_acc"] = start_acc
    if args.retrain_seed != -1:
        config["seed"] = args.retrain_seed

    if args.retrain_lr_decay > 0:
        config["optimizer_learning_rate"] = args.cifar_init_lr / args.retrain_lr_decay
        logger.info("optimizer_learning_rate is %f", config["optimizer_learning_rate"])

    if args.retrain_lr_decay_factor is not None:
        config["optimizer_decay_with_factor"] = args.retrain_lr_decay_factor
        logger.info("optimizer lr decay factor is %f", config["optimizer_decay_with_factor"])

    if args.retrain_lr_decay_epochs is not None:
        config["optimizer_decay_at_epochs"] = [int(ep) for ep in args.retrain_lr_decay_epochs.split("_")]
        logger.info("optimizer lr decay epochs is {}".format(config["optimizer_decay_at_epochs"]))

    # retrain
    best_acc = main(
        config,
        output_dir,
        args.gpu_id,
        pretrained_model=old_network,
        pretrained_dataset=(train_loader, test_loader),
        tensorboard_obj=tensorboard_obj,
    )
    # currently I don' return the best model, as it checkpointed
    return None, best_acc * 100


def store_checkpoint(output_dir, filename, model, epoch, test_accuracy):
    """Store a checkpoint file to the output directory"""
    path = os.path.join(output_dir, filename)

    # Ensure the output directory exists
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    time.sleep(1)  # workaround for RuntimeError('Unknown Error -1') https://github.com/pytorch/pytorch/issues/10577
    torch.save(
        {
            "epoch": epoch,
            "test_accuracy": test_accuracy * 100,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def store_data_separated_checkpoint(output_dir, filename, model, epoch, local_test_accuracy, test_accuracy):
    """Store a checkpoint file to the output directory"""
    path = os.path.join(output_dir, filename)

    # Ensure the output directory exists
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    time.sleep(1)  # workaround for RuntimeError('Unknown Error -1') https://github.com/pytorch/pytorch/issues/10577
    torch.save(
        {
            "epoch": epoch,
            "local_test_accuracy": local_test_accuracy * 100,
            "test_accuracy": test_accuracy * 100,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


if __name__ == "__main__":
    config = dict(
        dataset="Cifar10",
        model="resnet34_nobias_nobn",
        optimizer="SGD",
        optimizer_decay_at_epochs=[150, 250],
        optimizer_decay_with_factor=10.0,
        optimizer_learning_rate=0.1,
        optimizer_momentum=0.9,
        optimizer_weight_decay=0.0001,
        batch_size=256,
        num_epochs=300,
        seed=43,
    )

    output_dir = (
        "./" + str(config["model"]) + "_" + str(config["seed"]) + ".tmp"
    )  # Can be overwritten by a script calling this
    gpu_id = 0

    main(config, output_dir, gpu_id)
