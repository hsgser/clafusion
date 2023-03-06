import math
import os

import data
import numpy as np
import torch
from log import logger


def to_dataloader(dataset, bsz):
    return torch.utils.data.DataLoader(dataset, batch_size=bsz, shuffle=True)


def to_dataloader_from_tens(inp, tgt, bsz):
    return to_dataloader(torch.utils.data.TensorDataset(inp, tgt), bsz)


def to_tens_from_dataloader(dataset):
    inp, tgt = [], []

    for batch_x, batch_y in dataset:
        inp.append(batch_x)
        tgt.append(batch_y)

    return (torch.cat(inp), torch.cat(tgt))


def get_personalized_dataset(args):
    file_names = ["personal_train", "other_train", "personal_test", "other_test"]
    datasets = []

    for file_name in file_names:
        file_path = os.path.join(args.load_personalized_datasets, f"{file_name}.pth")
        if os.path.exists(file_path):
            datasets.append(torch.load(file_path))
        else:
            raise FileExistsError(f"File {file_path} does not exist!")

    logger.info(f"Model 0 has on {len(datasets[0][0])} training data and {len(datasets[2][0])} test data")
    logger.info(f"Model 1 has on {len(datasets[1][0])} training data and {len(datasets[3][0])} test data")

    return datasets


def partition_by_labels(args, dataset, label_ids, kind="train"):
    features = []
    labels = []

    for batch, label in dataset:
        sel = torch.zeros_like(label).bool()
        # select indices corresponding to images of the allowed class
        for idx in label_ids:
            sel += label == idx
        features.append(batch[sel])
        labels.append(label[sel])

    features = torch.cat(features)
    labels = torch.cat(labels)
    logger.info("# of instances %d", labels.shape[0])

    if kind == "train":
        bsz = args.batch_size_train
    elif kind == "test":
        bsz = args.batch_size_test

    return to_dataloader(torch.utils.data.TensorDataset(features, labels), bsz)


def split_mnist_by_labels(args, train_loader, test_loader, choice=None):
    if choice is None:
        choice = sorted(np.random.choice(np.arange(0, 10), size=5, replace=False))
    total = sorted(np.arange(0, 10))
    other = np.setdiff1d(total, choice)

    logger.info("First train and test loaders")
    train_loader_a = partition_by_labels(args, train_loader, choice, kind="train")
    if test_loader is not None:
        # this allows for the possibility to only split train loaders
        test_loader_a = partition_by_labels(args, test_loader, choice, kind="test")

    logger.info("Second train and test loaders")
    train_loader_b = partition_by_labels(args, train_loader, other, kind="train")
    if test_loader is not None:
        test_loader_b = partition_by_labels(args, test_loader, other, kind="test")

    if test_loader is not None:
        return (train_loader_a, test_loader_a), (train_loader_b, test_loader_b), other
    else:
        return train_loader_a, train_loader_b, other


def get_shuffled_data(_inp, _tgt):
    assert _inp.shape[0] == _tgt.shape[0]
    num_samples = _inp.shape[0]

    shuffle_idx = torch.randperm(num_samples)
    shuf_inp = _inp[shuffle_idx]
    shuf_tgt = _tgt[shuffle_idx]
    return shuf_inp, shuf_tgt


def get_personalized_split(args, personal_label=9, split_frac=0.1, is_train=True, return_dataset=False):
    inp, tgt = data.get_mnist_dataset(
        root="./data/", is_train=is_train, to_download=args.to_download, return_tensor=True
    )

    # add the examples with target equal to given personal_label in the personal user data
    req_idx = tgt == personal_label
    personal_inp = inp[req_idx]
    personal_tgt = tgt[req_idx]

    logger.info("num of personal class label of {} is {}".format(personal_label, len(personal_inp)))
    other_idx = tgt != personal_label
    other_inp = inp[other_idx]
    other_tgt = tgt[other_idx]

    # other one doesn't contain any labels of particular 'personal_label'
    assert (other_tgt != personal_label).all()

    # shuffle the other labels remaining
    shuf_other_inp, shuf_other_tgt = get_shuffled_data(other_inp, other_tgt)

    # amongst the shuffled, give split_frac
    num_other_labels = math.ceil(split_frac * other_inp.shape[0])
    personal_inp = torch.cat([personal_inp, shuf_other_inp[0:num_other_labels]])
    personal_tgt = torch.cat([personal_tgt, shuf_other_tgt[0:num_other_labels]])

    logger.info("after adding others: num of personal examples is {}".format(len(personal_inp)))

    logger.info("shuffling personal data as well")
    personal_inp, personal_tgt = get_shuffled_data(personal_inp, personal_tgt)

    other_inp = shuf_other_inp[num_other_labels:]
    other_tgt = shuf_other_tgt[num_other_labels:]

    logger.info("num of examples in main is {}".format(other_inp.shape[0]))
    # check if the splits add up to the training dataset
    assert (personal_inp.shape[0] + other_inp.shape[0]) == inp.shape[0]

    if is_train:
        bsz = args.batch_size_train
    else:
        bsz = args.batch_size_test

    if return_dataset:
        return (
            to_dataloader_from_tens(personal_inp, personal_tgt, bsz),
            to_dataloader_from_tens(other_inp, other_tgt, bsz),
            (personal_inp, personal_tgt),
            (other_inp, other_tgt),
        )
    else:
        return to_dataloader_from_tens(personal_inp, personal_tgt, bsz), to_dataloader_from_tens(
            other_inp, other_tgt, bsz
        )


def get_small_big_split(args, split_frac=0.1, is_train=True, return_dataset=False):
    if is_train:
        bsz = args.batch_size_train
    else:
        bsz = args.batch_size_test

    if args.dataset.lower()[0:7] == "cifar10":
        # inp, tgt = data.get_cifar10_dataset(root='./data/',
        #                 is_train=is_train, to_download=args.to_download, return_tensor=True)
        datal_a, datal_b = data.get_train_valid_loader(
            "./data", bsz, is_train=is_train, random_seed=args.partition_train_seed, valid_size=1 - split_frac
        )
        return datal_a, datal_b, to_tens_from_dataloader(datal_a), to_tens_from_dataloader(datal_b)
    elif args.dataset.lower() == "mnist":
        inp, tgt = data.get_mnist_dataset(
            root="./data/", is_train=is_train, to_download=args.to_download, return_tensor=True
        )
    else:
        raise ValueError(f"Dataset {args.dataset} is not supported!")

    # shuffle the input
    logger.info("shuffling data once before splitting")
    shuf_inp, shuf_tgt = get_shuffled_data(inp, tgt)

    # amongst the shuffled, give split_frac
    num_labels = math.ceil(split_frac * inp.shape[0])

    logger.info("splitting data")
    inp_a = shuf_inp[0:num_labels]
    tgt_a = shuf_tgt[0:num_labels]
    logger.info("first model has {} examples".format(inp_a.shape[0]))
    inp_b = shuf_inp[num_labels:]
    tgt_b = shuf_tgt[num_labels:]
    logger.info("second model has {} examples".format(inp_b.shape[0]))
    logger.info("and overall dataset had {} examples".format(inp.shape[0]))
    # check if the splits add up to the training dataset
    assert (inp_a.shape[0] + inp_b.shape[0]) == inp.shape[0]

    if return_dataset:
        return (
            to_dataloader_from_tens(inp_a, tgt_a, bsz),
            to_dataloader_from_tens(inp_b, tgt_b, bsz),
            (inp_a, tgt_a),
            (inp_b, tgt_b),
        )
    else:
        return to_dataloader_from_tens(inp_a, tgt_a, bsz), to_dataloader_from_tens(inp_b, tgt_b, bsz)
