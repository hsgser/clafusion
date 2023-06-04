import sys
from basic_config import PATH_TO_CIFAR
sys.path.append(PATH_TO_CIFAR)

import numpy as np
import torch
import train as cifar_train
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms


def get_inp_tar(dataset):
    return dataset.data.view(dataset.data.shape[0], -1).float(), dataset.targets


def get_mnist_dataset(root, is_train, to_download=False, return_tensor=False):
    mnist = datasets.MNIST(
        root,
        train=is_train,
        download=to_download,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    # only 1 channel
                    (0.1307,),
                    (0.3081,),
                ),
            ]
        ),
    )

    if not return_tensor:
        return mnist
    else:
        return get_inp_tar(mnist)


def get_cifar10_dataset(root, is_train, to_download=False, return_tensor=False):
    cifar = datasets.CIFAR10(
        root,
        train=is_train,
        download=to_download,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    # (mean_ch1, mean_ch2, mean_ch3), (std1, std2, std3)
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ]
        ),
    )

    if not return_tensor:
        return cifar
    else:
        # dataset: (num_samples, height, width, num_channels)
        # -> (num_samples, num_channels, height, width)
        reorder_dim = [0, 3, 1, 2]
        return (
            torch.from_numpy(cifar.data).float().permute(*reorder_dim).contiguous(),
            torch.Tensor(cifar.targets).long(),
        )


def get_dataloader(args, unit_batch=False, no_randomness=False):
    if unit_batch:
        bsz = (1, 1)
    else:
        bsz = (args.batch_size_train, args.batch_size_test)

    if no_randomness:
        enable_shuffle = False
    else:
        enable_shuffle = True

    if args.dataset.lower() == "mnist":
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "./data",
                train=True,
                download=args.to_download,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            # only 1 channel
                            (0.1307,),
                            (0.3081,),
                        ),
                    ]
                ),
            ),
            batch_size=bsz[0],
            shuffle=enable_shuffle,
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "./data",
                train=False,
                download=args.to_download,
                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
            ),
            batch_size=bsz[1],
            shuffle=enable_shuffle,
        )

        return train_loader, test_loader

    else:
        if args.cifar_style_data:
            train_loader, test_loader = cifar_train.get_dataset(args.config)
        else:
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(
                    "./data/",
                    train=True,
                    download=args.to_download,
                    transform=transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize(
                                # Note this normalization is not same as in MNIST
                                # (mean_ch1, mean_ch2, mean_ch3), (std1, std2, std3)
                                (0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5),
                            ),
                        ]
                    ),
                ),
                batch_size=bsz[0],
                shuffle=enable_shuffle,
            )

            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(
                    "./data/",
                    train=False,
                    download=args.to_download,
                    transform=transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize(
                                # (mean_ch1, mean_ch2, mean_ch3), (std1, std2, std3)
                                (0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5),
                            ),
                        ]
                    ),
                ),
                batch_size=bsz[1],
                shuffle=enable_shuffle,
            )

        return train_loader, test_loader


"""Adapted from https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb"""


def get_train_valid_loader(
    data_dir,
    batch_size,
    is_train=True,
    random_seed=0,
    augment=False,
    valid_size=0.1,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - is_train: whether to load train or test dataset.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert (valid_size >= 0) and (valid_size <= 1), error_msg

    # normalize = transforms.Normalize(
    #     mean=[0.5, 0.5, 0.5],
    #     std=[0.5, 0.5, 0.5],
    # )

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=is_train,
        download=True,
        transform=train_transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir,
        train=is_train,
        download=True,
        transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)
