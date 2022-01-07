# Copyright 2021 Ramon Casero 
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
"""Load data for neural networks.

This module contains functions to load and process input data for the neural networks.

Functions:
    dataloader_FashionMNIST: Download and prepare FashionMNIST data to be used by the neural networks.
"""

import os
import numpy as np
import torch
import torchvision


def dataloader_FashionMNIST(data_dir=os.path.join(os.curdir, 'data'), batch_size=64, validation_ratio=0.0):
    """Download and prepare FashionMNIST data to be used by the neural networks.

    FashionMNIST details: https://github.com/zalandoresearch/fashion-mnist

    1) Download FashionMNIST data from the internet, if not locally cached.
    2) Split data into training/validation/testing datasets.
    3) Create dataloaders for training/validating/testing neural networks (nn.Module).
    :param data_dir: (str, def './data') Path to directory to download/read the data.
    :param batch_size: (int, def 64) Batch size. If the number of images is not a multiple of batch_size, the last batch
    will be smaller than the others.
    :param validation_ratio: (float, def 0.0) Float with ratio of training data used for validation. E.g.
    validation_ratio=0.2 means that 80% will be used for training and 20% for validation.
    :return:
    * train_dataloader: (torch.utils.data.DataLoader) dataloader for training dataset.
    * validate_dataloader: (torch.utils.data.DataLoader) dataloader for validation dataset.
    * test_dataloader: (torch.utils.data.DataLoader) dataloader for testing dataset.
    * class_to_idx: (dict) Class labels and their indices (e.g. {'T-shirt/top': 0, 'Trouser': 1, ...).
    * (H, W): (tuple) height and width of the training/validation/testing images.
    """

    # training and testing data will be downloaded from the web in the first run, and cached for subsequent runs
    trainset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    testset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )

    # sanity check: dictionaries of labels/numerical indices are the same in training and test datasets
    assert (trainset.class_to_idx == testset.class_to_idx)

    # split the training dataset into training and validation
    idx = list(range(len(trainset)))
    np.random.shuffle(idx)
    split = int(np.floor(validation_ratio * len(trainset)))
    train_sample = torch.utils.data.sampler.SubsetRandomSampler(idx[split:])
    validate_sample = torch.utils.data.sampler.SubsetRandomSampler(idx[:split])

    _, train_height, train_width = trainset.data.shape
    test_n, test_height, test_width = testset.data.shape

    # assert: training/validation and test images have the same 2D size
    assert (train_width == test_width)
    assert (train_height == test_height)

    # dataloaders for training the network
    train_dataloader = torch.utils.data.DataLoader(trainset, sampler=train_sample, batch_size=batch_size)
    validate_dataloader = torch.utils.data.DataLoader(trainset, sampler=validate_sample, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    return train_dataloader, validate_dataloader, test_dataloader, trainset.class_to_idx, (train_height, train_width)
