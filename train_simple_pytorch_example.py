#!/usr/bin/env python

# Copyright 2021 Ramon Casero 
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
"""Python script to train/validate/test neural network classifiers on FashionMNIST.

This script trains a neural network to solve the FashionMNIST multi-class image classification problem. Network and
training parameters can be provided in a config file and/or the command line. The training data can optionally be split
between training and validation datasets. The script can display verbose information to the terminal when it is running,
and it also creates a TensorBoard log with accuracy, mean loss and confusion matrix metrics.

usage: train_simple_pytorch_example.py [-h] [-c CONFIG_FILE] [-v] [--workdir DIR] [-d STR] [-e N] [-b N] [-l F] [--validation_ratio F] [-n STR]
                                       [--conv_out_features N [N ...]] [--conv_kernel_size N] [--maxpool_kernel_size N]

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_FILE, --config CONFIG_FILE
                        config file path
  -v, --verbose         verbose output for debugging
  --workdir DIR         working directory to place data, logs, weights, etc subdirectories (def .)
  -d STR, --device STR  device to train on (def 'cuda', 'cpu')
  -e N, --epochs N      number of epochs for training (def 10)
  -b N, --batch_size N  batch size for training (def 64)
  -l F, --learning_rate F
                        learning rate for training (def 1e-3)
  --validation_ratio F  ratio of training dataset reserved for validation (def 0.0)
  -n STR, --nn STR      neural network architecture (def 'SimpleCNN', 'SimpleLinearNN')
  --conv_out_features N [N ...]
                        (SimpleCNN only) number of output features for each convolutional block (def 8 16)
  --conv_kernel_size N  (SimpleCNN only) kernel size of convolutional layers (def 3)
  --maxpool_kernel_size N
                        (SimpleCNN only) kernel size of max pool layers (def 2)

Args that start with '--' (eg. -v) can also be set in a config file (specified via -c). Config file syntax allows: key=value, flag=true,
stuff=[a,b,c] (for details, see syntax at https://goo.gl/R74nmi). If an arg is specified in more than one place, then commandline values override
config file values which override defaults.
"""

import os
import datetime
import socket
import matplotlib.pyplot as plt
import sklearn.metrics
import configargparse
import sty
import torch
import torch.utils.tensorboard
import torchvision
import torchinfo
import simplexample.data
import simplexample.nn


def main():

    ##################################################################################################################
    # parsing input arguments
    ##################################################################################################################

    # parser
    # (API: https://bw2.github.io/ConfigArgParse/configargparse.ArgumentParser.html)
    parser = configargparse.ArgumentParser(default_config_files=[], ignore_unknown_config_file_keys=True,
                                           args_for_setting_config_path=['-c', '--config'],
                                           config_arg_is_required=False)
    parser.add('-v', '--verbose', action='store_true',
               help='verbose output for debugging')
    parser.add('--workdir', metavar='DIR', default=os.path.curdir, type=str,
               help='working directory to place data, logs, weights, etc subdirectories (def ' + os.path.curdir + ')')
    parser.add('-d', '--device', metavar='STR', default='cuda', type=str,
               help='device to train on (def \'cuda\', \'cpu\')')
    parser.add('-e', '--epochs', metavar='N', default=10, type=int,
               help='number of epochs for training (def 10)')
    parser.add('-b', '--batch_size', metavar='N', default=64, type=int,
               help='batch size for training (def 64)')
    parser.add('-l', '--learning_rate', metavar='F', default=1e-3, type=float,
               help='learning rate for training (def 1e-3)')
    parser.add('--validation_ratio', metavar='F', default=0.0, type=float,
               help='ratio of training dataset reserved for validation (def 0.0)')
    parser.add('-n', '--nn', metavar='STR', default='SimpleCNN', type=str, choices=['SimpleCNN', 'SimpleLinearNN'],
               help='neural network architecture (def \'SimpleCNN\', \'SimpleLinearNN\')')
    parser.add('--conv_out_features', metavar='N', nargs='+', default=[8, 16], type=int,
               help='(SimpleCNN only) number of output features for each convolutional block (def 8 16)')
    parser.add('--conv_kernel_size', metavar='N', default=3, type=int,
               help='(SimpleCNN only) kernel size of convolutional layers (def 3)')
    parser.add('--maxpool_kernel_size', metavar='N', default=2, type=int,
               help='(SimpleCNN only) kernel size of max pool layers (def 2)')

    args = parser.parse_args()

    # experiment ID string to use for filenames (logs, parameters,
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_id = current_time + '_' + socket.gethostname() + '_' + args.nn

    # debug: input arguments
    if args.verbose:
        print(sty.fg.red + '** Experiment ID:' + sty.fg.rs)
        print(experiment_id)
        print(sty.fg.red + '** All args:' + sty.fg.rs)
        print(args)
        print(sty.fg.red + '** Arg breakdown (defaults / config file / command line):' + sty.fg.rs)
        print(parser.format_values())

    # check for GPU if needed
    if args.device == 'cuda' and torch.cuda.is_available():
        if args.verbose:
            print(sty.fg.red + '** GPU found:' + sty.fg.rs)
            print(torch.cuda.get_device_name())
    elif args.device == 'cuda':
        raise SystemError('GPU not found. Try option --device \'cpu\'')

    # check that working directory exists and we can write to it
    if not os.path.isdir(args.workdir) or not os.access(args.workdir, os.W_OK):
        raise NotADirectoryError(args.workdir + ' is not a valid writable directory')

    ##################################################################################################################
    # input data
    ##################################################################################################################

    # download FashionMNIST data if not cached, split into train/test/validate datasets and create dataloaders
    train_dataloader, validate_dataloader, test_dataloader, class_to_idx, im_size = \
        simplexample.data.dataloader_FashionMNIST(data_dir=os.path.join(args.workdir, 'data'), batch_size=args.batch_size,
        validation_ratio=args.validation_ratio)
    class_idx = list(class_to_idx.values())  # numerical labels for the class labels (needed for confusion mat)
    class_labels = list(class_to_idx.keys())  # text labels for the class labels

    train_num = len(train_dataloader.sampler)
    validate_num = len(validate_dataloader.sampler)
    test_num = len(test_dataloader.sampler)

    # debug: training and test dataset sizes
    if args.verbose:
        print(sty.fg.red + '** Datasets:' + sty.fg.rs)
        print('Image size (H, W): ' + str(im_size))
        print('Training samples: ' + str(train_num))
        print('Validation samples: ' + str(validate_num))
        print('Testing samples: ' + str(test_num))
        print('Classes: ' + str(class_to_idx))

    ##################################################################################################################
    # neural network
    ##################################################################################################################

    # instantiate network
    if args.nn == 'SimpleLinearNN':
        model = simplexample.nn.SimpleLinearNN(input_size=im_size)
    elif args.nn == 'SimpleCNN':
        model = simplexample.nn.SimpleCNN(input_size=im_size, conv_out_features=args.conv_out_features,
                                    conv_kernel_size=args.conv_kernel_size,
                                    maxpool_kernel_size=args.maxpool_kernel_size)
    else:
        # redundant check (already checked by parser)
        raise ValueError('Invalid neural network architecture: ' + args.nn)

    model = model.to(args.device)
    if args.verbose:
        print(sty.fg.red + '** Neural network architecture:' + sty.fg.rs)
        print(torchinfo.summary(model, input_size=(1, 1) + im_size, device=args.device, verbose=0))

    ##################################################################################################################
    # training
    ##################################################################################################################

    # training parameters
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    # open TensorBoard log
    tb = torch.utils.tensorboard.SummaryWriter(os.path.join(args.workdir, 'runs', experiment_id))

    if args.verbose:
        print(sty.fg.red + '** Training:' + sty.fg.rs)

    # training loop
    for epoch in range(args.epochs):

        if epoch == 0:
            # get one batch for display on TensorBoard
            train_features, train_labels = next(iter(train_dataloader))
            train_features = train_features.to(args.device)
            grid = torchvision.utils.make_grid(train_features)
            tb.add_image('training images', grid)
            tb.add_graph(model, train_features)

        if args.verbose:
            print(f'Epoch {epoch + 1}/{args.epochs}\n-------------------------------')

        # training and testing loops
        train_metrics = simplexample.nn.train_loop(args, train_dataloader, model, loss_fn, optimizer)
        test_metrics = simplexample.nn.test_loop(args, test_dataloader, model, loss_fn, class_idx=class_idx)
        validate_metrics = simplexample.nn.test_loop(args, validate_dataloader, model, loss_fn, class_idx=class_idx)

        if args.verbose:
            print(f'Training: '
                  + f'Mean loss: {train_metrics["mean_loss"]:>6.4f}')
            print(f'Test: '
                  + f'Accuracy: {(100 * test_metrics["accuracy"]):>0.1f}%, '
                  + f'Mean loss: {test_metrics["mean_loss"]:>6.4f}')
            print(f'Validation: '
                  + f'Accuracy: {(100 * validate_metrics["accuracy"]):>0.1f}%, '
                  + f'Mean loss: {validate_metrics["mean_loss"]:>6.4f}')

        # add another epoch to TensorBoard accuracy and loss plots
        tb.add_scalars('Mean loss ' + experiment_id,
                       {'train': train_metrics['mean_loss'],
                        'test': test_metrics['mean_loss'],
                        'validate': validate_metrics['mean_loss']}, global_step=epoch + 1)
        tb.add_scalars('Accuracy (%) ' + experiment_id,
                       {'test': 100 * test_metrics['accuracy'],
                        'validate': 100 * validate_metrics['accuracy']}, global_step=epoch + 1)

        # add confusion matrices for current epoch to TensorBoard log
        sklearn.metrics.ConfusionMatrixDisplay(test_metrics['cm'], display_labels=class_labels).plot()
        plt.xticks(range(len(class_labels)), class_labels, rotation=45)
        plt.tight_layout()
        tb.add_figure('Test confusion matrix', plt.gcf(), global_step=epoch + 1, close=True)

        sklearn.metrics.ConfusionMatrixDisplay(validate_metrics['cm'], display_labels=class_labels).plot()
        plt.xticks(range(len(class_labels)), class_labels, rotation=45)
        plt.tight_layout()
        tb.add_figure('Validation confusion matrix', plt.gcf(), global_step=epoch + 1, close=True)

    # close TensorBoard log
    tb.close()

    # save model, creating directories if necessary
    models_dir = os.path.join(args.workdir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(models_dir, experiment_id + '.state_dict'))


if __name__ == '__main__':
    main()
