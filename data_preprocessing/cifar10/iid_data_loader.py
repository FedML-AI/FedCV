import os
import argparse
import time
import math
import logging

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data.distributed import DistributedSampler




def load_iid_cifar10(dataset, data_dir, partition_method, 
        partition_alpha, client_number, batch_size, rank=0):

    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    image_size = 32
    train_transform = transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN , std=CIFAR_STD),
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN , std=CIFAR_STD),
        ])

    train_dataset = CIFAR10(root=data_dir, train=True,
                            transform=train_transform, download=False)

    test_dataset = CIFAR10(root=data_dir, train=False,
                            transform=test_transform, download=False)

    train_sampler = None
    shuffle = True
    if client_number > 1:
        train_sampler = data.distributed.DistributedSampler(
            train_dataset, num_replicas=client_number, rank=rank)
        train_sampler.set_epoch(0)
        shuffle = False

    train_sampler = train_sampler
    train_dl = data.DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=4, sampler=train_sampler)
    test_dl = data.DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=4)
    # classes = ('plane', 'car', 'bird', 'cat',
    #         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    class_num = 10

    train_data_num = len(train_dataset)
    test_data_num = len(test_dataset)

    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        train_data_local_dict[client_idx] = train_dl
        test_data_local_dict[client_idx] = test_dl
        data_local_num_dict[client_idx] = train_data_num // client_number
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, train_data_num))

    return train_data_num, test_data_num, train_dl, test_dl, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num








