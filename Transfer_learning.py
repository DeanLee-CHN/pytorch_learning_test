#!/usr/bin/env python
# encoding: utf-8

"""
@author: liding
@license: Apache Licence 
@contact: liding2016@ia.ac.cn
@site: 
@file: Transfer_learning.py
@time: 2017/10/17 17:48
"""

from __future__ import print_function, division

import numpy as np
import torch
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os

plt.ion()   # iterative mode

"""
Data augmentation and normalization for training
Just normalize for validation
:return: dataset for training
"""
data_transform = {
    'train': transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    }

data_dir = '/media/ld/hymenoptera_data/hymenoptera_data/'
# ImageFolder的文档要详查
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                           data_transform[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True,
                                              num_workers=4) for x in ['train', 'val']}
dataset_size = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

def imshow(inp, title = None):
    # Imshow for tensor

    inp = inp.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    mean = np.array([1, 0.485, 0.456, 0.406])
    std = np.array([1, 0.229, 0.224, 0.225])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))
print (inputs.shape, classes.shape)

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(inputs, title=[class_names[x] for x in classes])