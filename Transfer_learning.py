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
import torch.nn as nn

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

data_dir = '/media/ld/hymenoptera_data/'
# ImageFolder的文档要详查
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                           data_transform[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True,
                                              num_workers=4) for x in ['train', 'val']}
dataset_size = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

# def imshow(inp, title = None):
#     # Imshow for tensor
#
#     inp = inp.numpy().transpose((1, 2, 0))
#     # mean = np.array([0.485, 0.456, 0.406])
#     # std = np.array([0.229, 0.224, 0.225])
#     mean = np.array([1, 0.485, 0.456, 0.406])
#     std = np.array([1, 0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)
#
# # Get a batch of training data
# inputs, classes = next(iter(dataloaders['train']))
# print (inputs.shape, classes.shape)
#
# # Make a grid from batch
# out = torchvision.utils.make_grid(inputs)
# # imshow(inputs, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epoch=1):
    since = time.time()

    best_model_wts = model.state_dict() # .stade_dict()：Seeking the weights of the network?
    best_acc = 0.0

    print (model)
    for epoch in range(num_epoch):
        print ('Epoch {}/{}'.format(epoch, num_epoch - 1))
        print ('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler???Check the doc
                scheduler.step()
                model.train(True)   #set the model to training mode
            if phase == 'val':
                model.train(False)

            running_loss = 0.0
            running_correct = 0

            for data in dataloaders[phase]:
                inputs, labels = data
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                if phase =='train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss
                running_correct += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_correct / dataset_size[phase]

            print ('{} loss: {:.4f}  Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Copy to update the best_wts
            if phase == 'train' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print ('Training Acomplished in {:.0f}h{:.0f}m{:.0f}s'.format(time_elapsed//3600,
                                                                  time_elapsed//60, time_elapsed % 60))
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_imges=6):
    current_images = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, predited = torch.max(outputs, 1)

        for j in range(inputs.size()[0]):
            current_images += 1
            ax = plt.subplot(num_imges//2, 2, current_images)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(classes[predited[j]]))
            imshow(inputs.cpu().data[j])

            if current_images == num_imges:
                return

model_fit = models.resnet18(pretrained=True)
# print (model_fit)
# ???
num_ftrs = model_fit.fc.in_features
model_fit.fc = nn.Linear(num_ftrs, 2)

if use_gpu:
    model_fit = model_fit.cuda()
criterion_fit = torch.nn.CrossEntropyLoss()
optimizer_fit = optim.SGD(model_fit.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_fit, step_size=7, gamma=0.1)

model_fit = train_model(model_fit, criterion_fit, optimizer_fit, exp_lr_scheduler, num_epoch=1)



