#!/usr/bin/env python
# encoding: utf-8

"""
@author: liding
@license: Apache Licence
@contact: liding2016@ia.ac.cn
@file: classifier_cifar10.py
@time: 2017/10/16 9:33
"""

import os
import torch
import torchvision
from torch import nn
import torchvision.transforms as transforms
from torch import utils

from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

# torch.cuda.set_device(id)

# Transformation of data
# transform的normalize中参数为何是两组？
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(
    root='/media/ld/cifar10/trainset',
    train=True,
    transform=transform,
    download=True)
if not os.path.exists('/media/ld/cifar10/trainset'):
    os.mkdir('/media/ld/cifar10/trainset')
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2)

testet = torchvision.datasets.CIFAR10(
    root='/media/ld/cifar10/testset',
    train=True,
    transform=transform,
    download=True)
if not os.path.exists('/media/ld/cifar10/testset'):
    os.mkdir('/media/ld/cifar10/testset')
testloader = torch.utils.data.DataLoader(
    testet, batch_size=4, shuffle=True, num_workers=2)

classes = (
    'plane',
    'car',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck')


class Net(nn.Module):
    """
    Define the structure of the neural network and the process of the forward-computing
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, (5, 5))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Write the process of forward_computing block by block
        :param x:input img
        :return: Final output of neural network
        """
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


net = Net()
net.cuda()
print (net)

# Define the loss
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Process of training
for epoch in range(4):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # wrap the inputs and labels in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 1000 == 999:
            print ('[%d, %5d] loss: %.3f' %
                   (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

print('Finished Training')

# # test_test
# test_dataiter = iter(testloader)
# test_images, test_labels = test_dataiter.next()
# print('GroundTruth: ', ' '.join('%5s' % classes[test_labels[j]] for j in range(4)))
# test_outputs = Variable(net(test_images))
#
# _, predicted = torch.max(test_outputs, 1)
# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

for data in testloader:
    images, labels = data
    images = Variable(images.cuda())
    outputs = net(images)
    _, predited = torch.max(outputs.data, 1)
    # total += labels.size(0)
    # correct += sum(predited == labels)
    # print ('predicted:', predited, type(predited))
    # print ('labels:', labels, type(labels.cuda()))
    # print ((predited == labels))
    # c = (predited == labels).squeeze()

    for i in range(4):
        current_label = labels[i]
        current_prediction = predited[i]
        if (current_label == current_prediction):
            class_correct[current_label] += 1
        class_total[current_label] += 1

for i in range(10):
    print (
        'Accuracy of %5s: %2d %%' %
        (classes[i],
         100 *
         class_correct[i] /
         class_total[i]))
# print ('Accuracy of the network on 10000 test images: %d %%' % (100 * correct / total))
