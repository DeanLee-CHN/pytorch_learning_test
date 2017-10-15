#!/usr/bin/env python
# encoding: utf-8

"""
@author: liding
@license: Apache Licence 
@contact: liding2016@ia.ac.cn
@site: 
@file: nn_simple_model.py
@time: 2017/10/15 15:01
"""

import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        x = x.view(-1, self.num_flat_featrues(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_featrues(self, x):
        # print (x.size())
        size = list(x.size()[1 :])
        # print (size)
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print (net)
params = list(net.parameters())
print (len(params))
print (params[0].size())

input_data = Variable(torch.rand(1, 1, 32, 32))
# print (input_data)

output = net(input_data)
print (output)

target = Variable(torch.arange(1, 11))
criterion = nn.MSELoss()

loss = criterion(output, target)
print (loss)

net.zero_grad()
print ('conv1.bias.grad before backward: ', net.conv1.bias.grad)
loss.backward()
print ('conv1.bias.grad after backward: ', net.conv1.bias.grad)

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=False)

optimizer.zero_grad()
optimizer.step()