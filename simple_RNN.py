#!/usr/bin/env python
# encoding: utf-8

"""
@author: liding
@license: Apache Licence 
@contact: liding2016@ia.ac.cn 
@file: simple_RNN.py
@time: 2017/10/17 14:53
"""
import torch
import torchvision
from torch import nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, data_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size= output_size
        input_size = data_size + hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, data, last_hidden):
        input_data = torch.cat((data, last_hidden), 1)
        hidden = self.i2h(input_data)
        output = self.h2o(hidden)
        return hidden, output

rnn = RNN(50, 20, 10)
rnn.cuda()
print (rnn)
loss_fn = nn.MSELoss()

batch_size = 10
TIMESTEPS = 5

# Initialize the node in encoder and decoder
batch = Variable(torch.randn(batch_size, 50).cuda())
hidden = Variable(torch.zeros(batch_size, 20).cuda())
target = Variable(torch.zeros(batch_size, 10).cuda())

loss = 0
for t in range(TIMESTEPS):
    hidden, output = rnn(batch, hidden)
    loss += loss_fn(output, target)
    # loss += loss_fn(output)
loss.backward()