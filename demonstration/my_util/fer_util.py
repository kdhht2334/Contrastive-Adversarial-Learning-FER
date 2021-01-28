#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: KDH
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.functional as F 

import pretrainedmodels
import pretrainedmodels.utils as utils


model_name = 'alexnet'  # 'bninception'
#resnext = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet').cuda()
alexnet = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')#.cuda()


def _encoder2():
    encoder2 = Encoder2()
    return encoder2
def _regressor():
    regressor2 = Regressor_light()
    return regressor2
def _disc2():
    disc2 = Disc2_light()
    return disc2


def nn_output():
    encoder2  = _encoder2() #.cuda()
    regressor = _regressor() #.cuda()
    disc2     = _disc2() #.cuda()
    return encoder2, regressor, disc2


class Encoder2(nn.Module):
    
    def __init__(self):
        super(Encoder2, self).__init__()
        
        self.features = alexnet._features

    def forward(self, x):
        x = self.features(x)
        return x
    

class Regressor(nn.Module):
    
    def __init__(self):
        super(Regressor, self).__init__()
        
        self.avgpool = alexnet.avgpool
        self.lin0 = alexnet.linear0
        self.lin1 = alexnet.linear1
        self.relu0 = alexnet.relu0
        self.relu1 = alexnet.relu1
        self.drop0 = alexnet.dropout0
        self.drop1 = alexnet.dropout0
        self.last_linear = alexnet.last_linear
        self.va_regressor = nn.Linear(1000, 2)
        
    def forward(self, x):
        x = torch.flatten(self.avgpool(x), 1)
        x = self.relu0(self.lin0(self.drop0(x)))
        x = self.relu1(self.lin1(self.drop1(x)))
        
        x = self.last_linear(x)
        x = self.va_regressor(x)
        return x


class Regressor_light(nn.Module):

    def __init__(self):
        super(Regressor_light, self).__init__()

        self.avgpool = alexnet.avgpool
        self.lin0 = nn.Linear(9216, 64)
        self.lin1 = nn.Linear(64, 8)
        self.relu0 = alexnet.relu0
        self.relu1 = alexnet.relu1
        self.drop0 = alexnet.dropout0
        self.drop1 = alexnet.dropout0
#        self.last_linear = alexnet.last_linear
        self.va_regressor = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.flatten(self.avgpool(x), 1)
        x = self.relu0(self.lin0(self.drop0(x)))
        x = self.relu1(self.lin1(self.drop1(x)))

#        x = self.last_linear(x)
        x = self.va_regressor(x)
        return x

        
        
class Disc2(nn.Module):
    
    def __init__(self):
        super(Disc2, self).__init__()
        
        self.avgpool = alexnet.avgpool
        self.last_linear = nn.Linear(9216, 1000)  # resnext.last_linear

        self.lin1 = nn.Linear(1000, 256)
        self.lin2 = nn.Linear(256, 32)
        self.lin3 = nn.Linear(32, 16)  # Hyper-parameter; the original value is 1 (binary)

    def forward(self, x):
        x = torch.flatten(self.avgpool(x), 1)
        x = self.last_linear(x)
        
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return F.sigmoid(self.lin3(x))
#        return 0.5 * torch.tanh(self.lin3(x))  # Hyper-parameter; linear projection


class Disc2_light(nn.Module):
    
    def __init__(self):
        super(Disc2_light, self).__init__()
        
        self.avgpool = alexnet.avgpool
        self.last_linear = nn.Linear(9216, 64)  # resnext.last_linear

        self.lin1 = nn.Linear(64, 32)
        self.lin2 = nn.Linear(32, 16)

    def forward(self, x):
        x = torch.flatten(self.avgpool(x), 1)
        x = self.last_linear(x)
        
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        return F.sigmoid(self.lin2(x))
#        return 0.5 * torch.tanh(self.lin3(x))  # Hyper-parameter; linear projection
