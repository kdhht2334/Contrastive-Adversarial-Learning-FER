#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Models of CAL-FER """

import numpy as np
import torch
import torch.nn as nn
import torch.functional as F

from fabulous.color import fg256

import pretrainedmodels
model_name = 'alexnet'  #'resnet18'
alexnet = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None).cuda()

class Encoder2(nn.Module):
    
    def __init__(self):
        super(Encoder2, self).__init__()
        self.features = alexnet._features

    def forward(self, x):
        x = self.features(x)
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
        self.va_regressor = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.flatten(self.avgpool(x), 1)
        x = self.relu0(self.lin0(self.drop0(x)))
        x = self.relu1(self.lin1(self.drop1(x)))

        x = self.va_regressor(x)
        return x

        
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
        return self.lin2(x)
#        return F.sigmoid(self.lin2(x))
#        return 0.5 * torch.tanh(self.lin3(x))  # Hyper-parameter; linear projection


if __name__ == "__main__":
    
    from pytorch_model_summary import summary
    print(fg256("cyan", summary(Encoder2(), torch.ones_like(torch.empty(1, 3, 255, 255)).cuda(), show_input=True)))
    print(fg256("orange", summary(Regressor_light(), torch.ones_like(torch.empty(1, 256, 15, 15)), show_input=True)))
    print(fg256("yellow", summary(Disc2_light(), torch.ones_like(torch.empty(1, 256, 15, 15)), show_input=True)))