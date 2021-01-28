#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Utils of CAL-FER """

import numpy as np

import torch

def cov(m, y=None):
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov


def pcc_ccc_loss(labels_th, scores_th):

    std_l_v = torch.std(labels_th[:,0]); std_p_v = torch.std(scores_th[:,0])
    std_l_a = torch.std(labels_th[:,1]); std_p_a = torch.std(scores_th[:,1])
    mean_l_v = torch.mean(labels_th[:,0]); mean_p_v = torch.mean(scores_th[:,0])
    mean_l_a = torch.mean(labels_th[:,1]); mean_p_a = torch.mean(scores_th[:,1])
   
    PCC_v = torch.mean( (labels_th[:,0] - mean_l_v) * (scores_th[:,0] - mean_p_v) ) / (std_l_v * std_p_v)
    PCC_a = torch.mean( (labels_th[:,1] - mean_l_a) * (scores_th[:,1] - mean_p_a) ) / (std_l_a * std_p_a)
#    PCC_v = torch.mean( (labels_th[:,0] - mean_l_v).t() @ (scores_th[:,0] - mean_p_v)/(std_l_v * std_p_v) )
#    PCC_a = torch.mean( (labels_th[:,1] - mean_l_a).t() @ (scores_th[:,1] - mean_p_a)/(std_l_a * std_p_a) )
    CCC_v = (2.0 * std_l_v * std_p_v * PCC_v) / ( std_l_v.pow(2) + std_p_v.pow(2) + (mean_l_v-mean_p_v).pow(2) )
    CCC_a = (2.0 * std_l_a * std_p_a * PCC_a) / ( std_l_a.pow(2) + std_p_a.pow(2) + (mean_l_a-mean_p_a).pow(2) )
   
    PCC_loss = 1.0 - (PCC_v + PCC_a)/2
    CCC_loss = 1.0 - (CCC_v + CCC_a)/2
    return PCC_loss, CCC_loss, CCC_v, CCC_a


def cumulative_thresholding(d, thr1, thr2):

    d = torch.from_numpy(d).type(torch.FloatTensor)  # numpy -> torch.tensor
    i_s = torch.argmin(d)
    i_e = torch.argmax(d)
    i_m = (i_s+i_e)//2
    _bin = np.shape(d)[0]
    aaa = 1

    while aaa:
        w_l = torch.sum(d[0:i_m+1])
        w_r = torch.sum(d[i_m+1:i_e+1])
        w   = torch.sum(d)

        if w_l >= w*thr1 or w_l < w*thr2:
            aaa = 0
#            return d[0:i_m+1].sum()/_bin
            return d[0:i_m+1].sum()/w
        else:
            i_m += 2
            aaa = 1
            
            
def mutual_info(x, y):
    var = 0.2
    p_y_x=torch.exp(-(y-x)**2/(2*var))
    p_y_x_minus=torch.exp(-(y+1)**2/(2*var))
    p_y_x_plus=torch.exp(-(y-1)**2/(2*var))
    return torch.mean(torch.log(p_y_x/(0.5*p_y_x_minus+0.5*p_y_x_plus)))