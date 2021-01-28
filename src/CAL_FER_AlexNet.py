#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAL-FER on Aff-Wild dataset

@author: kdhht5022@gmail.com
@reference1: A. Srivastava et al. Generative ratio matching networks, ICLR, 2020.
@reference2: MI. Belghazi et al. Mutual Information Neural Estimation, ICML, 2018.
@reference3: A. Balsubramani et al. An Adaptive Nearest Neighbor Ruel for Classification, NeurIPS, 2019.
"""
from __future__ import print_function, division
import os

import argparse
app = argparse.ArgumentParser()
app.add_argument("-g", "--gpus", type=int, default=3, help='Which GPU do you want for training.')
app.add_argument("-t", "--train", type=int, default=1, help='Training vs. evaluation phase.')
app.add_argument("-f", "--freq", type=int, default=1, help='Saving frequency.')
args = vars(app.parse_args())

gpus = args["gpus"]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus)

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F 

from fabulous.color import fg256
import wandb
from aknn_alg import aknn, calc_nbrs_exact

from utils import pcc_ccc_loss, cumulative_thresholding, mutual_info
from models import Encoder2, Regressor_light, Disc2_light

import warnings
warnings.filterwarnings("ignore")

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:{}".format(str(gpus)))


def model_training(model, metric, optimizer, scheduler, num_epochs):

    wandb.init(project="Aff-wild_Fisher_aknn_cont_alex_first_trial")
    
    Lambda = torch.FloatTensor([0]).cuda()  # lagrange multipliers
    Lambda = Variable(Lambda, requires_grad=True)  # Trained via artisnal SGD

    enc_opt = optimizer[0]
    reg_opt = optimizer[1]
    disc_opt = optimizer[2]

    RMSE = metric[0]
    rho = 1e-6

    for epoch in range(num_epochs):
        print('epoch ' + str(epoch) + '/' + str(num_epochs-1))

        encoder = model[0]; regressor = model[1]; disc = model[2]
        
        enc_opt.step()
        reg_opt.step()
        disc_opt.step()

        for batch_i, data_i in enumerate(loaders['train']):
            
            hard_idx, weak_idx = [], []
            data, emotions = data_i['image'], data_i['va']
            valence = np.expand_dims(np.asarray(emotions[0]), axis=1)  # [64, 1]
            arousal = np.expand_dims(np.asarray(emotions[1]), axis=1)
            emotions = torch.from_numpy(np.concatenate([valence, arousal], axis=1)).float()

            th_v, th_a = cumulative_thresholding(valence, 0.75, 0.95), cumulative_thresholding(arousal, 0.75, 0.95)
            th = torch.sqrt(th_v.pow(2) + th_a.pow(2))
            offset = 0.25

            # Emotion grouping
            for i in range(emotions.size()[0]):
                if emotions[i][0].pow(2) + emotions[i][1].pow(2) > th + offset:  # > 15.0:
                    hard_idx.append(i)
                elif emotions[i][0].pow(2) + emotions[i][1].pow(2) < th + offset:  # < 10.0:
                    weak_idx.append(i)

            # Technical part (if we cannot find meaningful grouping indices)
            if len(hard_idx) == 0 or len(weak_idx) == 0:
                weak_idx = []  # renew
                for i in range(emotions.size()[0]):
                    if emotions[i][0].pow(2) + emotions[i][1].pow(2) > 0.65:
                        hard_idx.append(i)
                    elif emotions[i][0].pow(2) + emotions[i][1].pow(2) < 0.3:
                        weak_idx.append(i)
            if len(hard_idx) == 0: hard_idx = weak_idx

            hard_idx = hard_idx[:min(len(weak_idx), len(hard_idx))]
            weak_idx = weak_idx[:min(len(weak_idx), len(hard_idx))]
            hard_idx, weak_idx = torch.tensor([hard_idx]), torch.tensor([weak_idx])

            if use_gpu:
                inputs, correct_labels = Variable(data.cuda()), Variable(emotions.cuda())
            else:
                inputs, correct_labels = Variable(data), Variable(emotions)

            # ---------------
            # Train regressor
            # ---------------
            z = encoder(inputs)
            scores = regressor(z)

            pcc_loss, ccc_loss, ccc_v, ccc_a = pcc_ccc_loss(correct_labels, scores)
            RMSE_valence = RMSE(scores[:,0], correct_labels[:,0])
            RMSE_arousal = RMSE(scores[:,1], correct_labels[:,1])
            loss = (RMSE_valence + RMSE_arousal) + 0.5 * pcc_loss + 0.5 * ccc_loss
            
            enc_opt.zero_grad(); reg_opt.zero_grad()
            disc_opt.zero_grad(); 
            
            loss.backward(retain_graph=True)
            enc_opt.step()
            reg_opt.step()
            
            # -----------
            # Train disc.
            # -----------
            for p in disc.parameters(): p.requires_grad = True

            msize = hard_idx.size(1)//2

            z_high_emo = encoder(inputs[hard_idx].squeeze(0))
            z_low_emo = encoder(inputs[weak_idx].squeeze(0))
            
            if z_high_emo.size(0) == 1 or z_low_emo.size(0) == 1:
                D_hh_emo = disc(z_high_emo)
                D_ll_emo = disc(z_low_emo)
            else:
                D_hh_emo = disc(z_high_emo[:msize*2])
                D_ll_emo = disc(z_low_emo[:msize*2])

            # To check whether is nan value or not
            if torch.sum(D_hh_emo != D_hh_emo):
                print(fg256("red", "`NaN` value occured!"))
                D_hh_emo = torch.ones_like(D_hh_emo)
            else: pass

            if torch.sum(D_ll_emo != D_ll_emo):
                print(fg256("red", "`NaN` value occured!"))
                D_ll_emo = torch.zeros_like(D_ll_emo)
            else: pass

            mi = mutual_info(D_hh_emo, D_ll_emo)
            mi = torch.clamp(mi, 0.0, 2.5)  # rough clamping for training stability

            #---------------------------------------------------------------
            # Adaptive lower bound for choosing suitable `k` in metric balls
            #---------------------------------------------------------------
            nmn = torch.cat([D_hh_emo, D_ll_emo], dim=0).cpu().detach().numpy()
            nbrs_list = calc_nbrs_exact(nmn, k=10)
            confidence = 0.7 / np.sqrt(np.arange(nbrs_list.shape[1])+1)  # 0.75

            adaptive_ks = []
            labels = np.concatenate([np.ones(D_hh_emo.size(0)), np.zeros(D_ll_emo.size(0))], axis=0)
            for i in range(nbrs_list.shape[0]):
                (_, adaptive_k_ndx, _) = aknn(nbrs_list[i,:], labels, confidence)
                adaptive_ks.append(adaptive_k_ndx + 1)
            adaptive_ks = np.array(adaptive_ks)
            confidence = torch.from_numpy(confidence).float().cuda()

            d_ap, d_an = [], []
            minfo = []
            for i in range(0, msize):
                mi = torch.clamp(mutual_info(D_hh_emo[i], D_ll_emo[-i-1]), 0.0, 2.5)
                pos_dist = F.relu( ((D_hh_emo[i]-D_hh_emo[-i-1]).pow(2)).sum() )
                neg_dist = F.relu( mi + confidence[adaptive_ks[i]-2] - ((D_hh_emo[i]-D_ll_emo[-i-1]).pow(2)).sum() )
                d_ap.append(pos_dist); d_an.append(neg_dist); minfo.append(mi)
            for i in range(msize,D_hh_emo.size(0)):
                mi = torch.clamp(mutual_info(D_hh_emo[i], D_ll_emo[-i-1]), 0.0, 2.5)
                pos_dist = F.relu( ((D_hh_emo[i]-D_hh_emo[-i-1]).pow(2)).sum() )
                neg_dist = F.relu( mi + confidence[adaptive_ks[i]-2] - ((D_hh_emo[i]-D_ll_emo[-i-1]).pow(2)).sum() )
                d_ap.append(pos_dist); d_an.append(neg_dist); minfo.append(mi)

            d_ap, d_an = torch.stack(d_ap), torch.stack(d_an)
            minfo = torch.mean(torch.stack(minfo))
            wandb.log({"Mutual information": minfo.cpu()})

            if torch.sum(d_ap != d_ap):  # To check whether is nan value or not
                print(fg256("red", "`NaN` value occured!"))
                d_ap = D_hh_emo
                d_an = D_ll_emo
            else:
                wandb.log({
                    "epoch": epoch, "loss": loss, "RMSE (v)": RMSE_valence, "RMSE (a)": RMSE_arousal,
                    "d_ap": d_ap.cpu(), "d_an": d_an.cpu(), "threshold": th,
                    "CCC_v": ccc_v, "CCC_a": ccc_a
                })

            E_P_f,  E_Q_f  = d_ap.mean(), d_an.mean()
            E_P_f2, E_Q_f2 = (d_ap**2).mean(), (d_an**2).mean()
            constraint = (1 - (0.5*E_P_f2 + 0.5*E_Q_f2))

            D_loss = E_P_f + E_Q_f + Lambda * constraint - rho/2 * constraint**2
            D_loss.backward()
            disc_opt.step()

            # artisnal sgd. We minimize Lambda so Lambda <- Lambda + lr * (-grad) (FGAN, NeurIPS 2020)
            Lambda.data += rho * Lambda.grad.data
            Lambda.grad.data.zero_()
            wandb.log({"Lambda (art. sgd)": Lambda.cpu()})


            # ---------------------------
            # Train encoder (adversarial)
            # ---------------------------
            for p in disc.parameters(): p.requires_grad = False
            encoder.zero_grad()
            enc_opt.zero_grad()

            fake_z = encoder(inputs[weak_idx].squeeze(0))
            fake_emo = disc(fake_z)
            fake_loss = -torch.mean(fake_emo) * 0.1
            fake_loss.backward()
            enc_opt.step()

            wandb.log({
                "D_loss": D_loss.cpu(), "G_loss": fake_loss.cpu()
            })

            scheduler[0].step(); scheduler[1].step(); scheduler[2].step()

            del hard_idx, weak_idx
            
        if epoch % args["freq"] == 0 and epoch > 0:
            torch.save(encoder.state_dict(), '/path/to/enc_weights_{}.t7'.format(epoch))
            torch.save(regressor.state_dict(), '/path/to/reg_weights_{}.t7'.format(epoch))
            torch.save(disc.state_dict(), '/path/to/dec_weights_{}.t7'.format(epoch))


def model_evaluation(model, metric, num_epochs):
    
    encoder = model[0]; regressor = model[1]
    RMSE = metric[0]
    encoder.load_state_dict(torch.load('/path/to/enc_weights.t7'), strict=False)
    regressor.load_state_dict(torch.load('/path/to/reg_weights.t7'), strict=False)

    encoder.train(False)
    regressor.train(False)

    for epoch in range(num_epochs):
        print('epoch ' + str(epoch) + '/' + str(num_epochs-1))
        
        total_rmse_a, total_rmse_v, cnt = 0.0, 0.0, 0

        z_list, scores_list, labels_list = [], [], []
        with torch.no_grad():
            for batch_i, data_i in enumerate(loaders['val']):
                
                data, emotions = data_i['image'], data_i['va']
                valence = np.expand_dims(np.asarray(emotions[0]), axis=1)  # [64, 1]
                arousal = np.expand_dims(np.asarray(emotions[1]), axis=1)
                emotions = torch.from_numpy(np.concatenate([valence, arousal], axis=1)).float()

                if use_gpu:
                    inputs, correct_labels = Variable(data.cuda()), Variable(emotions.cuda())
                else:
                    inputs, correct_labels = Variable(data), Variable(emotions)                                       
                
                z = encoder(inputs)
                scores = regressor(z)

                z_list.append(z.detach().cpu().numpy())
                scores_list.append(scores.detach().cpu().numpy())
                labels_list.append(correct_labels.detach().cpu().numpy())
    
                RMSE_valence = RMSE(scores[:,0], correct_labels[:,0])
                RMSE_arousal = RMSE(scores[:,1], correct_labels[:,1])
                print('=====> [{}] RMSE_valence {} | arousal {} '.format(epoch, RMSE_valence, RMSE_arousal))
                
                total_rmse_v += RMSE_valence.item(); total_rmse_a += RMSE_arousal.item()
                cnt = cnt + 1

        scores_th = np.concatenate(scores_list, axis=0)
        labels_th = np.concatenate(labels_list, axis=0)

        std_l_v = np.std(labels_th[:,0]); std_p_v = np.std(scores_th[:,0])
        std_l_a = np.std(labels_th[:,1]); std_p_a = np.std(scores_th[:,1])
        mean_l_v = np.mean(labels_th[:,0]); mean_p_v = np.mean(scores_th[:,0])
        mean_l_a = np.mean(labels_th[:,1]); mean_p_a = np.mean(scores_th[:,1])

        PCC_v = np.cov(labels_th[:,0], np.transpose(scores_th[:,0])) / (std_l_v * std_p_v)
        PCC_a = np.cov(labels_th[:,1], np.transpose(scores_th[:,1])) / (std_l_a * std_p_a)
        CCC_v = (2.0 * std_l_v * std_p_v * PCC_v) / ( np.power(std_l_v,2) + np.power(std_p_v,2) + np.power(mean_l_v-mean_p_v,2) )
        CCC_a = (2.0 * std_l_a * std_p_a * PCC_a) / ( np.power(std_l_a,2) + np.power(std_p_a,2) + np.power(mean_l_a-mean_p_a,2) )

        sagr_v_cnt, sagr_a_cnt = 0, 0
        for i in range(len(labels_th)):
            if np.sign(labels_th[i,0]) == np.sign(scores_th[i,0]):
                sagr_v_cnt += 1
        SAGR_v = sagr_v_cnt / len(labels_th)

        for i in range(len(labels_th)):
            if np.sign(labels_th[i,1]) == np.sign(scores_th[i,1]):
                sagr_a_cnt += 1
        SAGR_a = sagr_a_cnt / len(labels_th)

        final_rmse_v = total_rmse_v/cnt
        final_rmse_a = total_rmse_a/cnt
        print("\n")
        print('=====> FINAL CCC valence {} | arousal {} '.format(CCC_v, CCC_a))
        print('=====> final PCC valence {} | arousal {} '.format(PCC_v, PCC_a))
        print('=====> final SAGR valence {} | arousal {} '.format(SAGR_v, SAGR_a))
        print('=====> FINAL RMSE valence {} | arousal {} '.format(final_rmse_v, final_rmse_a))


def _encoder2():
    encoder2 = Encoder2()
    return encoder2
def _regressor():
    regressor2 = Regressor_light()
    return regressor2
def _disc2():
    disc2 = Disc2_light()
    return disc2


class FaceDataset(Dataset):
    """ Face dataset loader """

    def __init__(self, csv_file, root_dir, transform=None, inFolder=None, landmarks=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.training_sheet = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        if inFolder is None:
            self.inFolder = np.full((len(self.training_sheet),), True)
        
        self.loc_list = np.where(inFolder)[0]
        self.infold = self.inFolder
        
    def __len__(self):
        return  np.sum(self.infold*1)

    def __getitem__(self, idx):
        valence = self.training_sheet.iloc[idx,1]
        arousal = self.training_sheet.iloc[idx,2]

        img_name = os.path.join(self.root_dir,
                                self.training_sheet.iloc[idx, 0])
        
        image = Image.open(img_name)
        sample = image
        
        if self.transform:
            sample = self.transform(sample)
        return {'image': sample, 'va': [valence, arousal]}



if __name__ == "__main__":

    #------------
    # Data loader
    #-----------
    training_path = '/path/to/training.csv'
    validation_path = '/path/to/validation.csv'

    face_dataset = FaceDataset(csv_file=training_path,  # all_path
                               root_dir='/media/CVIP/aff_wild/videos/train_ext1_crop_face/',
                               transform=transforms.Compose([
                                   transforms.Resize(256), transforms.RandomCrop(size=224),  # 128, 120
                                   transforms.ColorJitter(),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                               ]), inFolder=None)

    face_dataset_val = FaceDataset(csv_file=validation_path,
                                   root_dir='/media/CVIP/aff_wild/videos/train_ext1_crop_face/',
                                   transform=transforms.Compose([
                                       transforms.Resize(256), transforms.CenterCrop(size=224),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                   ]), inFolder=None)
    
    batch_size = 256  # resnet18: 128 / alexnet: 256
    dataloader = DataLoader(face_dataset, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(face_dataset_val, batch_size=16, shuffle=False)  # 16
    
    loaders = {'train': dataloader, 'val': dataloader_val}
    
    use_gpu = torch.cuda.is_available()
    dataset_size = {'train': len(face_dataset), 'val': len(face_dataset_val)}

    #----------------
    # Make DNN models
    #----------------
    encoder2  = _encoder2().cuda()
    regressor = _regressor().cuda()
    disc2     = _disc2().cuda()
    
    RMSE = nn.MSELoss()
    enc_opt   = optim.Adam(encoder2.parameters(), lr  = 1e-4, betas = (0.5, 0.9))
    reg_opt   = optim.Adam(regressor.parameters(), lr = 1e-4, betas = (0.5, 0.9))
    disc_opt  = optim.Adam(disc2.parameters(), lr     = 1e-4, betas = (0.5, 0.9))
    
    enc_exp_lr_scheduler  = lr_scheduler.StepLR(enc_opt, step_size  = 5000, gamma = 0.9)  # step_size (default) is 5000
    reg_exp_lr_scheduler  = lr_scheduler.StepLR(reg_opt, step_size  = 5000, gamma = 0.9)
    disc_exp_lr_scheduler = lr_scheduler.StepLR(disc_opt, step_size = 5000, gamma = 0.9)
    
    #-----------------------
    # Training or evaluation
    #-----------------------
    if args['train']:
        print(fg256("cyan", "Training phase"))
        model_training([encoder2             , regressor            , disc2]                 ,
                       [RMSE]                ,
                       [enc_opt              , reg_opt              , disc_opt]              ,
                       [enc_exp_lr_scheduler , reg_exp_lr_scheduler , disc_exp_lr_scheduler] ,
                       num_epochs=131)
    else:
        print(fg256("yellow", "Evaluation phase"))
        model_evaluation([encoder2             , regressor            , disc2]                 ,
                         [RMSE]                ,
                         num_epochs=1)