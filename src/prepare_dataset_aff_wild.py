__author__ = "kdhht5022@gmail.com"
# -*- coding: utf-8 -*-
# python 3.6
import os
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import csv

import numpy as np
import scipy.misc as ms
import json
import glob

import cv2
from tqdm import tqdm

from src import detect_faces, show_bboxes
from PIL import Image


if __name__ == "__main__":
    
    #---------------------------
    # Face detection using MTCNN
    #---------------------------

    folder_no = 111
    mypath = '/home/kdh/Desktop/pytorch/AffectNet/vid/real_children/close/4/'
    
    pic_list = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    ll = []
    for i in range(len(pic_list)):
        ll.append(str(i+1247)+'.png')
    cropped_list = []
    
    for i in range(len(ll)):
        NAME = ll[i][:-4]
    
        img = Image.open(mypath+'/'+str(NAME)+'.png')
        bounding_boxes, landmarks = detect_faces(img)
        show_bboxes(img, bounding_boxes, landmarks)
    
        if len(bounding_boxes) == 1:
            bb = np.squeeze(bounding_boxes)
            cropped_face = img.crop((bb[0]-10, bb[1]-10, bb[2]+10, bb[3]+10))  # near: -40,-20,+40,+40 / middle: -20,-10,+20,+20
            cropped_list.append(cropped_face)                                  # far: -10,-10,+10,+10
        else:
            for j in range(len(bounding_boxes)):
                bb = bounding_boxes[j]
                cropped_face = img.crop((bb[0]-10, bb[1]-10, bb[2]+10, bb[3]+10))
                cropped_list.append(cropped_face)
                
    
    plt.imshow(cropped_face)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    for i in range(len(cropped_list)):
        if cropped_list[i].size[0] != 0 and cropped_list[i].size[1] != 0:
            ms.imsave('../AffectNet/custom/112/dy/4/'+ll[i], cropped_list[i])
            
            
    #----------------
    # Data annotation
    #----------------

    ll = glob.glob('/home/kdh/Desktop/pytorch/aff_wild/videos/train/*')
    ll.sort()
    
    ll1 = []
    for i in range(len(ll)):
        ll1.append(ll[i][-7:-4])
    
    
    mylist = [['subDirectory_filePath', 'valence', 'arousal']]
    
    for j in tqdm(range(len(ll1))):
    
        mypath = '/media/CVIP/aff_wild/videos/train_ext1_crop_face/'+str(ll1[j])+'/*'
        img_path = sorted(glob.glob(mypath), key=os.path.getmtime)
        
        with open('/media/CVIP/aff_wild/annotations/train/arousal/'+str(ll1[j])+'.txt') as f:
            a_path = f.read().splitlines()  # int(f.read().splitlines())
        with open('/media/CVIP/aff_wild/annotations/train/valence/'+str(ll1[j])+'.txt') as f:
            v_path = f.read().splitlines()  # int(f.read().splitlines())
    
        ### make csv file using list
        name_list = []
        for i in range(len(img_path)):
            nn = img_path[i].split('/')[-2] + '/' + img_path[i].split('/')[-1]
            name_list.append(nn)
            
        min_value = np.min([len(v_path), len(name_list)])
        for i in range(min_value):
            mylist.append([name_list[i], v_path[i], a_path[i]])
        
        del img_path, a_path, v_path
    
    
    with open('/media/CVIP/aff_wild/videos/train_ext1_crop_face/training.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(mylist)
        
        
    


    import pretrainedmodels
    import pretrainedmodels.utils as utils
    from audtorch.metrics import ConcordanceCC
    from audtorch.metrics import PearsonR

    model_name = 'resnet18'  #'se_resnext50_32x4d'
    resnet = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    
    import torch
    from pytorch_model_summary import summary
    print(summary(resnet, torch.ones_like(torch.empty(1, 3, 255, 255)), show_input=True))


