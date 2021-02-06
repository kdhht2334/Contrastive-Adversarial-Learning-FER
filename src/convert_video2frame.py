#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert video to sequence frame

@author: kdhht5022@gmail.com
"""
import glob
import subprocess

import cv2
import numpy as np

import argparse
app = argparse.ArgumentParser()
app.add_argument("-n", "--number", type=int, default=0)
args = vars(app.parse_args())

number = args['number']


ll = glob.glob('/aff_wild/Train_Set/*')
ll.sort()


ll1 = []
for i in range(len(ll)):
    ll1.append(ll[i].split('/')[-1][:-4])
ll1.pop()

ll2 = []
for i in range(len(ll)):
    ll2.append(ll[i].split('/')[-1][-4:])
ll2.pop()
    

if number == 0:
    for i in range(len(ll1)):
        subprocess.Popen('mkdir ../train_ext2/{}'.format(ll1[i]), shell=True)
else:
    pass

i = number
print("[INFO] ======================= Start {}-th folder!!".format(ll1[i]))
vid = cv2.VideoCapture("{}{}".format(ll1[i], ll2[i]))
fps = int(np.ceil(vid.get(cv2.CAP_PROP_FPS)))
subprocess.Popen('ffmpeg -i {}{} -vf fps={} ../train_ext2/{}/%d.png'
                .format(ll1[i], ll2[i], fps, ll1[i]), 
                shell=True)