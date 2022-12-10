#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 09:25:52 2022

@author: max
"""
import scipy 
import numpy as np
import os 




A = scipy.io.loadmat('/home/max/Documents/s194119/Bachelor/Data/POETdataset/etData/etData_sofa.mat',squeeze_me=True,struct_as_record=False)
etData = A["etData"]
#etData[0].fixations[0].imgCoord.fixR.pos
NUM_TRACKERS = 5

eyeData = np.empty((len(etData),NUM_TRACKERS,1),dtype=object)#np.empty(shape=(len(etData),5,2)) #number of measurements, number of subjects, number of channels (Left, Right)
filename = []
im_dims = []
for i,j in enumerate(etData): #i is image, k is patient
    filename.append(etData[i].filename + ".jpg")
    im_dims.append(etData[i].dimensions)
    #print(j)
    
    for k in range(5):
        w_max = im_dims[i][1]
        h_max = im_dims[i][0]
        LP = etData[i].fixations[k].imgCoord.fixL.pos[:]
        RP = etData[i].fixations[k].imgCoord.fixR.pos[:]
        #remove invalid values . if larger than image-dims or outside of image (negative vals)
        #LP = LP[~np.any((LP[:,]),:]
        BP = np.vstack((LP,RP)) #LP ; RP 
        #eyeData[i,k,0] = [LP,RP] #fill with list values 
        eyeData[i,k,0] = BP #fill with matrix
        

#eyePosR = etData["fixations"]
#eyePosR = eyePosR["imgCoord"]
#eyePosR = eyePosR["fixR"]
#filenames = etData["filename"]

import matplotlib.pyplot as plt
from matplotlib import image
import cv2 
import random

#load a random image
classOI = "sofa_"
choice = random.randint(0,len(etData))
path = os.path.join("/home/max/Documents/s194119/Bachelor/Data/POETdataset/PascalImages/",classOI+filename[choice])
im = image.imread(path)
plt.title("{}".format(filename[choice]))


#now for eye-tracking-data
ax = plt.gca()
for i in range(NUM_TRACKERS):
    mylabel = str(i+1)
    num_fix = int(eyeData[choice,i][0][:,0].shape[0]/2)
    print(num_fix) #number of fixations on img
    #Left eye
    color = next(ax._get_lines.prop_cycler)['color']
    plt.scatter(eyeData[choice,i][0][:,0],eyeData[choice,i][0][:,1],alpha=0.8,label=mylabel,color = color) #wrong - you are not plotting pairwise here, you are plotting R as function of L 
    plt.plot(eyeData[choice,i][0][0:num_fix,0],eyeData[choice,i][0][0:num_fix,1],label=str(),color= color)
    plt.plot(eyeData[choice,i][0][num_fix:,0],eyeData[choice,i][0][num_fix:,1],label=str(),color = color)

    
plt.legend()
plt.imshow(im)