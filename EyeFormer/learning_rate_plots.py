#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 18:11:55 2022

@author: max
"""

import torch 
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from matplotlib import rc,rcParams
"""
#-------------------------Learning-curves-plotting script for all classes---------------------------------#

Usage: first scp all run-results
cp folder-contens of nL_1_nH_2 to inst folder of interest
then simply run

#Todo: cp cat, cow, diningtable, dog, horse, motorbike and sofa as follows
        scp -r s194119@login1.gbar.dtu.dk:BA/eyeFormer/Data/POETdataset/boat/nL_1_nH_2 /home/max/Documents/s194119/Bachelor/Results/1_fold_results/boat/

Additionally: you ran it without dropout -.- 

#---------------------------------------------------------------------------------------------------------#
"""

FSZ = 18
lFSZ = 16
params = {'legend.fontsize': 20,
          'legend.handlelength': 2}

plt.rcParams.update(params)

sns.set_theme()
root_path = os.path.dirname(__file__)+"/Big_dset_results/all_classes"#"/Data/POETdataset/"
#classes = ["aeroplane","bicycle","boat","cat","cow","diningtable","dog","horse","motorbike","sofa"]
model_types = ["L_378_nL_1_nH_1","L_378_nL_3_nH_1","L_378_nL_3_nH_1","L_378_nL_5_nH_1","L_378_nL_6_nH_1","L_378_nL_9_nH_1"]

epochDict = {"L_378_nL_1_nH_1":[-1,-1],"L_378_nL_3_nH_1":[-1,-1],"L_378_nL_5_nH_1":[-1,-1],
             "L_378_nL_6_nH_1":[-1,-1],"L_378_nL_9_nH_1":[-1,-1]} #make a dict for storing data for optimal number of epochs. 

pwds = os.path.dirname(__file__)+"/train_curves/"
if not os.path.exists(pwds):
    os.mkdir(pwds)

plt.ioff()

#for inst in classes:
for model in model_types:
    #PLOT LOSS-CURVES
    fig = plt.figure()
    path = root_path + "/"+model+"/"
    train_loss = torch.load(path+"all_classes_result_losses.pth")
    val_loss = torch.load(path+"all_classes_val_losses.pth")
    minVal = min(val_loss)
    idx = val_loss.index(minVal)
    epochDict[model][0] = idx
    epochDict[model][1] = minVal
    
    fig,axes = plt.subplots(1,2,figsize=(16,7),sharex=True)
    axes[0].plot(train_loss)
    axes[0].plot(val_loss)
    axes[0].plot(idx,minVal,marker='x',markersize=11,markerfacecolor="red",markeredgewidth = 2,markeredgecolor="red",linestyle="None")
    axes[0].set_xlabel("Epochs",fontsize=FSZ,fontweight="bold")
    axes[0].set_ylabel("L1-Loss",fontsize=FSZ,fontweight="bold")
    axes[0].tick_params(axis='both', which='major', labelsize=13)
    plt.xlim([0,350])
    axes[1].semilogy(train_loss)
    axes[1].semilogy(val_loss)
    axes[1].set_xlabel("Epochs",fontsize=FSZ,fontweight="bold")
    axes[1].set_ylabel("Logarithmic Loss",fontsize=FSZ,fontweight="bold")
    axes[1].semilogy(idx,minVal,marker='x',markersize=11,markerfacecolor="red",markeredgewidth = 2,markeredgecolor="red",linestyle="None")
    axes[1].tick_params(axis='both', which='major', labelsize=13)
    plt.xlim([0,350])
    
    leg = fig.legend(["Train","Validation","Chosen epoch"],loc="upper center",ncol=3,bbox_to_anchor=(0.5, 0.96),
              bbox_transform=fig.transFigure,fontsize=lFSZ) #prop={'size': 13}
    #fig.legend(fontsize=FSZ)
    for line in leg.get_lines():
        line.set_linewidth(6.0)
    
    fig.suptitle(model,size=FSZ-2)#,y=1.2)
    plt.savefig(pwds+model+"_learning_curves.pdf")
    plt.close(fig)
    
np.save(path+"../../"+"optimal_no_epochs.npy",epochDict) #save number of epochs to file     
    
for model in model_types:
    ####PLOT ACCURACY-CURVES
    fig = plt.figure()
    
    path = root_path+"/"+model+"/"
    train_acc= torch.load(path+"all_classes_result_acc.pth")
    val_acc = torch.load(path+"all_classes_val_acc.pth")
    
    train_acc_mv = np.convolve(train_acc,np.ones(10)/10)[:2000]
    val_acc_mv = np.convolve(val_acc,np.ones(10)/10)[:2000]
    fig,axes = plt.subplots(1,2,figsize=(16,7),sharex=True)
    axes[0].plot(train_acc_mv)
    axes[0].plot(val_acc_mv)
    axes[0].set_xlabel("Epochs",fontsize=FSZ,fontweight="bold")
    axes[0].set_ylabel("m. avg. IOU accuracy",fontsize=FSZ,fontweight="bold")
    axes[0].tick_params(axis='both', which='major', labelsize=13)
    plt.xlim([0,350])
    axes[1].semilogy(train_acc_mv)
    axes[1].semilogy(val_acc_mv)
    axes[1].set_xlabel("Epochs",fontsize=FSZ,fontweight="bold")
    axes[1].set_ylabel("m. avg. Logarithmic IOU accuracy",fontsize=FSZ,fontweight="bold")
    axes[1].tick_params(axis='both', which='major', labelsize=13)
    plt.xlim([0,350])
    
    leg = fig.legend(["Train","Validation"],loc="upper center",ncol=3,bbox_to_anchor=(0.5, 0.96),
              bbox_transform=fig.transFigure,fontsize=lFSZ) #prop={'size': 13}
    #fig.legend(fontsize=FSZ)
    for line in leg.get_lines():
        line.set_linewidth(6.0)
    
    fig.suptitle(model,size=FSZ-2)#,y=1.2)
    plt.savefig(pwds+model+"_accuracy_curves.pdf")
    plt.close(fig)
    
    
    
    
    

    

