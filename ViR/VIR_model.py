#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:58:32 2022

@author: max
"""
import torch
import math
import torch.nn as nn
from torchvision import ops

class VIT_REG_MODEL(nn.Module):
    def __init__(self,input_dim=768,output_dim=4,dropout=0.0):
        super().__init__()
        self.d_model = input_dim
        self.first_decoder = nn.Linear(self.d_model,2048)
        #self.first_decoder = nn.Linear(self.d_model,2048,bias=True)
        self.activation = nn.ReLU()
        self.second_decoder = nn.Linear(2048,4,bias=True)
        if(dropout==0.0):
            self.dropout = None
        else:
            self.dropout = nn.Dropout(dropout)
        
    
    def forward(self,x):
        x = self.first_decoder(x)
        if self.dropout!=None:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.second_decoder(x)
        return x


    
class NoamOpt:
    #"Optim wrapper that implements rate."
    # !Important: warmup is number of steps (number of forward pass), not number of epochs. 
    # number of forward passes in one epoch: len(trainloader.dataset)/len(trainloader)
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
    def get_std_opt(model):
        return NoamOpt(model.d_model, 2, 4000,torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def boxIOU(preds,labels):
    """
    Own-implemented batch-solid IOU-calculator. 
    Returns: tensor, [BS]
    """
    BSZ = preds.shape[0]
    assert BSZ == labels.shape[0], "zeroth dimension of preds and labels are not the same"
    IOU = torch.zeros(BSZ)
    for i in range(BSZ):
        A_target = (labels[i,2]-labels[i,0])*(labels[i,3]-labels[i,1]) #(x2-x1)*(y2-y1)
        A_pred = (preds[i,2]-preds[i,0])*(preds[i,-1]-preds[i,1]) #(x2-x1)*(y2-y1)
        U_width = torch.min(labels[i,2],preds[i,2]) - torch.max(labels[i,0],preds[i,0]) #width is min(lx2,px2)-(max(lx0,px0))
        U_height = torch.min(labels[i,3],preds[i,3]) - torch.max(labels[i,1],preds[i,1])  
        A_U = U_width * U_height
        IOU[i] = A_U / (A_target+A_pred-A_U)
    return IOU      
    
def pascalACC(preds,labels): #TODO: does not work for batched input. Fix
    """
    Function for calculating the accuracy between a batch of predictions and corresponding batch of targets. 
    Returns: number of correct predictions in batch, number of false predictions in batch and a list of IOU-scores for the batch
    """
    
    BSZ = preds.shape[0]
    assert BSZ == labels.shape[0],"Batch-size dimensions between target and tensor not in corresondance!"
    
    no_corr = 0 
    no_false = 0
    IOU_li = []
    
    if preds.dim()==1:
        preds = preds.unsqueeze(0)
    if labels.dim()==1:
        labels = labels.unsqueeze(0)
        
    #no loop approach: 
        #may be more effective for small batches. But only the diagonal of IOU_tmp is of interest for you - thus many wasted calculations
    #IOU_tmp = ops.box_iou(preds,labels) #calculates pairwise 
    #print(torch.diagonal(IOU_tmp))
    
    for i in range(BSZ): #get pascal-criterium accuraccy
        pred_tmp = preds[i,:].unsqueeze(0)
        label_tmp = labels[i,:].unsqueeze(0)
        IOU = ops.box_iou(pred_tmp,label_tmp)
        IOU_li.append(IOU.item())
        if(IOU>0.5):
            no_corr += 1
        else:
            no_false += 1

    return no_corr,no_false,IOU_li

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VIT_REG_MODEL().to(device)
    print(get_n_params(model))
    #model.switch_debug()
    #deep_seq = torch.rand(2,32,770)
    #model(deep_seq)

