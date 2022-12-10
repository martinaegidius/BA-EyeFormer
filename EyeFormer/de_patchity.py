#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 10:02:19 2022

@author: max
"""

import torch 
import math

torch.manual_seed(2)
x = torch.rand(1,196,768)
#,768)


def depatchify(t,order="rowMajor",patchsize=16):
    """Takes in a ViT-output tensor of shape [1,196,768] and converts into a 14x14x768 image using specified order
    input-args: 
        t: torch tensor 
        order: string, "rowMajor" or "columnMajor". rowMajor is default
        patchsize: int, indicating the dimensions of every patch used
        
    """
    if not(order=="rowMajor" or order=="colMajor"):
        print("Wrong input-order. Aborted execution of depatchify")
        return None
    
    n_patches = int(math.sqrt(t.shape[1])) #always square after resize 
    holder_t = torch.zeros(n_patches,n_patches,768)
    #linear_t = torch.zeros(196,768)
    ncol = 0
    nrow = 0
    if(order=="rowMajor"):
        for i in range(t.shape[1]): #196-dimension
            if(i%n_patches==0 and i!=0): 
                #print("reset loop")
                nrow += 1
                ncol = 0
            linear_t = t[0,i,:]
            holder_t[nrow,ncol,:] = linear_t 
            #nrow += 1
            ncol += 1
          
    else:
        for i in range(t.shape[1]): #196-dimension
            if(i%n_patches==0 and i!=0): 
                #print("reset loop")
                ncol += 1
                nrow = 0
            linear_t = t[0,i,:]
            holder_t[nrow,ncol,:] = linear_t 
            #nrow += 1
            nrow += 1
    return holder_t
          
out = depatchify(x)
out2 = depatchify(x,order="colMajor")
