#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:59:02 2022

@author: max
"""

import torch
import matplotlib.pyplot as plt
import numpy as np 

"""Function to plot used learning rate"""


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

opts = [NoamOpt(2, 1, 22000, None), 
        NoamOpt(2, 1/5, 22000, None),
        NoamOpt(2, 1/10, 22000, None)]
plt.plot(np.arange(1, 40000), [[opt.rate(i) for opt in opts] for i in range(1, 40000)])
plt.legend(["1:22000","1/5:22000", "1/10:22000"])
None

