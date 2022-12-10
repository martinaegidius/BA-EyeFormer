#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 16:22:30 2022

@author: max
"""

import sys
import os 

#for running exp 1 

classOC = int(sys.argv[1])


L = [1,4,8,16]



num_layers = [1,3,6,9]
num_heads = [1,2]



for length in L:
    for heads in num_heads: 
        for layer in num_layers:
            command = "python3 custom_multihead_attention.py " + str(classOC) + " "+ str(length) + " " + str(layer) + " " + str(heads) + " " 
            os.system(command)
            
        
