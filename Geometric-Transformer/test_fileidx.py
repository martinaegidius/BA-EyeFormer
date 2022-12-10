#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:30:33 2022

@author: max
"""

import os 


root_dir = os.path.dirname(__file__)
classesOC = [0,9]
def get_split(root_dir,classesOC):
    split_dir = root_dir +"/Data/POETdataset/split/"
    classlist = ["aeroplane","bicycle","boat","cat","cow","diningtable","dog","horse","motorbike","sofa"]

    training_data = []
    for classes in classesOC: 
        train_filename = split_dir + classlist[classes]+"_Rbbfix.txt"
        with open(train_filename) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        for i in range(len(lines)): 
            lines[i] = classlist[classes]+"_"+lines[i]+".jpg"
            
        training_data.append(lines)
        
    test_data = []
    for classes in classesOC: 
        train_filename = split_dir + classlist[classes]+"_Rfix.txt"
        with open(train_filename) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        for i in range(len(lines)): 
            lines[i] = classlist[classes]+"_"+lines[i]+".jpg"
            
        test_data.append(lines)
        
    return training_data,test_data
    


train,test = get_split(root_dir,classesOC)



        
