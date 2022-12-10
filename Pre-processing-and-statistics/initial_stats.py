#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 12:58:14 2022

@author: max
"""
import os 


from load_POET import pascalET
import numpy as np


DEBUG = False
DEBUG_CLEANUP = False
dset = pascalET()
dset.loadmat()


#debug new format 
BP = dset.convert_eyetracking_data(CLEANUP=True,STATS=True)
dset.load_images_for_class(0)
dset.specific_plot(0,"2009_005036.jpg")
idx = dset.filename.index("2009_005036.jpg") #remember - this only works with current filename structure. Right now, filenames are saved when images are loaded


try:
    print("Number of fixes in general: ",dset.eyeData_stats[0,idx,:,0])
    print("Number of fixes in bbox: ", dset.eyeData_stats[0,idx,:,1])
except:
    pass


#TODO: figure out why your histogram is wrong!!! 
print(dset.eyeData[0,idx])
print("Comparing image to dataset it seems right!:-)")
#todo: now, per image per class, sum number of fixations (np.sum along dim 3). This gives you histograms per image instance. This gives histogram.

import matplotlib.pyplot as plt 

airplanes = dset.eyeData_stats[0][:dset.num_files_in_class[0]]




#get histogram of number of fixations outside of box

countsO, binsO = np.histogram(airplanes[:,:,0],range=(0,10))

#and inside of box
plt.figure(13367)
countsI, binsI = np.histogram(airplanes[:,:,1],range=(0,10))
plt.hist(binsI[:-1], binsI, weights=countsI)#IS WRONG AND I DO NOT KNOW WHY 
plt.show()




#for debugging data-cleanup-procedure
if(DEBUG_CLEANUP == True):
    tmpbbx = dset.etData[0][idx].gtbb
    im_dims = dset.etData[0][idx].dimensions
    xs,ys,w,h = dset.get_bounding_box(tmpbbx)
    nbbx = [xs,ys,w,h]
    
    CLEANUP = True
    
    countB = []
    countA = []
    for i in range(5):
        LP = dset.etData[0][idx].fixations[i].imgCoord.fixL.pos[:]
        RP = dset.etData[0][idx].fixations[i].imgCoord.fixR.pos[:]
        BP = np.vstack((LP,RP)) #LP|RP    
        countA.append(dset.get_num_fix_in_bbox(nbbx, BP))
        print("BP before cleaning: ",BP)
        
        if(CLEANUP == True and BP.shape[1]>0): #necessary to failcheck; some measurements are erroneus. vstack of two empty arrs gives BP.shape=(2,0)
            BP = np.delete(BP,np.where(np.isnan(BP[:,0])),axis=0)
            BP = np.delete(BP,np.where(np.isnan(BP[:,1])),axis=0)
            BP = np.delete(BP,np.where((BP[:,0]<0)),axis=0) #delete all fixations outside of quadrant
            BP = np.delete(BP,np.where((BP[:,1]<0)),axis=0) #delete all fixations outside of quadrant
            BP = np.delete(BP,np.where((BP[:,1]>im_dims[0])),axis=0) #remove out of images fixes on x-scale 
            BP = np.delete(BP,np.where((BP[:,0]>im_dims[1])),axis=0) #remove out of images fixes on y-scale
        print("BP after cleaning: ",BP)
        countB.append(dset.get_num_fix_in_bbox(nbbx,BP))
            
    
    if(DEBUG==True):
        dset.load_images_for_class(8)
        dset.random_sample_plot()
    
    
    

#dset.basic_hists()
#dset.basic_stats()
#dset.specific_plot(9,"2009_003646.jpg") #guy with muddy sofa and TV :-) 