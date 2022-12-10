#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 09:05:39 2022

@author: max
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 20:51:26 2022

@author: max
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch 
import os 
from random import sample
import math
#plt.rcParams['legend.title_fontsize'] = 'xx-small'

#-------------------Script flags----------------------#
DEBUG = True

#-----------------------END>--------------------------#
className = "all_classes"
impath = os.path.dirname(__file__)+"/../../New_bachelor/Data/POETdataset" #path for image-data
dpath = os.path.dirname(__file__) + "/all_classes/L_44152_nL_5_nH_2/eval/"
#dpath = impath + "/Results/"+className+"/nL_6_nH_2" #path for saved model-results
#dpath = os.path.dirname(__file__)+"/timm_scripts/"+className+"/nL_1_nH_1"
impath += "/PascalImages/"


def load_results(root_dir,className,file_specifier):
    path = root_dir + className +"_" +file_specifier+".pth"
    print(path)
    if(DEBUG):
        print("path is: ",path)
    try: 
        entrys = torch.load(path,map_location=torch.device('cpu'))
        print("...Success loading...")
        return entrys
    except: 
        print("Error loading file. Check filename/path: ",path)
        return None

def rescale_coords(data,idx=None):
    #takes imdims and bbox data and rescales. [MUTATES]
    #Returns tuple of two tensors (target,preds)
    if idx==None:
        imdims = data[-1].squeeze(0).clone().detach()
        target = data[3].squeeze(0).clone().detach()
        preds = data[4].squeeze(0).clone().detach()
        
        target[0::2] = target[0::2]*imdims[-1]
        target[1::2] = target[1::2]*imdims[0]
        preds[0::2] = preds[0::2]*imdims[-1]
        preds[1::2] = preds[1::2]*imdims[0]
    
    else: 
        imdims = data[-1][idx-1].squeeze(0).clone().detach()
        target = data[3][idx-1].squeeze(0).clone().detach()
        preds = data[4][idx-1].squeeze(0).clone().detach()
        target[0::2] = target[0::2]*imdims[-1]
        target[1::2] = target[1::2]*imdims[0]
        preds[0::2] = preds[0::2]*imdims[-1]
        preds[1::2] = preds[1::2]*imdims[0]
    
    #print("Rescaled target: ",target)
    #print("Rescaled preds: ",preds)
    
    return target,preds
    
def single_rescale_coords(box,data,idx=None):
    #takes imdims from data and bbox data and rescales. Return type: tensor with scaled box coordinates, fitting on image-dims
    if(idx==None):
        imdims = data[-1].squeeze(0).clone().detach()
    else:
        imdims = data[-1][idx-1]
    nbox = torch.zeros(4)
    nbox[0::2] = box[0::2]*imdims[-1]
    nbox[1::2] = box[1::2]*imdims[0]
    return nbox

def single_format_bbox(box,data,idx=None):
    """Returns bounding box in format ((x0,y0),w,h) for a mean-model-box
    Args: 
        input: 
            box: generated mean-data-box
            data: saved list-of-list-of-list containing model-data and image-names. 
                Format: [filename,class,IOU,target,output,size]
        returns: 
            Bounding box coordinates in format ((x0,y0),w,h) for mean/median box
    """   
    if idx==None:
        scaled = single_rescale_coords(box,data)
    else: 
        scaled = single_rescale_coords(box,data,idx)
    x0 = scaled[0]
    y0 = scaled[1]
    w = scaled[2] - x0
    h = scaled[3] - y0
    #print("x0: ",x0,"y0: ",y0,"w: ",w,"h: ",h)
    return [x0,y0,w,h]

def bbox_for_plot(data,idx=None):
    """Returns bounding box in format ((x0,y0),w,h)
    Args: 
        input: 
            data: saved list-of-list-of-list containing model-data and image-names. 
                Format: [filename,prediction-result,IOU,target,output,size]
        returns: 
            Bounding box coordinates in format ((x0,y0),w,h) for preds and target 
            (target,preds)
    """   
    
    targets,preds = rescale_coords(data,idx)
 
    #for targetholder_li.append()
    x0 = targets[0]
    y0 = targets[1]
    w = targets[2] - x0
    h = targets[3] - y0
    #print("x0: ",x0,"y0: ",y0,"w: ",w,"h: ",h)
    target = [x0,y0,w,h]
    
    #for preds 
    x0 = preds[0]
    y0 = preds[1]
    w = preds[2] - x0
    h = preds[3] - y0
    #print("x0: ",x0,"y0: ",y0,"w: ",w,"h: ",h)
    preds = [x0,y0,w,h]
    
    return target,preds
    
def mean_model(data): #does not work for batched data yet. 
    #Generate mean-model of training-data
    #holder_t = torch.zeros(len(data)*BATCH_SZ,4)
    holder_t = torch.zeros(len(data),4)
    for entry in range(len(data)):
        holder_t[entry] = data[entry][3].squeeze(0)
        #print(data[entry][3].squeeze(0))
    return torch.mean(holder_t,0)
        
def median_model(data): #does not work for batched data yet.
    #Generate median-model of training-data
    #old implementation (single-batch)
    holder_t = torch.zeros(len(data),4)
    for entry in range(len(data)):
        holder_t[entry] = data[entry][3].squeeze(0)
    
      
    return torch.median(holder_t,0)[0]
    
    

def plot_train_set(data,root_dir,classOC=0):
    """Prints subset of data which is overfit upon
    Args: 
        input: 
            data: saved list-of-list-of-list containing model-data and image-names
                Format: [filename,class,IOU,target,output,size]
            root_dir: imagedir
            classOC: descriptor of class which is used
        returns: 
            None
    """                  
    meanBox = mean_model(data)
    medianBox = median_model(data)          
    NSAMPLES = len(data)
    NCOLS = 2
    
    if NSAMPLES > 9: 
        NSAMPLES = 9
        data = sample(data,9)
        NCOLS = 3
    
    fig, ax = plt.subplots(math.ceil(NSAMPLES/NCOLS),NCOLS,figsize=(14,12))
    col = 0
    row = 0
    for entry in range(NSAMPLES):
        filename = data[entry][0]
        im = plt.imread(root_dir+filename)
        target,preds = bbox_for_plot(data[entry])
        mean_box = single_format_bbox(meanBox, data[entry])
        median_box = single_format_bbox(medianBox,data[entry])
        rectT = patches.Rectangle((target[0],target[1]), target[2], target[3], linewidth=3, edgecolor='r', facecolor='none')
        rectP = patches.Rectangle((preds[0],preds[1]), preds[2], preds[3], linewidth=3, edgecolor='m', facecolor='none')
        rectM = patches.Rectangle((mean_box[0],mean_box[1]),mean_box[2],mean_box[3],linewidth=2,edgecolor='c',facecolor='none')
        rectMed = patches.Rectangle((median_box[0],median_box[1]),median_box[2],median_box[3],linewidth=2,edgecolor='lightcoral',facecolor='none')
        if entry%NCOLS==0 and entry!=0: #NUMBER OF COLS
            row += 1 
            col = 0 
        print("Col: ",col,"Row:",row)
        ax[row,col].imshow(im)
        ax[row,col].tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
        ax[row,col].add_patch(rectT)
        ax[row,col].add_patch(rectP)
        ax[row,col].add_patch(rectM)
        ax[row,col].add_patch(rectMed)
        
        
        #plt.text(target[0]+target[2]-10,target[1]+10, "IOU", bbox=dict(facecolor='red', alpha=0.5))
        #ax[row,col].text(data[entry][4][0][2].item()-0.1,data[entry][4][0][1].item()+0.1,"IOU:{:.2f}".format(data[entry][2][0].item()),bbox=dict(facecolor='magenta', alpha=0.5),transform=ax[row,col].transAxes)
        ax[row,col].text(0.02,0.05,"IOU:{:.2f}".format(data[entry][2]),fontweight="bold",bbox=dict(facecolor='magenta', alpha=0.75),transform=ax[row,col].transAxes)
        #ax[row,col].set_title(filename,fontweight="bold",size=8)
        col += 1
    plt.rcParams['legend.handlelength'] = 1
    plt.rcParams['legend.handleheight'] = 1.125
    plt.tight_layout(rect=(0,0,1,0.90))
    fig.legend((rectT,rectP,rectM,rectMed),("Target","Prediction","Mean box","Median box"),loc="upper center",ncol=4,framealpha=0.0,prop={'size':22,'weight':"bold"},bbox_to_anchor=(0.5, 0.97))
    #fig.suptitle("Predictions on subset of trainset")
    plt.savefig("results_on_trainset.pdf")
    plt.show()
    return None

def plot_three_train_set(data,root_dir,classOC=0):
    """Prints subset of data which is overfit upon
    Args: 
        input: 
            data: saved list-of-list-of-list containing model-data and image-names
                Format: [filename,class,IOU,target,output,size]
            root_dir: imagedir
            classOC: descriptor of class which is used
        returns: 
            None
    """                  
    meanBox = mean_model(data)
    medianBox = median_model(data)          
    NSAMPLES = 3 #number of batches in sample
    NCOLS = 1
     
    data = sample(data,NSAMPLES)
    
    
    NCOLS = NSAMPLES
    
    fig, ax = plt.subplots(math.ceil(NSAMPLES/NCOLS),NCOLS,figsize=(12,12))
    col = 0
    row = 0
    for entry in range(NSAMPLES):
        filename = data[entry][0]
        im = plt.imread(root_dir+filename)
        target,preds = bbox_for_plot(data[entry])
        mean_box = single_format_bbox(meanBox, data[entry])
        median_box = single_format_bbox(medianBox,data[entry])
        rectT = patches.Rectangle((target[0],target[1]), target[2], target[3], linewidth=2, edgecolor='r', facecolor='none')
        rectP = patches.Rectangle((preds[0],preds[1]), preds[2], preds[3], linewidth=2, edgecolor='m', facecolor='none')
        rectM = patches.Rectangle((mean_box[0],mean_box[1]),mean_box[2],mean_box[3],linewidth=2,edgecolor='c',facecolor='none')
        rectMed = patches.Rectangle((median_box[0],median_box[1]),median_box[2],median_box[3],linewidth=2,edgecolor='lightcoral',facecolor='none')
        #if entry%NCOLS==0 and entry!=0: #NUMBER OF COLS
        #    row += 1 
        #    col = 0 
        print("Col: ",col,"Row:",row)
        ax[col].imshow(im)
        ax[col].tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
        ax[col].add_patch(rectT)
        ax[col].add_patch(rectP)
        ax[col].add_patch(rectM)
        ax[col].add_patch(rectMed)
        
        
        #plt.text(target[0]+target[2]-10,target[1]+10, "IOU", bbox=dict(facecolor='red', alpha=0.5))
        #ax[row,col].text(data[entry][4][0][2].item()-0.1,data[entry][4][0][1].item()+0.1,"IOU:{:.2f}".format(data[entry][2][0].item()),bbox=dict(facecolor='magenta', alpha=0.5),transform=ax[row,col].transAxes)
        ax[col].text(0.02,0.05,"IOU:{:.2f}".format(data[entry][2]),bbox=dict(facecolor='magenta', alpha=0.75),transform=ax[col].transAxes)
        #ax[col].set_title(filename,fontweight="bold",size=8)
        col += 1
    plt.rcParams['legend.handlelength'] = 1
    plt.rcParams['legend.handleheight'] = 1.125
    fig.legend((rectT,rectP,rectM,rectMed),("Target","Prediction","Mean box","Median box"),loc="upper center",ncol=4,framealpha=0.0,bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("Predictions on subset of trainset")
    plt.show()
    plt.savefig("class_"+str(classOC)+"eval_on_trainset.pdf")
    return None

def plot_three_test_set(data,root_dir,classOC=0,meanBox=None,medianBox=None,mode=None):
    """Prints subset of data which model is tested on
    Args: 
        input: 
            data: saved list-of-list-of-list containing model-data and image-names
                Format: [0: filename,
                         1: prediction-result (0:false,1:correct),
                         2: IOU-score
                         4: target [tensor]
                         5: prediction [tensor],
                         6: image-dimensions in h,w]
            root_dir: imagedir
            classOC: descriptor of class which is used
            meanBox: box with means of trainset
            medianBox: box with medians of trainset
        returns: 
            None
    """      
            
    NSAMPLES = 3
    NCOLS = 3
    if(mode!=None):
        tmpData = []
        for i in range(len(data)):
            if(mode=="success"):
                if(int(data[i][1])==1):
                    tmpData.append(data[i])
            elif(mode=="failure"):
                if(int(data[i][1])==0):
                    tmpData.append(data[i])
        data = tmpData
        del tmpData
        
        
    data = sample(data,NSAMPLES)
    NCOLS = 3
    
    fig, ax = plt.subplots(math.ceil(NSAMPLES/NCOLS),NCOLS,figsize=(13,6))
    col = 0
    row = 0
    for entry in range(NSAMPLES):
        filename = data[entry][0]
        im = plt.imread(root_dir+filename)
        target,preds = bbox_for_plot(data[entry])
        if(meanBox!=None):
            mean_box = single_format_bbox(meanBox, data[entry])
            rectM = patches.Rectangle((mean_box[0],mean_box[1]),mean_box[2],mean_box[3],linewidth=2,edgecolor='c',facecolor='none')
        if(medianBox!=None):
            median_box = single_format_bbox(medianBox,data[entry])
            rectMed = patches.Rectangle((median_box[0],median_box[1]),median_box[2],median_box[3],linewidth=2,edgecolor='lightcoral',facecolor='none')
        rectT = patches.Rectangle((target[0],target[1]), target[2], target[3], linewidth=2, edgecolor='r', facecolor='none')
        rectP = patches.Rectangle((preds[0],preds[1]), preds[2], preds[3], linewidth=2, edgecolor='m', facecolor='none')
        
        if entry%NCOLS==0 and entry!=0: #NUMBER OF COLS
            row += 1 
            col = 0 
        print("Col: ",col,"Row:",row)
        ax[col].imshow(im)
        ax[col].tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
        ax[col].add_patch(rectT)
        ax[col].add_patch(rectP)
        legend_tuple = (rectT,rectP)
        legend_name_tuple = ("Target","Prediction")
        legend_ncol = 2
        
        if(meanBox!=None):
            ax[col].add_patch(rectM)
            new_legend_tuple = legend_tuple + (rectM,)
            new_legend_name_tuple = legend_name_tuple + ("Mean box",)
            legend_tuple = new_legend_tuple 
            legend_name_tuple = new_legend_name_tuple
            legend_ncol += 1
            del new_legend_tuple,new_legend_name_tuple
            
        if(medianBox!=None):
            ax[col].add_patch(rectMed)
            new_legend_tuple = legend_tuple + (rectMed,)
            new_legend_name_tuple = legend_name_tuple + ("Median box",)
            legend_tuple = new_legend_tuple 
            legend_name_tuple = new_legend_name_tuple
            legend_ncol += 1
            del new_legend_tuple,new_legend_name_tuple
            
        
        ax[col].text(0.02,0.05,"IOU:{:.2f}".format(data[entry][2]),bbox=dict(facecolor='magenta', alpha=0.75),transform=ax[col].transAxes)
        ax[col].set_title(filename,fontweight="bold",size=8)
        #ax[row,col].text(preds[2],preds[3],"{:.2f}".format(data[entry][2][0].item()),bbox=dict(facecolor='magenta', alpha=0.5),transform=ax[row,col].transAxes)
        col += 1
    plt.rcParams['legend.handlelength'] = 1
    plt.rcParams['legend.handleheight'] = 1.125
    fig.legend(legend_tuple,legend_name_tuple,loc="upper center",ncol=legend_ncol,framealpha=0.0,bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("Predictions on subset of testset with mode: "+mode)
    plt.show()
    plt.savefig("class_"+str(classOC)+"eval_on_testnset"+mode+".pdf")
    
    return None
            
def plot_test_set(data,root_dir,classOC=0,meanBox=None,medianBox=None,mode=None):
    """Prints subset of data which model is tested on
    Args: 
        input: 
            data: saved list-of-list-of-list containing model-data and image-names
                Format: [0: filename,
                         1: prediction-result (0:false,1:correct),
                         2: IOU-score
                         4: target [tensor]
                         5: prediction [tensor],
                         6: image-dimensions in h,w]
            root_dir: imagedir
            classOC: descriptor of class which is used
            meanBox: box with means of trainset
            medianBox: box with medians of trainset
        returns: 
            None
    """      
            
    NSAMPLES = len(data)
    NCOLS = 2
    if NSAMPLES > 9: 
        NSAMPLES = 9
        if(mode!=None):
            tmpData = []
            for i in range(len(data)):
                if(mode=="success"):
                    if(int(data[i][1])==1):
                        tmpData.append(data[i])
                elif(mode=="failure"):
                    if(int(data[i][1])==0):
                        tmpData.append(data[i])
            data = tmpData
            del tmpData
            
        
    data = sample(data,NSAMPLES)
    NCOLS = 3
    
    fig, ax = plt.subplots(math.ceil(NSAMPLES/NCOLS),NCOLS,figsize=(14,12))
    col = 0
    row = 0
    for entry in range(NSAMPLES):
        filename = data[entry][0]
        im = plt.imread(root_dir+filename)
        target,preds = bbox_for_plot(data[entry])
        if(meanBox!=None):
            mean_box = single_format_bbox(meanBox, data[entry])
            rectM = patches.Rectangle((mean_box[0],mean_box[1]),mean_box[2],mean_box[3],linewidth=2,edgecolor='c',facecolor='none')
        if(medianBox!=None):
            median_box = single_format_bbox(medianBox,data[entry])
            rectMed = patches.Rectangle((median_box[0],median_box[1]),median_box[2],median_box[3],linewidth=2,edgecolor='lightcoral',facecolor='none')
        rectT = patches.Rectangle((target[0],target[1]), target[2], target[3], linewidth=3, edgecolor='r', facecolor='none')
        rectP = patches.Rectangle((preds[0],preds[1]), preds[2], preds[3], linewidth=3, edgecolor='m', facecolor='none')
        
        if entry%NCOLS==0 and entry!=0: #NUMBER OF COLS
            row += 1 
            col = 0 
        print("Col: ",col,"Row:",row)
        ax[row,col].imshow(im)
        ax[row,col].tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
        ax[row,col].add_patch(rectT)
        ax[row,col].add_patch(rectP)
        legend_tuple = (rectT,rectP)
        legend_name_tuple = ("Target","Prediction")
        legend_ncol = 2
        
        if(meanBox!=None):
            ax[row,col].add_patch(rectM)
            new_legend_tuple = legend_tuple + (rectM,)
            new_legend_name_tuple = legend_name_tuple + ("Mean box",)
            legend_tuple = new_legend_tuple 
            legend_name_tuple = new_legend_name_tuple
            legend_ncol += 1
            del new_legend_tuple,new_legend_name_tuple
            
        if(medianBox!=None):
            ax[row,col].add_patch(rectMed)
            new_legend_tuple = legend_tuple + (rectMed,)
            new_legend_name_tuple = legend_name_tuple + ("Median box",)
            legend_tuple = new_legend_tuple 
            legend_name_tuple = new_legend_name_tuple
            legend_ncol += 1
            del new_legend_tuple,new_legend_name_tuple
            
        
        ax[row,col].text(0.02,0.05,"IOU:{:.2f}".format(data[entry][2]),fontweight="bold",bbox=dict(facecolor='magenta', alpha=0.75),transform=ax[row,col].transAxes)
        #ax[row,col].set_title(filename,fontweight="bold",size=8)
        #ax[row,col].text(preds[2],preds[3],"{:.2f}".format(data[entry][2][0].item()),bbox=dict(facecolor='magenta', alpha=0.5),transform=ax[row,col].transAxes)
        col += 1
    plt.rcParams['legend.handlelength'] = 1
    plt.rcParams['legend.handleheight'] = 1.125
    plt.tight_layout(rect=(0,0,1,0.90))
    fig.legend(legend_tuple,legend_name_tuple,loc="upper center",ncol=legend_ncol,prop={'size':22,'weight':"bold"},framealpha=0.0,bbox_to_anchor=(0.5, 0.97))
    #fig.suptitle("Predictions on subset of testset with mode: "+mode)
    plt.savefig("results_on_test_"+str(mode)+".pdf")
    plt.show()
    return None

def plot_best_set(data,root_dir,classOC=0,meanBox=None,medianBox=None,n=9):
    """Prints subset of data which model is tested on
    Args: 
        input: 
            data: saved list-of-list-of-list containing model-data and image-names
                Format: [0: filename,
                         1: prediction-result (0:false,1:correct),
                         2: IOU-score
                         4: target [tensor]
                         5: prediction [tensor],
                         6: image-dimensions in h,w]
            root_dir: imagedir
            classOC: descriptor of class which is used
            meanBox: box with means of trainset
            medianBox: box with medians of trainset
        returns: 
            None
    """      
            
    NSAMPLES = len(data)
    #get 9 best performing samples
    if NSAMPLES > 9: 
        NSAMPLES = 12
        tmpData = []
        maxIOU_li = []
        maxIOU_IDX = []
        for i in range(len(data)):
            if(i>=0 and i<12):
                maxIOU_li.append(float(data[i][2]))
                maxIOU_IDX.append(i)
                tmpData.append(data[i])
            else:
                if(float(data[i][2])>min(maxIOU_li)):
                    tmpData.append(data[i])
                    minVal = min(maxIOU_li)
                    minIdx = maxIOU_li.index(minVal)
                    maxIOU_li[minIdx] = float(data[i][2])
                    maxIOU_IDX[minIdx] = i
                    tmpData[minIdx] = data[i]
        data = tmpData
        del tmpData
            
   # data = sample(data,NSAMPLES)
    if(n==12):
        NCOLS = 3
    else:
        NCOLS = 3
        
    fig, ax = plt.subplots(math.ceil(NSAMPLES/NCOLS),NCOLS,figsize=(12,14))
    col = 0
    row = 0
    for entry in range(NSAMPLES):
        filename = data[entry][0]
        im = plt.imread(root_dir+filename)
        target,preds = bbox_for_plot(data[entry])
        if(meanBox!=None):
            mean_box = single_format_bbox(meanBox, data[entry])
            rectM = patches.Rectangle((mean_box[0],mean_box[1]),mean_box[2],mean_box[3],linewidth=2,edgecolor='c',facecolor='none')
        if(medianBox!=None):
            median_box = single_format_bbox(medianBox,data[entry])
            rectMed = patches.Rectangle((median_box[0],median_box[1]),median_box[2],median_box[3],linewidth=2,edgecolor='lightcoral',facecolor='none')
        rectT = patches.Rectangle((target[0],target[1]), target[2], target[3], linewidth=3, edgecolor='r', facecolor='none')
        rectP = patches.Rectangle((preds[0],preds[1]), preds[2], preds[3], linewidth=3, edgecolor='m', facecolor='none')
        
        if entry%NCOLS==0 and entry!=0: #NUMBER OF COLS
            row += 1 
            col = 0 
        print("Col: ",col,"Row:",row)
        ax[row,col].imshow(im)
        ax[row,col].tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
        ax[row,col].add_patch(rectT)
        ax[row,col].add_patch(rectP)
        legend_tuple = (rectT,rectP)
        legend_name_tuple = ("Target","Prediction")
        legend_ncol = 2
        
        if(meanBox!=None):
            ax[row,col].add_patch(rectM)
            new_legend_tuple = legend_tuple + (rectM,)
            new_legend_name_tuple = legend_name_tuple + ("Mean box",)
            legend_tuple = new_legend_tuple 
            legend_name_tuple = new_legend_name_tuple
            legend_ncol += 1
            del new_legend_tuple,new_legend_name_tuple
            
        if(medianBox!=None):
            ax[row,col].add_patch(rectMed)
            new_legend_tuple = legend_tuple + (rectMed,)
            new_legend_name_tuple = legend_name_tuple + ("Median box",)
            legend_tuple = new_legend_tuple 
            legend_name_tuple = new_legend_name_tuple
            legend_ncol += 1
            del new_legend_tuple,new_legend_name_tuple
            
        
        ax[row,col].text(0.02,0.05,"IOU:{:.2f}".format(data[entry][2]),fontweight="bold",bbox=dict(facecolor='magenta', alpha=0.75),transform=ax[row,col].transAxes)
        #ax[row,col].set_title(filename,fontweight="bold",size=8)
        #ax[row,col].text(preds[2],preds[3],"{:.2f}".format(data[entry][2][0].item()),bbox=dict(facecolor='magenta', alpha=0.5),transform=ax[row,col].transAxes)
        col += 1
    plt.rcParams['legend.handlelength'] = 1
    plt.rcParams['legend.handleheight'] = 1.125
    plt.tight_layout(rect=(0,0,1,0.90),h_pad=0.1,w_pad=0.1)
    fig.legend(legend_tuple,legend_name_tuple,loc="upper center",ncol=legend_ncol,prop={'size':22,'weight':"bold"},framealpha=0.0,bbox_to_anchor=(0.5, 0.97))
    #fig.suptitle("Predictions on subset of testset with mode: "+mode)
    plt.savefig("max_results_on_test.pdf")
    plt.show()
    return None

def debatch_data(data):
    """You accidentally saved the whole batch per evaluation-point. 
    Thus you repeat a whole batch of targets, preds, imdims, filenames, but have correctly saved IOU[entry] and CorLoc[entry]
    This function gets out matching elements 
    """
    holder_li = []
    BC = 0 
    for i in range(len(data)):
        if(BC==8):
            BC = 0
        filename = data[i][0][BC]
        imdims = data[i][-1][BC]
        target = data[i][3][BC]
        preds = data[i][4][BC]
        locScore = data[i][1]
        IOU = data[i][2]
        holder_li.append([filename,locScore,IOU,target,preds,imdims])
        BC += 1 
    return holder_li
        
def get_size_single_box(box):
    x0 = float(box[0])
    y0 = float(box[1])
    w = box[2] - x0 #typecast automatically
    h = box[3] - y0
    return (x0,y0),w.item(),h.item()

def get_relative_area(boxsize,imdims):
    "takes in tuple of w h of target box and image dims and returns relative area of box"
    area_box = boxsize[0]*imdims[1]*boxsize[1]*imdims[0]
    imArea = float(imdims[0])*float(imdims[1])
    return area_box/imArea
    
    

def performance_across_bins(li):
    "Takes in list in standard-format, gets relative areas, calculates even bins, and evaluates performance in accordance to bins"
    import numpy as np 
    rel_areas = []
    for entry in li: 
        target = entry[3] #format corner-coordinates
        imdims = entry[-1]
        _,w,h = get_size_single_box(target)
        REL = get_relative_area((w,h),imdims)
        rel_areas.append(REL)
    
    rel_areas.sort() # inplace
    bins = np.array_split(rel_areas,3)
    binedges = [max(bins[0]),min(bins[1]),max(bins[1]),min(bins[2])]
    binString = "{0:"+str(binedges[0])+"},{"+str(binedges[1])+":"+str(binedges[2])+"},{"+str(binedges[2])+":1}"
    performanceDict = {"binedges": binString,
                       "descriptor":["NoCorrect","NoFalse","Total Performance"],
                       "Small":[0,0,0],
                       "Medium":[0,0,0],
                       "Large":[0,0,0]}
    for i, entry in enumerate(li): 
        target = entry[3]
        imdims = entry[-1]
        _,w,h = get_size_single_box(target)
        REL = get_relative_area((w,h),imdims)
        if(REL>=0 and REL<=binedges[0]): #small group
            group = "Small"
        elif(REL>=binedges[1] and REL <=binedges[2]):
            group = "Medium"
        else:
            group = "Large"
        
        if(entry[1]==str(1)):
            performanceDict[group][0] += 1
        elif(entry[1]==str(0)):
            performanceDict[group][1] += 1
        else:
            print("Error. No Loc entry found in: ",i)
    
    performanceDict["Small"][-1] = performanceDict["Small"][0]/(performanceDict["Small"][0]+performanceDict["Small"][1])
    performanceDict["Medium"][-1] = performanceDict["Medium"][0]/(performanceDict["Medium"][0]+performanceDict["Medium"][1])
    performanceDict["Large"][-1] = performanceDict["Large"][0]/(performanceDict["Large"][0]+performanceDict["Large"][1])
    
    print("\n------ Stratufued Results: ----------\n")    
    print(performanceDict)        
        
    return performanceDict
    
def performance_across_classes(li):
    classes = ["aeroplane","bicycle","boat","cat","cow","diningtable","dog","horse","motorbike","sofa"]
    performanceDict = {}
    for name in classes: 
        performanceDict[name] = [0,0,0]
        
    for i, entry in enumerate(li): 
        fname = entry[0]
        for name in classes:
            if name in fname:
                if(entry[1]==str(1)):
                    performanceDict[name][0] += 1
                elif(entry[1]==str(0)):
                    performanceDict[name][1] += 1
                else:
                    print("Error. No Loc entry found in: ",i)
                break 
            else:
                pass
                #print("Error, no class found for entry ",i)
    for key in performanceDict: 
        performanceDict[key][-1] = performanceDict[key][0]/(performanceDict[key][0]+performanceDict[key][1])
        
    print("\n ------- Classwise results generated -----------\n")
    print(performanceDict)
    return performanceDict

def performance_across_number_of_boxes(li):
    identity = torch.load("multiple_box_struct.pth")
    performanceDict = {"Single box":[0,0,0],"Multiple boxes":[0,0,0]}
    for entry in li: 
        fname = entry[0]
        for i in range(len(identity)):
            if identity[i][0] == fname: 
                nboxes = identity[i][1]
            else:
                pass
        if(nboxes == 1):
            key = "Single box"
        elif(nboxes > 1):
            key= "Multiple boxes"
        else:
            print("Error, no number of boxes identified for ",fname)
        if(entry[1]==str(1)):
            performanceDict[key][0] += 1
        elif(entry[1]==str(0)):
            performanceDict[key][1] += 1
    for key in performanceDict: 
        performanceDict[key][-1] = performanceDict[key][0]/(performanceDict[key][0]+performanceDict[key][1])
    print("\n ------------Performance dependant on number of boxes in sample--------------\n")
    print(performanceDict)
    return performanceDict  

def fixational_statistics(mode="all"):
    liC = torch.load("fixational_statistics.pth") #list in format ([class,number of fixes (concat LE;RE),participant])
    liR = torch.load("fixational_statistics_before_cleanup.pth")    #list in format ([class,number of fixes (concat LE;RE),participant])
    classes = ["aeroplane","bicycle","boat","cat","cow","diningtable","dog","horse","motorbike","sofa"]
    num_ims = [666+47,536+38,504+35,1051+74,301+21,498+35,1257+88,480+34,510+36,467+33]
    fixDict = {}
    for name in classes: 
        fixDict[name] = [0,0,0,0,0] #{format: total number of cleaned fixes, total number of raw fixes, min number of cleaned fixes, max number of cleaned fixes,average number of fixes}
    if mode=="all":
        for i, entry in enumerate(liC):
            if(i==0):
                min_tmp = entry[1]
                max_tmp = entry[1]
            name = entry[0]
            fixDict[name][0] += entry[1]
            if entry[1] < min_tmp: 
                min_tmp = entry[1]
                fixDict[name][2] = min_tmp
            elif entry[1] > max_tmp: 
                max_tmp = entry[1]
                fixDict[name][3] = max_tmp
        for entry in liR: 
            name = entry[0]
            fixDict[name][1] += entry[1]
        for i, name in enumerate(classes):
            fixDict[name][-1] = fixDict[name][0]/(5*num_ims[i])
        
    if mode=="first":
        for i, entry in enumerate(liC):
            if(i==0):
                min_tmp = entry[1]
                max_tmp = entry[1]
            name = entry[0]
            if(entry[2] == 0):
                fixDict[name][0] += entry[1]
                if entry[1] < min_tmp: 
                    min_tmp = entry[1]
                    fixDict[name][2] = min_tmp
                elif entry[1] > max_tmp: 
                    max_tmp = entry[1]
                    fixDict[name][3] = max_tmp
        
        for entry in liR:
            name = entry[0]
            if entry[2]==0:
                fixDict[name][1] += entry[1]
        for i, name in enumerate(classes):
            fixDict[name][-1] = fixDict[name][0]/(num_ims[i])
    if(mode=="all"):
        print("\n...Generated statistics for all participants...\n")
    else:
        print("\n...Generated statistics for first participant...\n")
    print("Format: number of cleaned fixes, number of raw fixes, min number of fixes, max number of fixes, average number of fixes per image per participant\n")
    print(fixDict)
    
    print("\n...Average number of fixes across all classes...\n")
    seqsum = 0
    for i, key in enumerate(fixDict):
        seqsum += fixDict[key][0]
    print(seqsum/10)
    
    print("\n ...Total number of cleaned fixes...")
    seqsum = 0
    for i, key in enumerate(fixDict):
        seqsum += fixDict[key][1]-fixDict[key][0]
    print(seqsum)
    
    print("\n...Total number of fixes kept...")
    seqsum = 0
    for i, key in enumerate(fixDict):
        seqsum += fixDict[key][0]
    print(seqsum)
    
    print("\n...Total number of raw fixes...\n")
    seqsum = 0
    for i, key in enumerate(fixDict):
        seqsum += fixDict[key][1]
    print(seqsum)
    
    return fixDict

def IOU_performance_areas_histogram_equal_bins(li):
    "Takes in list in standard-format, gets relative areas, calculates even bins, and evaluates performance in accordance to bins"
    import numpy as np
    rel_areas = []
    for entry in li: 
        target = entry[3] #format corner-coordinates
        imdims = entry[-1]
        _,w,h = get_size_single_box(target)
        REL = get_relative_area((w,h),imdims)
        rel_areas.append(REL)
    
    rel_areas.sort() # inplace
    bins = np.array_split(rel_areas,10)
    #bins = np.linspace(0.1,1,10)
    binedges = [[min(x),max(x)] for x in bins]
    group_IOU = np.zeros((len(li),len(binedges)))
    no_wrong = np.zeros(len(binedges))
    no_correct = np.zeros(len(binedges))
    for i, entry in enumerate(li): 
        target = entry[3]
        imdims = entry[-1]
        _,w,h = get_size_single_box(target)
        REL = get_relative_area((w,h),imdims)
        for idx, group in enumerate(binedges): 
            if(REL>=group[0] and REL <= group[1]):
                groupidx = idx
            else:
                pass
        group_IOU[i,groupidx] = entry[2]    
        if(entry[1]==str(1)):
            no_correct[groupidx] += 1
        if(entry[1]==str(0)):
            no_wrong[groupidx] += 1
    
    group_IOU[group_IOU==0] = np.nan
    meanIOU = np.nanmean(group_IOU,axis=0)
    meanCL = np.divide(no_correct,(no_correct+no_wrong))
    
    fig, ax = plt.subplots(1,2)
    x = np.linspace(1,10,len(binedges))
    ax[0].bar(x,meanIOU)
    ax[1].bar(x,meanCL)
    plt.show()
    #np.histogram(group_IOU)
    
    return meanIOU,meanCL   

def IOU_performance_areas_histogram(li):
    "Takes in list in standard-format, gets relative areas, calculates even bins, and evaluates performance in accordance to bins"
    import numpy as np
    from matplotlib import ticker
    
    rel_areas = []
    for entry in li: 
        target = entry[3] #format corner-coordinates
        imdims = entry[-1]
        _,w,h = get_size_single_box(target)
        REL = get_relative_area((w,h),imdims)
        rel_areas.append(REL)
    
    rel_areas.sort() # inplace
    #bins = np.array_split(rel_areas,10)
    bins = np.linspace(0.1,1,10)
    #binedges = [[min(x),max(x)] for x in bins]
    group_IOU = np.zeros((len(li),len(bins)))
    no_wrong = np.zeros(len(bins))
    no_correct = np.zeros(len(bins))
    for i, entry in enumerate(li): 
        target = entry[3]
        imdims = entry[-1]
        _,w,h = get_size_single_box(target)
        REL = get_relative_area((w,h),imdims)
        for idx, group in enumerate(bins): 
            if(idx==0):
                if(REL<=group):
                    groupidx = idx
            else:
                if(REL<=group and REL>=bins[idx-1]):
                    groupidx = idx
        
        group_IOU[i,groupidx] = entry[2]    
        if(entry[1]==str(1)):
            no_correct[groupidx] += 1
        if(entry[1]==str(0)):
            no_wrong[groupidx] += 1
    
    group_IOU[group_IOU==0] = np.nan
    meanIOU = np.nanmean(group_IOU,axis=0)
    meanCL = np.divide(no_correct,(no_correct+no_wrong))
    
    fig, ax = plt.subplots(1,2,sharey=True,sharex=True,figsize=(16,6))
    
    x = np.arange(10)
    M = 10
    yticks = ticker.MaxNLocator(M)
    
    plt.tight_layout(rect=(0.01,0.01,1,1),w_pad=3)
    ax[0].set_ylim((0.0,0.97))
    ax[0].bar(x,meanIOU,align="edge")
    ax[1].bar(x,meanCL,align="edge")
    positions = [x for x in range(0,11)]
    labels = ['0','0.1', '0.2', '0.3', '0.4', '0.5', '0.6','0.7','0.8','0.9','1']
    ax[0].xaxis.set_major_locator(ticker.FixedLocator(positions))
    ax[0].xaxis.set_major_formatter(ticker.FixedFormatter(labels))
    ax[1].xaxis.set_major_locator(ticker.FixedLocator(positions))
    ax[1].xaxis.set_major_formatter(ticker.FixedFormatter(labels))
    ax[0].set_xlabel("Bounding box relative area",fontsize=22,weight='bold')
    ax[1].set_xlabel("Bounding box relative area",fontsize=22,weight='bold')
    ax[0].set_ylabel("Mean IOU score",fontsize=22,weight='bold')
    ax[1].set_ylabel("CorLoc score",fontsize=22,weight='bold')
    ax[1].yaxis.set_tick_params(labelleft=True)
    ax[0].yaxis.set_major_locator(yticks)
    ax[1].yaxis.set_major_locator(yticks)
    #ax[0].grid(zorder=0)
    #ax[1].grid(zorder=0)
    ax[0].grid(color='black', which='major', axis='y', linestyle='solid')
    ax[1].grid(color='black', which='major', axis='y', linestyle='solid')
    
    plt.savefig("relative_areas_performance_histograms.pdf")
    #plt.show()
    
    
    return meanIOU,meanCL   
    
def distribution_of_multibox_images():
    identity = torch.load("multiple_box_struct.pth")
    classes = ["aeroplane","bicycle","boat","cat","cow","diningtable","dog","horse","motorbike","sofa"]
    boxDict = {}
    for name in classes: 
        boxDict[name] = [0,0,0,0] #{format: single boxes, multiple boxes, total number of boxes,percentage multiple}
    
    for entry in identity: 
        fname = entry[0]
        for name in classes:
            if(name in fname):
                key = name
                if(entry[1]==1):
                    boxDict[key][0] += 1
                    boxDict[key][-2] += 1
                elif(entry[1]>1):
                    boxDict[key][1] += 1
                    boxDict[key][-2] += entry[1]
            else:
                pass
    bar_li = []
    for name in classes:
        boxDict[name][-1] = boxDict[name][2]/(boxDict[name][0]+boxDict[name][1])
        bar_li.append(boxDict[name][-1])
        print("\n {}{}".format(name,boxDict[name]))
    plt.figure(figsize=(18,10))
    plt.bar(x=classes,height=bar_li)
    plt.ylabel("Average number of bounding boxes",fontsize=24,weight="bold")
    plt.yticks(fontsize=22,weight="bold")
    plt.xticks(fontsize=22,weight="bold")
    plt.tight_layout()
    plt.savefig("fraction_multiple_boxes.pdf")
    
    #print(boxDict)
    return boxDict
    
def dim_result_scatter():
    import numpy as np
    import matplotlib.cm as cm 
    import numpy as np
    #from matplotlib.ticker import FormatStrFormatter
    classes = ["aeroplane","bicycle","boat","cat","cow","diningtable","dog","horse","motorbike","sofa"]
    dim_R = [0.616,0.41,0.311,0.586,0.318,0.529,0.453,0.413,0.538,0.514]
    max_R = [0.645,0.462,0.301,0.619,0.432,0.492,0.502,0.511,0.540,0.491]
    colors = cm.tab20(np.linspace(0,1,len(dim_R)))
    fig = plt.figure(figsize=(14,9))
    x = np.linspace(min(dim_R)-0.02,max(max_R)+0.01,1000)
    plt.plot(x,x,linestyle='dashed',color="red")
    for i, entry in enumerate(dim_R):
        scale = 600
        plt.scatter(entry, max_R[i], c=colors[i], s=scale, label=classes[i],
                   alpha=1, edgecolors='none')

    #scatter = plt.scatter(dim_R,max_R,c=colors)
    plt.xlabel("Dim. et al. class CorLoc",fontsize=30,weight="bold")
    plt.ylabel("EF class CorLoc",fontsize=30,weight="bold")
    plt.yticks(fontsize=22,weight="bold")
    plt.xticks(fontsize=22,weight="bold")
    #legend1 = plt.legend(*scatter.legend_elements(),
                    #loc="lower right", title="Class")
    #ax.add_artist(legend1)
    plt.tight_layout()
    plt.legend(ncol=1,loc="lower right",title="Class",fontsize=22)
    plt.grid('on')
    fig.savefig("class_comparison.pdf",dpi=fig.dpi)
    
    
    
testdata = load_results(dpath,className,"test_on_test_results")
testdata = debatch_data(testdata)
#GET PERFORMANCE EVALUATED ACROSS NUMBER OF PARAMETERS
#_ = performance_across_bins(testdata)
#_ = performance_across_classes(testdata)
#_ = performance_across_number_of_boxes(testdata)
#_ = fixational_statistics()
_ = distribution_of_multibox_images()

#testOnTrain = load_results(dpath,className,"test_on_train_results")
#testOnTrain = debatch_data(testOnTrain)
#mean = mean_model(testOnTrain)
#median = median_model(testOnTrain)
#plot_best_set(testdata,impath,classOC="all_classes",meanBox=mean,medianBox=median,n=12)
#out = IOU_performance_areas_histogram(testdata)


#if wanna redo plots for report 
#mean = mean_model(testOnTrain)
#median = median_model(testOnTrain)
#plot_train_set(testOnTrain,impath)
#plot_test_set(testdata,impath,classOC="all_classes",meanBox=mean,medianBox=median,mode="success")
#plot_test_set(testdata,impath,classOC="all_classes",meanBox=mean,medianBox=median,mode="failure")



#plot_three_train_set(testOnTrain,impath,classOC="all_classes")
#plot_three_test_set(testdata,impath,classOC="all_classes",meanBox=mean,medianBox=median,mode="failure")




def extract_specific_image_data(imagename,dataset):
    for i, entry in enumerate(dataset):
            try:
                entry[0].index(imagename)
                return entry
            except ValueError:
                pass
    print("Not found in dataset")
    return None
        
    
extract_specific_image_data("motorbike_2008_002047.jpg",testdata)

       
      
