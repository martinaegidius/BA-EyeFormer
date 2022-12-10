#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 08:38:04 2022

@author: max
"""
import os 
import scipy 
import numpy as np
import tqdm
import random
import matplotlib.pyplot as plt
from matplotlib import image
import cv2
import copy
#from plotting_lib import extract_specific_image_data
import torch

DEBUG = True

class pascalET():
    classes = ["aeroplane","bicycle","boat","cat","cow","diningtable","dog","horse","motorbike","sofa"]
    p = os.path.dirname((__file__))
    matfiles = []
    etData = []
    eyeData = None
    filename = None
    im_dims = None
    chosen_class = None
    NUM_TRACKERS = 5
    ratios = None
    class_counts = None
    pixel_num = None
    fix_nums = None
    classwise_num_pixels = None
    classwise_ratios = None
    bbox = None #old solution. Deprecated and should be deleted at some point
    num_files_in_class = []
    eyeData_stats = None
    bboxes = None #new solution
    debug_box_BP = []
    chosen_bbox = None
    
    
    
    def loadmat(self,root_dir = None):
        for name in self.classes: 
            if(root_dir==None):
                self.matfiles.append(self.p+"/Data/POETdataset/etData/"+"etData_"+name+".mat")
            else: 
                self.matfiles.append(root_dir+"/Data/POETdataset/etData/"+"etData_"+name+".mat")
        for file in self.matfiles: 
            A = scipy.io.loadmat(file,squeeze_me=True,struct_as_record=False)
            self.etData.append(A["etData"])
            #etData[0].fixations[0].imgCoord.fixR.pos
        
        #self.eyeData = 
    
    def convert_eyetracking_data(self,CLEANUP: bool,STATS: bool,num=[x for x in range(10)]):
        """Takes in mat-format and instead makes usable format.
            Args: 
                CLEANUP: Bool. If true, remove all invalid fixations (outside of image)
                STATS: Bool. If true, save statistics of eyeTracking-points
                
            Returns: 
                Nothing. 
                Mutates object instance by filling :
                    self.eyeData-matrix
                if STATS == True additionally mutates:     
                    self.eyeData_stats (counts number of fixations in image and in bbox)
                    self.etData.gtbb overwritten with BEST box
                    
        """
        
        #self.eyeData = np.empty((self.etData[num].shape[0],self.NUM_TRACKERS,1),dtype=object)
        #num = [x for x in range(10)] #classes
        
        #get maximal number of images in class for creating arrays which can hold all data: 
        max_dim = 0
        for cN in num: 
            cDim = len(self.etData[cN])
            if(cDim>max_dim):
                max_dim=cDim
            self.num_files_in_class.append(cDim) #append value to structure, important for later slicing. 
        
        #allocate arrays
        self.eyeData = np.empty((len(num),max_dim,self.NUM_TRACKERS),dtype=object) #format: [num_classes,max(num_images),num_trackers]. size: [9,1051,5,1] for complete dset. Last index holds list of eyetracking for person
        self.im_dims = np.empty((len(num),max_dim,2))
        self.bboxes = np.empty((len(num),max_dim),dtype=object) #has to be able to save lists of arrays, as some images have multiple bboxes. 
        self.chosen_box = np.empty((len(num),max_dim),dtype=object)

        if(STATS==True): #for eyetracking-statistics
            num_stats = 2 #number of fixes in bbox, number of fixes on image 
            #old self.eyeData_stats = np.empty((len(num),max_dim,self.NUM_TRACKERS,num_stats)) #format: [classNo, imageNo, personNo, 2:(number of fixes, number of fixes in bbox)]
            self.eyeData_stats = np.empty((len(num),max_dim,num_stats)) #format: [classNo, imageNo, 2:(number of fixes in img, number of fixes in bbox)]

        for cN in num: #class-number
            self.debug_box_BP = [] #reset at every new class
            for i in range(len(self.etData[cN])):
                im_dims = self.etData[cN][i].dimensions[:2]
                #print("Im dims: ",im_dims[0],im_dims[1])
                self.im_dims[cN,i,:] = im_dims[:]
                self.bboxes[cN,i] = [self.etData[cN][i].gtbb]
                
                
                
                for k in range(self.NUM_TRACKERS): #loop for every person looking
                    NOFIXES = False 
                    fixes_counter = 0  #reset image-wise #atm unused
                    fixes_in_bbox_counter = 0 #atm unused
                    #print(cN,i,k)
                    #w_max = self.im_dims[i][1] #for removing irrelevant points
                    #h_max = self.im_dims[i][0]
                    LP = self.etData[cN][i].fixations[k].imgCoord.fixL.pos[:]
                    RP = self.etData[cN][i].fixations[k].imgCoord.fixR.pos[:]
                    BP = np.vstack((LP,RP)) #LP|RP
                    if(BP.shape[0] == 0 or BP.shape[1]==0):
                        NOFIXES = True #necessary flag as array else is (0,2) ie not None even though is empty
                    
                    if(CLEANUP == True and NOFIXES == False): #necessary to failcheck; some measurements are erroneus. vstack of two empty arrs gives BP.shape=(2,0)
                        BP = np.delete(BP,np.where(np.isnan(BP[:,0])),axis=0)
                        BP = np.delete(BP,np.where(np.isnan(BP[:,1])),axis=0)
                        BP = np.delete(BP,np.where((BP[:,0]<0)),axis=0) #delete all fixations outside of image-quadrant
                        BP = np.delete(BP,np.where((BP[:,1]<0)),axis=0) #delete all fixations outside of image-quadrant
                        BP = np.delete(BP,np.where((BP[:,0]>im_dims[1])),axis=0) #remove out of images fixes on x-scale. Remember: dimensions are given as [y,x]
                        BP = np.delete(BP,np.where((BP[:,1]>im_dims[0])),axis=0) #remove out of images fixes on y-scale
                        
                        
                    self.eyeData[cN,i,k] = [BP] #fill with matrix as list #due to variable size    
                    
                    if(k==0): #create fixArr
                        if(BP.shape==(2,0)):
                            fixArr = copy.deepcopy(np.transpose(BP)) #invalid measurements are for some reason stored as shape (0,2) (transpose of other measurements)
                        else:
                            fixArr = copy.deepcopy(BP)
                    else:
                        if(BP.shape[1] == fixArr.shape[1]): #necessary check as None array can not be concat
                            fixArr = np.vstack((fixArr,BP)) 
                        else: 
                            pass
                    
                    del BP 
                self.debug_box_BP.append(fixArr)
                #probably this part needs to go into function for itself, and it needs to go out of inner-loop!
                if(STATS==True):
                    tmp_bbox = self.etData[cN][i].gtbb #for STATS
                    if(NOFIXES == False): #NOFIXES True if zero fixes in image across all participants
                        #fixes_counter += fixArr[i].shape[0] #atm unused
                        self.eyeData_stats[cN,i,0] = int(fixArr.shape[0]) #number of fixes in total saved in col 0. Number of fixes in total is length of fixArr-array
                        xs,ys,w,h = self.get_bounding_box(tmp_bbox,fixArr) #look comment line 135
                        self.chosen_box[cN][i] = [xs,ys,w,h]
                        self.etData[cN][i].gtbb = np.array([xs,ys,xs+w,ys+h]) #use broadcast - NOTE OVERWRITES BBOX TO SINGLE
                        nbbx = [xs,ys,w,h]
                        self.eyeData_stats[cN,i,1] = self.get_num_fix_in_bbox(nbbx,fixArr)  #NEED, but removed for debugging
                    else:
                        self.eyeData_stats[cN,i,0] = 0
                        self.eyeData_stats[cN,i,1] = 0
                del fixArr #after each image, reset fixArr
                
        
    
    #def get_ground_truth(self):
    def get_bounding_box(self,inClass,fixArr=None,DEBUG=None): #Args: inClass: bounding-box-field. Type: array. fixArr called when called from bbox-stats module in order to maximize fixes in bbox of choice.
        #convert to format for patches.Rectangle; it wants anchor point (upper left), width, height
        #print("Input-array: ",inClass)
        if(DEBUG):
            print("Failing in: ", DEBUG)
        if (isinstance(inClass,np.ndarray) and isinstance(fixArr,np.ndarray)):
            if fixArr.any(): #check if is initialized, ie. called from method which wants to maximise bbox-hits.
                if(inClass.ndim>1): #If more than one bounding box
                    boxidx = self.maximize_fixes(inClass,fixArr)
                    inClass = inClass[boxidx]
                    #print("Actively chose best box to be: ",boxidx)
                    #print("Went into funky-loop. Outputarr: ",inClass)
        
        #    inClass = inClass[0]
        #old: only used now for debugging, must be removed
        #if(isinstance(inClass,np.ndarray) and inClass.ndim>1):
            #inClass = inClass[0]
            #print("Multiple boxes detected. Used first box: ",inClass)
        
        xs = inClass[0] #upper left corner
        ys = inClass[1] #upper left corner
        w = inClass[2]-inClass[0]
        h = inClass[-1]-inClass[1]
        return xs,ys,w,h
    
    def get_num_fix_in_bbox(self,bbx: list, BP: np.array):
        tmpBP = np.delete(BP,np.where((BP[:,0]<=bbx[0])),axis=0) #left constraint
        tmpBP = np.delete(tmpBP,np.where((tmpBP[:,1]<=bbx[1])),axis=0) #up constraint
        tmpBP = np.delete(tmpBP,np.where((tmpBP[:,0]>=bbx[0]+bbx[2])),axis=0) #right constraint #only keep between 200 and 400
        tmpBP = np.delete(tmpBP,np.where((tmpBP[:,1]>=bbx[1]+bbx[3])),axis=0) #only keep between 100 and 200
        
        count = tmpBP.shape[0]
        
        return int(count)
    
    def maximize_fixes(self,bbox_arr,BP):
        best = 0 #initialize as zero-fixes best
        idx = 0 
        for i in range(bbox_arr.shape[0]): #go through array rowwise. Send array-row as list get_bounding_box. Check number of fixes on this bbox.
            #print("Testing box i = ",i)
            tmp = self.get_num_fix_in_bbox(self.get_bounding_box(bbox_arr[i].tolist(),fixArr=BP),BP=BP) #convert every bounding box to a list, and get number of points in box
            #print("Result of hits in box ",i," = ",tmp)
            if(tmp>best):
                best = tmp
                idx = i
        #print("Best box: no: ",idx," has number of fixes: ",best)
        return idx
        
    
    def load_images_for_class(self,num):
    #0: aeroplane
    #1: bicycle 
    #2: boat
    #3: cat
    #4: cow 
    #5: diningtable
    #6: dog 
    #7: horse
    #8: motorbike
    #9: sofa
        classdict = {0:"aeroplane",1:"bicycle",2:"boat",3:"cat",4:"cow",5:"diningtable",6:"dog",7:"horse",8:"motorbike",9:"sofa"}    
        self.chosen_class = classdict[num]
        print("Loading class instances for class: ",classdict[num])
        #self.eyeData = np.empty((self.etData[num].shape[0],self.NUM_TRACKERS,1),dtype=object)
        self.filename = []
        self.im_dims = []
        self.bbox = []
        for i,j in enumerate(range(self.num_files_in_class[num])): #i is image, k is patient
            self.filename.append(self.etData[num][i].filename + ".jpg")
            self.im_dims.append(self.etData[num][i].dimensions)
            #print(j)
            self.bbox.append(self.etData[num][i].gtbb)
            
            """for k in range(5):
                #w_max = self.im_dims[i][1] #for removing irrelevant points
                #h_max = self.im_dims[i][0]
                LP = self.etData[num][i].fixations[k].imgCoord.fixL.pos[:]
                RP = self.etData[num][i].fixations[k].imgCoord.fixR.pos[:]
                #remove invalid values . if larger than image-dims or outside of image (negative vals)
                #LP = LP[~np.any((LP[:,]),:]
                BP = np.vstack((LP,RP)) #LP ; RP 
                #eyeData[i,k,0] = [LP,RP] #fill with list values 
                self.eyeData[i,k,0] = BP #fill with matrix"""
        
        print("Loading complete. Loaded ",self.num_files_in_class[num], "images.")                
        #Remember that format for eye-tracking data is (num_instances,num_persons,1). In the one dimension, a matrix is saved, consisting of [LP,RP]-signals.
    
    def random_sample_plot(self): #still needs updating with new format
        from matplotlib.patches import Rectangle
        #random.seed(18)
        classOC = self.classes.index(self.chosen_class)
        num = random.randint(0,len(self.etData[classOC]))
        path = self.p + "/Data/POETdataset/PascalImages/" +self.chosen_class+"_"+self.filename[num]
        im = image.imread(path)
        plt.title("{}:{}".format(self.chosen_class,self.filename[num]))
        #now for eye-tracking-data
        print("Plotting a random sample from loaded images. Chosen index of given class: ",num)
        print("Detected fixes in total: ",self.eyeData_stats[classOC][num][0])
        print("Detected fixes in box: ",self.eyeData_stats[classOC][num][1])
        ax = plt.gca()
        global x 
        global y
        global lol
        lol = self.bbox[num]
        x,y,w,h = self.get_bounding_box(self.bbox[num])
        rect = Rectangle((x,y),w,h,linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        for i in range(self.NUM_TRACKERS):
            color = next(ax._get_lines.prop_cycler)['color']
            mylabel = str(i+1)
            #num_fix = int(self.eyeData[num,i][0][:,0].shape[0]/2)
            num_fix = int(self.eyeData[classOC,num,i][0].shape[0]/2)
            print(num_fix) #number of fixations on img. Deprecated - you have calculated total.
            #Left eye
            plt.scatter(self.eyeData[classOC,num,i][0][:,0],self.eyeData[classOC,num,i][0][:,1],alpha=0.8,label=mylabel,color = color) #wrong - you are not plotting pairwise here, you are plotting R as function of L 
            plt.plot(self.eyeData[classOC,num,i][0][0:num_fix,0],self.eyeData[classOC,num,i][0][0:num_fix,1],label=str(),color= color)
            plt.plot(self.eyeData[classOC,num,i][0][num_fix:,0],self.eyeData[classOC,num,i][0][num_fix:,1],label=str(),color = color)
        plt.legend()
        plt.imshow(im)
        
    def specific_plot(self,classOC,filename,resized=False,multiple=False,last32=False,colorcoded = False,pred_box = None,imdims = None):
        from matplotlib.patches import Rectangle
        classdict = {0:"aeroplane",1:"bicycle",2:"boat",3:"cat",4:"cow",5:"diningtable",6:"dog",7:"horse",8:"motorbike",9:"sofa"}    
        self.chosen_class = classdict[classOC]
        
        try: 
            idx = self.filename.index(filename) #get index of requested file
        except:
            print("Filename not found in loaded data. Did you type correctly?")
            return
            
        if resized==False:
            path = self.p + "/Data/POETdataset/PascalImages/" +self.chosen_class+"_"+filename
        else:
            path = self.p + "/Data/POETdataset/PascalImages/Resized/"+self.chosen_class+"_"+filename
            
        
        #single box: 
        if(multiple==False):
            x,y,w,h = self.get_bounding_box(self.bbox[idx])
        else:
            pass
                    
        im = image.imread(path)
        
        savename = filename.replace(".jpg","")
        
        rect = Rectangle((x,y),w,h,linewidth=4, edgecolor='r', facecolor='none')
        if not (colorcoded):
            fig = plt.figure(3201,figsize=(14,11))
            #plt.title("{}:{}".format(self.chosen_class,filename))
            #now for eye-tracking-data
            ax = plt.gca()
            ax.add_patch(rect)
            for i in range(self.NUM_TRACKERS):
                color = next(ax._get_lines.prop_cycler)['color']
                mylabel = str(i+1)
                num_fix = int(self.eyeData[classOC,idx,i][0].shape[0]/2) #get #no of rows
                print(num_fix) #number of fixations on img
                #Left eye
                """plt.scatter(self.eyeData[idx,i][0][:,0],self.eyeData[idx,i][0][:,1],alpha=0.8,label=mylabel,color = color) #format: [classNo,file in class No, tracker No, listindex (always 0)]
                plt.plot(self.eyeData[idx,i][0][0:num_fix,0],self.eyeData[idx,i][0][0:num_fix,1],label=str(),color= color)
                plt.plot(self.eyeData[idx,i][0][num_fix:,0],self.eyeData[idx,i][0][num_fix:,1],label=str(),color = color)
                """
                if(last32==False):
                    plt.scatter(self.eyeData[classOC,idx,i][0][:,0],self.eyeData[classOC,idx,i][0][:,1],alpha=0.8,label=mylabel,facecolors='none',edgecolors = color,s=450,linewidths=5) #format: [classNo,file in class No, tracker No, listindex (always 0)]
                    plt.plot(self.eyeData[classOC,idx,i][0][0:num_fix,0],self.eyeData[classOC,idx,i][0][0:num_fix,1],label=str(),color= color,linewidth=4)
                    plt.plot(self.eyeData[classOC,idx,i][0][num_fix:,0],self.eyeData[classOC,idx,i][0][num_fix:,1],label=str(),color = color,linewidth=4)
                else: 
                    plt.scatter(self.eyeData[classOC,idx,i][0][-32:,0],self.eyeData[classOC,idx,i][0][-32:,1],alpha=0.8,label=mylabel,facecolors='none',edgecolors = color,s=450,linewidths=5) #format: [classNo,file in class No, tracker No, listindex (always 0)]
                    plt.plot(self.eyeData[classOC,idx,i][0][-32:,0],self.eyeData[classOC,idx,i][0][-32:,1],label=str(),color= color,linewidth=4)
                    plt.plot(self.eyeData[classOC,idx,i][0][-32:,0],self.eyeData[classOC,idx,i][0][-32:,1],label=str(),color = color,linewidth=4)
            
            legendtuple = (rect,)
            legendnametuple = ("Target",)
            legend_ncol = 1
            if pred_box!=None:
                predrect = Rectangle(pred_box[:2],pred_box[2],pred_box[3],linewidth=4, edgecolor='magenta', facecolor='none')
                ax.add_patch(predrect)
                legendtuple += (predrect,)
                legendnametuple += ("Prediction",)
                legend_ncol += 1
            
            #leg_1 = plt.legend(prop={'size': 25,'weight':'bold'},ncol=5,bbox_to_anchor=(0.5,1.15))
            leg_2 = plt.legend(legendtuple,legendnametuple,loc="upper center",prop={'size': 25,'weight':'bold'},ncol=legend_ncol,framealpha=0.0,bbox_to_anchor=(0.5, 1.15))
            plt.imshow(im)
            #ax.add_artist(leg_1)
            #if pred_box!=None:
            ax.add_artist(leg_2)
            ax.axis('off')
            #plt.savefig(savename+"standard_fix_plot.pdf")
            plt.savefig("after_cleanup.pdf")
            #plt.close(fig)
            
                
            
        else:
            
            #make a colormap which is relative to index 
            #print(seqpos)
            mylabel = str(1)
            num_fix_li = []
            
            if(last32==False):
                import math
                plt.figure()
                fig, axes = plt.subplots(1,1,figsize=(14,11))
                color = next(axes._get_lines.prop_cycler)['color']
                for i in range(self.NUM_TRACKERS):#self.NUM_TRACKERS):
                    if self.eyeData[classOC,idx,i][0].shape[0]%2 != 0: #if the complete length modulo 2 is not zero, different signal length for left eye and right eye exist. We discard last signal. This *may* be erroneus, as it might aswell be first one.
                        num_fix = math.floor(self.eyeData[classOC,idx,i][0].shape[0]/2)
                        correction = -1
                    else: 
                        correction = None
                        num_fix = int(self.eyeData[classOC,idx,i][0].shape[0]/2)
                    if self.eyeData[classOC,idx,i][0].shape[0]==0: #invalid fixes
                        pass
                    else:
                        num_fix_li.append(num_fix)
                        norm = plt.Normalize(0,1)
                        seqpos = np.arange(num_fix)
                        if(max(seqpos)!=0):
                            seqpos = seqpos/max(seqpos) #normalize to unit-interval, ie. last fixation always 1 and first always 0 
                            seqpos = np.concatenate((seqpos,seqpos),axis=0)
                            #,alpha=0.8,label=mylabel,cmap=plt.get_cmap('brg')) #format: [classNo,file in class No, tracker No, listindex (always 0)]
                            #following two lines are for lines
                            #axes.plot(self.eyeData[classOC,idx,i][0][0:num_fix,0],self.eyeData[classOC,idx,i][0][0:num_fix,1],alpha=0.6,label=str(),color = color) #lines 
                            #axes.plot(self.eyeData[classOC,idx,i][0][num_fix:correction,0],self.eyeData[classOC,idx,i][0][num_fix:correction,1],alpha=0.6,label=str(),color = color) #lines
                            sc = axes.scatter(x=self.eyeData[classOC,idx,i][0][:correction,0],y=self.eyeData[classOC,idx,i][0][:correction,1],s=400,alpha=0.65,marker="o",c=seqpos,cmap=("cool"),norm=norm)
                        else:
                            pass
                smap = plt.cm.ScalarMappable(cmap='cool', norm=norm)
                cbar = fig.colorbar(smap, ax=axes, fraction=0.046,pad=0.04)#shrink=1)# fraction=0.046,pad=0.04)#, shrink = 0.8)
                cbar.ax.tick_params(labelsize=25)
                for tick in cbar.ax.yaxis.get_major_ticks():
                    tick.label2.set_fontweight('bold')
                cbar.ax.set_ylabel('Sequence position', rotation=90, labelpad = 12, fontdict = {"size":25,'weight':'bold'})
                axes.add_patch(rect)
                legendtuple = (rect,)
                legendnametuple = ("Target",)
                legend_ncol = 1
                if pred_box!=None:
                    predrect = Rectangle(pred_box[:2],pred_box[2],pred_box[3],linewidth=4, edgecolor='magenta', facecolor='none')
                    axes.add_patch(predrect)
                    legendtuple += (predrect,)
                    legendnametuple += ("Prediction",)
                    legend_ncol += 1
                
                plt.legend(legendtuple,legendnametuple,loc="upper center",prop={'size': 25,'weight':'bold'},ncol=legend_ncol,framealpha=0.0,bbox_to_anchor=(0.5, 1.15))
                plt.imshow(im)
                plt.grid(None)
                plt.savefig(savename + "sequence_plot.pdf")
                plt.close()
            else: 
                for i in range(self.NUM_TRACKERS):
                    plt.scatter(self.eyeData[classOC,idx,i][0][-32:,0],self.eyeData[classOC,idx,i][0][-32:,1],alpha=0.8,label=mylabel,color = color) #format: [classNo,file in class No, tracker No, listindex (always 0)]
                    plt.plot(self.eyeData[classOC,idx,i][0][-32:,0],self.eyeData[classOC,idx,i][0][-32:,1],label=str(),color= color)
                    plt.plot(self.eyeData[classOC,idx,i][0][-32:,0],self.eyeData[classOC,idx,i][0][-32:,1],label=str(),color = color)
                plt.imshow(im)
                plt.legend()
                plt.savefig(savename+"no_32_.pdf")
                plt.close()
#        plt.legend()
        #plt.legend(legendtuple,legendnametuple,loc="upper center",ncol=legend_ncol,framealpha=0.0,bbox_to_anchor=(0.5, 0.95))
        #plt.imshow(im)
        
        #plt.savefig("after_cleanup.png")
       
        
        
    def basic_hists(self):
        #goals: 
            #histogram of num of class instances[done]
            #histogram of aspect ratios [done]
            #histogram of number of pixels in images (complete and per class) [done]
            #means and variances histograms of both ratio and number of pixels 
            #sum of number of fixations on image 
        
        classdict = {0:"aeroplane",1:"bicycle",2:"boat",3:"cat",4:"cow",5:"diningtable",6:"dog",7:"horse",8:"motorbike",9:"sofa"}
        self.class_count = {v: k for k, v in classdict.items()}
        self.ratios = []
        self.num_pixels = []
        color_list = [] #for saving same color per class as list format
        
        self.classwise_num_pixels = []
        self.classwise_ratios = []
        #RATIOS
        ax = plt.gca()
        plt.figure(figsize=(1920/160, 1080/160), dpi=160)
        for i,j in enumerate(self.classes):
            print("Round: {}".format(i))
            self.load_images_for_class(i)
            #number of class instances:
            self.class_count[j] = self.eyeData.shape[0]
            tmp_ratios = []
            tmp_num_pixels = []
            #self.classwise_ratios.append([]) #list of lists
            #self.classwise_num_pixels.append([])
            plt.suptitle("Ratios, classwise")
            for k in range(len(self.im_dims)):
                #for stats on complete dataset
                self.ratios.append(self.im_dims[k][1]/self.im_dims[k][0]) #weirdly enough, image-format is stored as height, width 
                self.num_pixels.append(self.im_dims[k][1]*self.im_dims[k][0]) #weirdly enough, image-format is stored as height, width 
                #for stats on class: 
                tmp_ratios.append(self.im_dims[k][1]/self.im_dims[k][0]) #weirdly enough, image-format is stored as height, width 
                tmp_num_pixels.append(self.im_dims[k][1]*self.im_dims[k][0]) #weirdly enough, image-format is stored as height, width 
            self.classwise_ratios.append(tmp_ratios)
            self.classwise_num_pixels.append(tmp_num_pixels)
            plt.subplot(2,5,i+1)
            color = next(ax._get_lines.prop_cycler)['color']
            color_list.append(color)
            plt.hist(tmp_ratios,bins=30,range=(0,4),color = color)
            plt.title("Class: {}".format(j))    
        
        plt.savefig("/home/max/Documents/s194119/Bachelor/Graphs/ratios-classwise.pdf",format="pdf",dpi=160)
        
        
        #overall data-set image-ratios
        plt.figure(200)
        plt.hist(self.ratios,bins=100)
        plt.savefig("/home/max/Documents/s194119/Bachelor/Graphs/ratios-overall.pdf",format="pdf",dpi=160)
    
        #number of class-instances:
        plt.figure(num=199,figsize=(1920/160,1080/160),dpi=160)
        plt.bar(self.class_count.keys(),self.class_count.values(),color=color_list)
        plt.savefig("/home/max/Documents/s194119/Bachelor/Graphs/class-balance.pdf",format="pdf",dpi=160)
        
        #total ds pixelnumbers
        plt.figure(201)
        plt.hist(self.num_pixels,bins=15)
        plt.savefig("/home/max/Documents/s194119/Bachelor/Graphs/overall_num_pixels.pdf",format="pdf",dpi=160)
        
        #classwise pixelnumbers (same loop as earlier for ratios)
        ax = plt.gca()
        plt.figure(num=202,figsize=(1920/160, 1080/160), dpi=160)
        plt.suptitle("Total number of pixels, classwise")
        for i,j in enumerate(self.classes):
            plt.subplot(2,5,i+1)
            color = next(ax._get_lines.prop_cycler)['color']
            plt.hist(self.classwise_num_pixels[i],color=color)
            plt.title("Class: {}".format(j))    
        plt.savefig("/home/max/Documents/s194119/Bachelor/Graphs/classwise_numpx.pdf",format="pdf",dpi=160)
            
        
    def basic_stats(self): #cannot run w/o basic_hist()
        #show means and variances of 1: ratios, 2: pixel-numbers
        ratio_means = []
        ratio_var = []
        for i in range(len(self.classwise_ratios)): 
            ratio_means.append(sum(self.classwise_ratios[i])/len(self.classwise_ratios[i])) #(looks a bit funny because list of lists)
            ratio_var.append(np.var(self.classwise_ratios[i]))
        num_px_means = []
        num_px_var = []
        for i in range(len(self.classwise_num_pixels)):
            num_px_means.append(sum(self.classwise_num_pixels[i])/len(self.classwise_num_pixels[i])) #(looks a bit funny because list of lists)
            num_px_var.append(np.var(self.classwise_num_pixels[i])) #values are huge, but when looking at mean it seems to be quite fine
            
        print("Classwise ratio means: \n",ratio_means)
        print("Classwise ratio vars: \n",ratio_var)
        print("Classwise num_px means: \n",num_px_means)
        print("Classwise num_px vars: \n",num_px_var)
       
    def specific_plot_multiple_boxes(self,classOC,filename):
        from matplotlib.patches import Rectangle
        classdict = {0:"aeroplane",1:"bicycle",2:"boat",3:"cat",4:"cow",5:"diningtable",6:"dog",7:"horse",8:"motorbike",9:"sofa"}    
        self.chosen_class = classdict[classOC]
        
        try: 
            idx = self.filename.index(filename) #get index of requested file
        except:
            print("Filename not found in loaded data. Did you type correctly?")
            return
            
        
        path = self.p + "/Data/POETdataset/PascalImages/" +self.chosen_class+"_"+filename
            
        fig = plt.figure(3289,figsize=(14,11))
        im = image.imread(path)
        #plt.title("{}:{}".format(self.chosen_class,filename))
        #now for eye-tracking-data
        
        ax = plt.gca()
        #single box: 
        print(self.bbox[idx])
        for i in range(self.bbox[idx].shape[0]):
            print(i)
            print(self.bbox[idx][0])
            x,y,w,h = self.get_bounding_box(self.bbox[idx][i])
            rect = Rectangle((x,y),w,h,linewidth=4, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        
        for i in range(self.NUM_TRACKERS):
            color = next(ax._get_lines.prop_cycler)['color']
            mylabel = str(i+1)
            num_fix = int(self.eyeData[classOC,idx,i][0].shape[0]/2) #get #no of rows
            print(num_fix) #number of fixations on img
            #Left eye
            """plt.scatter(self.eyeData[idx,i][0][:,0],self.eyeData[idx,i][0][:,1],alpha=0.8,label=mylabel,color = color) #format: [classNo,file in class No, tracker No, listindex (always 0)]
            plt.plot(self.eyeData[idx,i][0][0:num_fix,0],self.eyeData[idx,i][0][0:num_fix,1],label=str(),color= color)
            plt.plot(self.eyeData[idx,i][0][num_fix:,0],self.eyeData[idx,i][0][num_fix:,1],label=str(),color = color)
            """
            plt.scatter(self.eyeData[classOC,idx,i][0][:,0],self.eyeData[classOC,idx,i][0][:,1],alpha=0.8,label=mylabel,facecolors='none',edgecolors = color,s=450,linewidths=5) #format: [classNo,file in class No, tracker No, listindex (always 0)]
            plt.plot(self.eyeData[classOC,idx,i][0][0:num_fix,0],self.eyeData[classOC,idx,i][0][0:num_fix,1],label=str(),color= color,linewidth=4)
            plt.plot(self.eyeData[classOC,idx,i][0][num_fix:,0],self.eyeData[classOC,idx,i][0][num_fix:,1],label=str(),color = color,linewidth=4)
            
                 
        plt.imshow(im)
        plt.legend(ncol = 5,loc="upper center",prop={'size': 25,'weight':'bold'},framealpha=0.5,bbox_to_anchor=(0.5, 1.15))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("before_cleanup.png")
        

        
if __name__ == "__main__":
    dset = pascalET()
    dset.loadmat()
    #dset.convert_eyetracking_data(CLEANUP=False, STATS=False)
    dset.convert_eyetracking_data(CLEANUP=True, STATS=True)
    dset.load_images_for_class(0)
    dset.specific_plot(0,"2008_003475.jpg",last32=True)
    #dset.specific_plot_multiple_boxes(0,"2008_003475.jpg")
    #del dset
    #del dset """
    #dset = pascalET()
    #dset.loadmat()
    #dset.convert_eyetracking_data(CLEANUP=True, STATS=True) #this script runs without mean. It runs BP - no issue.
    
    
    """
    #FOR SEQUENCE-FIXATION-PLOT
    #get pred bbox: atm manually. Go to plotting_lib.py and use extract_specific_image_data(filename,dataset) and take out bounding box data
    pred_box = torch.tensor([0.2214, 0.2976, 0.5340, 0.6835])
    imdims = torch.tensor([375, 500])
    pred_box[0::2] = pred_box[0::2]*imdims[1].to(dtype=torch.int16)
    pred_box[1::2] = pred_box[1::2]*imdims[0].to(dtype=torch.int16)
    w = int(pred_box[2]-pred_box[0].item())
    h = int(pred_box[-1]-pred_box[1].item())
    x = int(pred_box[0].item())
    y = int(pred_box[1].item())
    #dset.specific_plot(8,"2008_002047.jpg",last32=False,colorcoded = False)#,pred_box=(x,y,w,h))
    #dset.specific_plot(8,"2008_002047.jpg",last32=False,colorcoded = True,pred_box=(x,y,w,h))
    
    #GENERATE ALL PLOTS OF INTEREST
    filenames = ["2011_003134.jpg","2008_004979.jpg","2011_002803.jpg","2010_004714.jpg"]
    pred_boxes = [torch.tensor([0.1572, 0.2672, 0.6670, 0.7373]),torch.tensor([0.2292, 0.2544, 1.0626, 0.9171]),torch.tensor([0.1591, 0.5225, 0.7960, 0.9533]),torch.tensor([0.4740, 0.5294, 0.9060, 0.6551]),torch.tensor([0.2214, 0.2976, 0.5340, 0.6835])]
    imdims = [torch.tensor([333,500]),torch.tensor([375,500]),torch.tensor([375,500]),torch.tensor([374,500]),torch.tensor([375,500])]
    classNo = [2,5,5,2,8]
    for i,_ in enumerate(filenames):
        dset.load_images_for_class(classNo[i])
        pred_box = pred_boxes[i]
        pred_box[0::2] = pred_box[0::2]*imdims[1].to(dtype=torch.int16)
        pred_box[1::2] = pred_box[1::2]*imdims[0].to(dtype=torch.int16)
        w = int(pred_box[2]-pred_box[0].item())
        h = int(pred_box[-1]-pred_box[1].item())
        x = int(pred_box[0].item())
        y = int(pred_box[1].item())
        dset.specific_plot(classNo[i],filenames[i],last32=False,colorcoded = False)#,pred_box=(x,y,w,h))
        dset.specific_plot(classNo[i],filenames[i],last32=False,colorcoded = True,pred_box=(x,y,w,h))
        
        
        
        
        
    
    #images used 
    #1. small boat with big nile ship, class 2, file 2011_003134.jpg
    #pred_box = torch.tensor([0.1572, 0.2672, 0.6670, 0.7373])
    #target_box = torch.tensor([0.3200, 0.4865, 0.4760, 0.5495])
    #imdims = torch.tensor([333, 500])
    
    #2. diningtable with person upper right, class 5, filename 2008_004979.jpg
    # pred_box = torch.tensor([0.2292, 0.2544, 1.0626, 0.9171])
    # target_box = torch.tensor([0.0020, 0.3493, 0.9900, 1.0000])
    # imdims = torch.tensor([375, 500])
    
    #3. diningtable with two persons left right, class 5, 2011_002803.jpg
    #pred_box = torch.tensor([0.1591, 0.5225, 0.7960, 0.9533])
    #imdims = torch.tensor([375, 500])
    
    #4. boat hidden behind bushes, class 2, filename 2010_004714.jpg
    #pred_box =  tensor([[0.4740, 0.5294, 0.9060, 0.6551]], dtype=torch.float64),
    #tensor([[0.1520, 0.0831, 0.7774, 0.7681]]),
    #tensor([[374., 500.]], dtype=torch.float64)]
    
    #5. motorbike with two persons in foreground, class  8
    # [['motorbike_2008_002047.jpg'],
    #  tensor([[0.2214, 0.2976, 0.5340, 0.6835]]),
    #  tensor([[375., 500.]], dtype=torch.float64)]
    
    
    #for normal plot
    #dset.specific_plot(5,"2011_003134.jpg",last32=False,colorcoded=False)
    #dset.specific_plot_multiple_boxes(2,"2011_003134.jpg")#,last32=False)
    
    #dset.specific_plot(3,"2010_002137.jpg",last32=False)
    """