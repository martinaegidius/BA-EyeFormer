#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 13:41:38 2022

@author: max
"""

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
    
    def bbox_stats(self):
        cN = [x for x in range(10)]
        classes = ["aeroplane","bicycle","boat","cat","cow","diningtable","dog","horse","motorbike","sofa"]
        print("----Calculating the average percentage of image covered by bounding box----")
        stat_holder_li = [] #classwise statistics
        #for C in cN: 
        rel_areas = []
        for C in cN:
            holder_dict = {"Name":classes[C],"mean relative box-size": None, 
                           "standard deviation relative box-size": None,
                           "Box mean width": None,
                           "Box mean height": None,
                           "Box width standard deviation": None, 
                           "Box height standard deviation": None,
                           "Mean CX": None,
                           "Mean CY": None,
                           "Box center x standard deviation": None, 
                           "Box center y standard deviation": None,
                           "Image mean width": None,
                           "Image mean height": None,
                           "Image width standard deviation": None,
                           "Image height standard deviation": None}
            rel_areas_tmp = []
            widths = []
            heights = []
            CX = []
            CY = []
            imWidths = []
            imHeights = []
            #get all relevant sizes from D-set
            for i in range(self.num_files_in_class[C]-1):
                bbox = self.chosen_box[C][i] #format [x0,y0,w,h]
                x0 = float(bbox[0])
                y0 = float(bbox[1])
                w = float(bbox[2])
                h = float(bbox[3])
                area_bbox = w*h
                imW = float(self.im_dims[C][i][1])
                imH = float(self.im_dims[C][i][0])
                im_area = float(self.im_dims[C][i][0])*float(self.im_dims[C][i][1])
                rel_area = area_bbox/im_area
                rel_areas_tmp.append(rel_area)
                rel_areas.append(rel_area)
                #print("{w,h}: ",w,h)
                cx = x0 + w/2
                cy = y0 + h/2
                #print("{cx,cy}: ",cx,cy)
                widths.append(w)
                heights.append(h)
                CX.append(cx)
                CY.append(cy)
                imWidths.append(imW)
                imHeights.append(imH)
            holder_dict["mean relative box-size"] = np.mean(rel_areas)
            holder_dict["standard deviation relative box-size"] = np.std(rel_areas)
            holder_dict["Box mean width"] = np.mean(widths)
            holder_dict["Box mean height"] = np.mean(heights)
            holder_dict["Box width standard deviation"] = np.std(widths)
            holder_dict["Box height standard deviation"] = np.std(heights)
            holder_dict["Mean CX"] = np.mean(CX)
            holder_dict["Mean CY"] = np.mean(CY)
            holder_dict["Box center x standard deviation"] = np.std(CX)
            holder_dict["Box center y standard deviation"] = np.std(CY)
            holder_dict["Image mean width"] = np.mean(imWidths)
            holder_dict["Image mean height"] = np.mean(imHeights)
            holder_dict["Image width standard deviation"] = np.std(imWidths)
            holder_dict["Image height standard deviation"] = np.mean(imHeights)
            stat_holder_li.append(holder_dict)
            del holder_dict
        print("....Finished generating bobx statistics....:\n")
        
        #print in a summarizing fashion 
        for i in range(len(stat_holder_li)):
            for entry in stat_holder_li[i]:
                if entry=="Name":
                    print("\n Statistics for ",stat_holder_li[i]["Name"])
                #elif entry=="mean relative box-size" or entry=="standard deviation relative box-size":
                    #print("\n {:<30}:{}".format(entry,stat_holder_li[i][entry]))
                else:
                    val = str(stat_holder_li[i][entry]).replace(".",",")
                    print("{} ".format(val))
                    #print("\n {:<30}:{}".format(entry,stat_holder_li[i][entry]))
        
        for i in range(len(stat_holder_li)):
            for entry in stat_holder_li[i]:
                print("{}".format(entry))
        
        return stat_holder_li,rel_areas
   

def get_dict_of_vals(stat_holder_li):
    holder_d = {"Name": [],
    "mean relative box-size": [],
    "standard deviation relative box-size": [],
    "Box mean width": [],
    "Box mean height": [],
    "Box width standard deviation": [], 
    "Box height standard deviation": [],
    "Mean CX": [],
    "Mean CY": [],
    "Box center x standard deviation": [], 
    "Box center y standard deviation": [],
    "Image mean width": [],
    "Image mean height": [],
    "Image width standard deviation": [],
    "Image height standard deviation": []}
    key_list = list(stat_holder_li[0].keys())
    for i, statDict in enumerate(stat_holder_li):
        for j, entry in enumerate(stat_holder_li[i]):
            for k, statistic in enumerate(key_list):
                if entry==statistic:
                    holder_d[statistic].append(stat_holder_li[i][key_list[k]])
    return holder_d
        
def dict_to_histogram(inD):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 22, 
                         'font.weight': 'bold'})
    names = inD["Name"]
    path = os.path.dirname(__file__)+"/Histograms/"
    if not os.path.exists(path):
        os.mkdir(path)
        
    y_pos = np.arange(len(names))
    for i, entry in enumerate(inD):
        if entry!="Name":
            fig = plt.figure(i,figsize=(20,7),dpi=300)
            plt.bar(y_pos,inD[entry])
            plt.xticks(y_pos,names)
            plt.title(entry,fontsize="22",fontweight="bold")
            fig.tight_layout()
            plt.show()
            fig.savefig(path+entry+".pdf",dpi=fig.dpi)
            plt.close()
            
def list_to_histogram(inL,title="Unkown",nbins = 10):
    import matplotlib.pyplot as plt
    plt.figure(1337)
    plt.hist(inL,bins=nbins,density=True)
    plt.title(title)
    plt.show()
    
        
if __name__ == "__main__":
    dset = pascalET()
    dset.loadmat()
    dset.convert_eyetracking_data(CLEANUP=True, STATS=True) #this script runs without mean. It runs BP - no issue.
    
    stat_holder_li,rel_areas = dset.bbox_stats()
    rel_areas.sort() #inplace operation
    bins = np.array_split(rel_areas,3) #get three equal sized bins
    intervals = {"small":[min(bins[0]),max(bins[0])],"medium":[min(bins[1]),max(bins[1])],"large":[min(bins[-1]),max(bins[-1])]}
    
    #lol = get_dict_of_vals(stat_holder_li)
    #dict_to_histogram(lol)
    
    
