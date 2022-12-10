from load_POET import pascalET
import os 
from torch.utils.data import Dataset, DataLoader
import torch 
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision import transforms, ops
import math



class AirplanesBoatsDataset(Dataset):
    def __init__(self,pascalobj,root_dir,classes,transform=None):
        """old init: 
            def __init__(self,pascalobj,root_dir,classes):#,eyeData,filenames,targets,classlabels,root_dir,classes):
                self.ABDset = generate_DS(pascalobj,classes)
                self.eyeData = eyeData
                self.filenames = filenames
                self.root_dir = root_dir
                self.targets = targets #bbox
                self.classlabels = classlabels
        """

        self.ABDset,self.filenames,self.eyeData,self.targets,self.classlabels,self.imdims,self.length = generate_DS(pascalobj,classes)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self): 
        return len(self.ABDset)
    
    
    def __getitem__(self,idx): #for images
        if torch.is_tensor(idx):
            idx = idx.tolist() #for indexing
        
        SINGLE_PARTICIPANT = True
        
        #PROBABLY THE PROBLEM IS YOU ARE SETTING THE VARS FROM NUMPY "OBJECT"-TYPE
        #filename = self.ABDset[idx,0]
        filename = self.filenames[idx]
        #target = self.ABDset[idx,1].astype(np.int16) #earlier issue: cant convert type np.uint8
        #objClass = self.ABDset[idx,2]
        #signals = self.ABDset[idx,3:]
       
        if(SINGLE_PARTICIPANT==True):
            signals = self.eyeData[idx][0] #participant zero only
        
        else: 
            signals = self.eyeData[idx,:]
        
        targets = self.targets[idx,:]
        
        sample = {"signal": signals,"target": targets,"file": filename, "index": idx,"size":self.imdims[idx],"class":self.classlabels[idx],"mask":None}
        
        if self.transform: 
            sample = self.transform(sample)
        
        return sample
        
                

def rescale_coordsO(eyeCoords,imdims,bbox):
    """
    Args: 
        eyeCoords: Tensor
        imdims: [1,2]-tensor with [height,width] of image
        imW: int value with width of image 
        imH: int value of height of image 
    Returns: 
        Rescaled eyeCoords (inplace) from [0,1].
    """
    outCoords = torch.zeros_like(eyeCoords)
    outCoords[:,0] = eyeCoords[:,0]/imdims[1]
    outCoords[:,1] = eyeCoords[:,1]/imdims[0]
    
    tbox = torch.zeros_like(bbox)
    #transform bounding-box to [0,1]
    tbox[0] = bbox[0]/imdims[1]
    tbox[1] = bbox[1]/imdims[0]
    tbox[2] = bbox[2]/imdims[1]
    tbox[3] = bbox[3]/imdims[0]
    
    
    return outCoords,tbox
   

class tensorPad(object):
    """
    Object-function which pads a batch of different sized tensors to all have [bsz,32,2] with value -999 for padding

    Parameters
    ----------
    object : lambda, only used for composing transform.
    sample : with a dict corresponding to dataset __getitem__

    Returns
    -------
    padded_batch : tensor with dimensions [batch_size,32,2]

    """
    def __call__(self,sample):
        x = sample["signal"]
        xtmp = torch.zeros(32,2)
        numel = x.shape[0]
        xtmp[:numel,:] = x[:,:]
        
        maskTensor = (xtmp[:,0]==0)    
        sample["signal"] = xtmp
        sample["mask"] = maskTensor
        return sample
        
class rescale_coords(object):
    """
    Args: 
        eyeCoords: Tensor
        imdims: [1,2]-tensor with [height,width] of image
        imW: int value with width of image 
        imH: int value of height of image 
    Returns: 
        Rescaled eyeCoords (inplace) from [0,1].
    """
    
    def __call__(self,sample):
        eyeCoords,imdims,bbox = sample["signal"],sample["size"],sample["target"]
        mask = sample["mask"]
        outCoords = torch.ones(32,2)
        
        #old ineccesary #eyeCoords[inMask] = float('nan') #set nan for next calculation -> division gives nan. Actually ineccesary
        #handle masking
        #inMask = (eyeCoords==0.0) #mask-value
        #calculate scaling
        outCoords[:,0] = eyeCoords[:,0]/imdims[1]
        outCoords[:,1] = eyeCoords[:,1]/imdims[0]
        
        #reapply mask
        outCoords[mask] = 0.0 #reapply mask
        
        tbox = torch.zeros_like(bbox)
        #transform bounding-box to [0,1]
        tbox[0] = bbox[0]/imdims[1]
        tbox[1] = bbox[1]/imdims[0]
        tbox[2] = bbox[2]/imdims[1]
        tbox[3] = bbox[3]/imdims[0]
        
        sample["signal"] = outCoords
        sample["target"] = tbox
        
        return sample
        #return {"signal": outCoords,"size":imdims,"target":tbox}
    
    
    


"""FOR IMAGES
    def __getitem__(self,idx): #for images
        if torch.is_tensor(idx):
            idx = idx.tolist() #for indexing
        
        classdict = {0:"aeroplane",1:"bicycle",2:"boat",3:"cat",4:"cow",5:"diningtable",6:"dog",7:"horse",8:"motorbike",9:"sofa"}
        className = classdict[self.ABDset[idx,2]]+"_"
        img_name = self.root_dir+className+ self.ABDset[idx,0]
        print(img_name)
"""
    
    
    
def generate_DS(dataFrame,classes=[x for x in range(10)]):
    """ Note: as implemented now works for two classes only
    Args
        classes: list of input-classes of interest {0:"aeroplane",1:"bicycle",2:"boat",3:"cat",4:"cow",5:"diningtable",6:"dog",7:"horse",8:"motorbike",9:"sofa"}   
        If none is provided, fetch all 9 classes
        
        returns: 
            A tensor which contains: 1. image-filenames, 2. bounding box, 3. class number, 4:8: a list of processed cleaned both-eye-fixations
            A list of filenames
            A tensor which contains all eyetracking data
            A tensor containing bounding boxes
            A tensor containing numerated classlabels
            A tensor containing image-dimensions
    """
    dataFrame.loadmat()
    dataFrame.convert_eyetracking_data(CLEANUP=True,STATS=True)
    
    dslen = 0
    for CN in classes: 
        dslen += len(dataFrame.etData[CN])
    
    df = np.empty((dslen,8),dtype=object)
    
    #1. fix object type by applying zero-padding
    num_points = 32
    eyes = torch.zeros((dslen,dataFrame.NUM_TRACKERS,num_points,2))
    #2.saving files to list
    filenames = []
    #3: saving target in its own array
    targets = np.empty((dslen,4))
    #classlabels and imdims to own array
    classlabels = np.empty((dslen))
    imdims = np.empty((dslen,2))
    
    
    
    datalist = []
    for CN in classes:
        datalist.append(dataFrame.etData[CN])
    
    for i in range(dslen):
        if(i<len(datalist[0])):
            df[i,0] = [dataFrame.etData[classes[0]][i].filename+".jpg"]
            filenames.append(dataFrame.classes[classes[0]]+"_"+dataFrame.etData[classes[0]][i].filename+".jpg")
            df[i,1] = dataFrame.etData[classes[0]][i].gtbb
            df[i,2] = classes[0]
            targets[i] =  dataFrame.etData[classes[0]][i].gtbb
            classlabels[i] = classes[0]
            imdims[i] = dataFrame.etData[classes[0]][i].dimensions
            for j in range(dataFrame.NUM_TRACKERS):
                sliceShape = dataFrame.eyeData[classes[0]][i][j][0].shape
                df[i,3+j] = dataFrame.eyeData[classes[0]][i][j][0] #list items
                if(sliceShape != (2,0)): #for empty
                    eyes[i,j,:sliceShape[0],:] = torch.from_numpy(dataFrame.eyeData[classes[0]][i][j][0][-32:])
                else: 
                    eyes[i,j,:,:] = 0.0
                    print("error-filled measurement [entryNo,participantNo]: ",i,",",j)
                    print(eyes[i,j])
                
                    
                #old: padding too early #eyes[i,j] = zero_pad(dataFrame.eyeData[classes[0]][i][j][0],num_points,-999) #list items
                
                
        else: 
            nextIdx = i-len(datalist[0])
            df[i,0] = [dataFrame.etData[classes[1]][nextIdx].filename+".jpg"]
            filenames.append(dataFrame.classes[classes[1]]+"_"+dataFrame.etData[classes[1]][nextIdx].filename+".jpg")
            df[i,1] = dataFrame.etData[classes[1]][nextIdx].gtbb
            targets[i] =  dataFrame.etData[classes[1]][nextIdx].gtbb
            df[i,2] = classes[1]
            classlabels[i] = classes[1]
            imdims[i] = dataFrame.etData[classes[1]][nextIdx].dimensions
            for j in range(dataFrame.NUM_TRACKERS):
                sliceShape = dataFrame.eyeData[classes[1]][nextIdx][j][0].shape
                df[i,3+j] = dataFrame.eyeData[classes[1]][nextIdx][j][0]
                if(sliceShape != (2,0)): #for empty
                    eyes[i,j,:sliceShape[0],:] = torch.from_numpy(dataFrame.eyeData[classes[1]][nextIdx][j][0][-32:]) #last 32 points
                else:
                    eyes[i,j,:,:] = 0.0
                
                #old: padding too early #eyes[i,j] = zero_pad(dataFrame.eyeData[classes[1]][nextIdx][j][0],num_points,-999) #list items
                
            
        
        
    print("Total length is: ",dslen)
    
    targetsTensor = torch.from_numpy(targets)
    classlabelsTensor = torch.from_numpy(classlabels)
    imdimsTensor = torch.from_numpy(imdims)
    
    return df,filenames,eyes,targetsTensor,classlabelsTensor,imdimsTensor,dslen
   


def zero_pad(inArr: np.array,padto: int,padding: int):
    """
    Args:
        inputs: 
            inArr: array to pad
            padto: integer value of the array-size used. Uses last padto elements of eye-signal
            padding: integer value of padding value, defaults to zero
            
        output: 
            torch.tensor of dtype torch.float32 with size [padto,2]
        
    """
    if(inArr.shape[0]>padto):
        return torch.from_numpy(inArr[:padto]).type(torch.float32)
        
    else: 
        outTensor = torch.ones((padto,2)).type(torch.float32)*padding #[32,2]
        if(inArr.shape[1]==0): #Case: no valid entries in eyeData
            return outTensor
        
        numEntries = inArr.shape[0]
        outTensor[:numEntries,:] = torch.from_numpy(inArr[-numEntries:,:]) #get all
        return outTensor
    
    
torch.manual_seed(1)
CHECK_BALANCE = False
GENERATE_DATASET = True

if(GENERATE_DATASET == True):
    dataFrame = pascalET()
    root_dir = os.path.dirname(__file__) + "/Data/POETdataset/PascalImages/"
    
    """Old implementation, corresponding to commented part in dataset-init:
    #DF,FILES,EYES,TARGETS,CLASSLABELS, IMDIMS = generate_DS(dataFrame,[0,9]) #init DSET, extract interesting stuff as tensors. Aim for making DS obsolete and delete
    #airplanesBoats = AirplanesBoatsDataset(dataFrame, EYES, FILES, TARGETS,CLASSLABELS, root_dir, [0,9]) #init dataset as torch.Dataset.
    """
    
    classesOC = [0,9]
    composed = transforms.Compose([transforms.Lambda(tensorPad()),
                                   transforms.Lambda(rescale_coords())]) #transforms.Lambda(tensor_pad() also a possibility, but is probably unecc.
    
    airplanesBoats = AirplanesBoatsDataset(dataFrame, root_dir, classesOC,transform=composed) #init dataset as torch.Dataset.
    
    
    
    #"""
    #check that transforms work 
    
    #define split ratio
    test_size = int(0.2*airplanesBoats.length)
    train_size = airplanesBoats.length-test_size
    
    #set seed 
    
    #get split 
    train,test = torch.utils.data.random_split(airplanesBoats,[train_size,test_size])
    ###load split from data
    
    split_root_dir = os.path.dirname(__file__)

def get_split(root_dir,classesOC):
    """
    Function to get professors splits
    
        Args: 
        root_dir, ie. path to main-project-folder
        classesOC: List of classes of interest
        
       Returns:
        list of filenames for training
        list of filenames for testing
    """
        
    split_dir = root_dir +"/Data/POETdataset/split/"
    classlist = ["aeroplane","bicycle","boat","cat","cow","diningtable","dog","horse","motorbike","sofa"]

    training_data = []
    for classes in classesOC: #loop saves in list of lists format
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
    
    #unpack lists of lists to single list for train and for test
    train = []
    test = []
    for i in range(len(training_data)):
        train = [*train,*training_data[i]]
        test = [*test,*test_data[i]]        
         
    return train,test

root_dir = os.path.dirname(__file__) + "/Data/POETdataset/"
if(GENERATE_DATASET == True):
    trainLi,testLi = get_split(split_root_dir,classesOC) #get Dimitrios' list of train/test-split.
    trainIDX = [airplanesBoats.filenames.index(i) for i in trainLi] #get indices of corresponding entries in data-structure
    testIDX = [airplanesBoats.filenames.index(i) for i in testLi]
    #subsample based on IDX
    train = torch.utils.data.Subset(airplanesBoats,trainIDX)
    test = torch.utils.data.Subset(airplanesBoats,testIDX)
    root_dir = os.path.dirname(__file__) + "/Data/POETdataset/"
    #torch.save(train,root_dir+"airplanesBoatsTrain.pt")
    #torch.save(test,root_dir+"airplanesBoatsTest.pt")
    #print("................. Wrote datasets to disk ....................")

BATCH_SZ = 1 
trainloader = DataLoader(train,batch_size=BATCH_SZ,shuffle=True,num_workers=0)
testloader = DataLoader(test,batch_size=BATCH_SZ,shuffle=True,num_workers=0)


boxes = torch.zeros(80,4)
for i, data in enumerate(trainloader): 
    boxes[i,:] = data["target"]
    
print("Average box on complete training set",torch.mean(boxes,0))

#first five samples 
boxes = torch.zeros(5,4)
for i, data in enumerate(testloader): 
    if i==5: 
        break
    print(data["file"])
    boxes[i,:] = data["target"]

print("Average box first five sampels torch.manual.seed(1): ",torch.mean(boxes,0))

