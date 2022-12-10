#import timm
import torch
import math
import os 
import numpy as np 
import matplotlib.pyplot as plt
import torchvision.transforms 
import torchvision.io
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import PIL.Image as Image
import timm
import matplotlib.pyplot as plt
import matplotlib as mpl 
import torch.nn as nn
import ViT_model as vm
from tqdm import tqdm
import sys
#import handle_pascal as hdl
from load_POET_timm import pascalET
import box_operations as boxops
from copy import deepcopy

###CALL WITH SYSTEM ARGUMENTS
#arg 1: classChoice
#arg 2: NUM_IN_OVERFIT
#arg 3: NUM_LAYERS
#arg 4: NUM_HEADS
#arg 5: adam LR 

 
#try:
#    classChoice = int(sys.argv[1])
#except:
#    classChoice = None
#    pass
classChoice = [x for x in range(10)]
#classChoice = [0]

SAVEDSET = False #flag for defining if data-sets are saved to scratch after dataloader generation. For ViT false because big files. 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("FOUND DEVICE: ",device)

def generate_DS(dataFrame,classes=[x for x in range(10)]):
    """
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
    dataFrame.convert_eyetracking_data(CLEANUP=True,STATS=True,mode="mean") #MEAN FIXATION EXPERIMENT 
    NUMCLASSES = len(classes) 
    
    
    dslen = 0
    dslen_li = []
    for CN in classes: 
        #print(CN)
        dslen += len(dataFrame.etData[CN])
        dslen_li.append(len(dataFrame.etData[CN]))
        #print("Total length is ",dslen)
        
    #print(dslen_li)
    
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
        #print("DSET {} LENGTH is: ".format(CN),len(dataFrame.etData[CN]))
    
    idx = 0
    IDXSHIFTER = 0
    k = 0
    for i in range(dslen): 
        if((i==sum(dslen_li[:k+1])) and (i!=0)): #evaluates to true when class-set k is depleted
            #print("depleted dataset {} at idx {}".format(k,i))
            IDXSHIFTER += dslen_li[k]
            k += 1
            
        idx = i - IDXSHIFTER
        df[i,0] = [dataFrame.etData[classes[k]][idx].filename+".jpg"]
        filenames.append(dataFrame.classes[classes[k]]+"_"+dataFrame.etData[classes[k]][idx].filename+".jpg")
        df[i,1] = dataFrame.etData[classes[k]][idx].gtbb
        df[i,2] = classes[k]
        targets[i] =  dataFrame.etData[classes[k]][idx].gtbb
        classlabels[i] = classes[k]
        imdims[i] = dataFrame.etData[classes[k]][idx].dimensions[:2] #some classes also have channel-dim
        for j in range(dataFrame.NUM_TRACKERS):
            sliceShape = dataFrame.eyeData[classes[k]][idx][j][0].shape
            df[i,3+j] = dataFrame.eyeData[classes[k]][idx][j][0] #list items
            if(sliceShape != (2,0) and sliceShape != (0,) and sliceShape != (2,)): #for all non-empty
                eyes[i,j,:sliceShape[0],:] = torch.from_numpy(dataFrame.eyeData[classes[k]][idx][j][0][-32:].astype(np.int32)) #some entries are in uint16, which torch does not support
            else: 
                eyes[i,j,:,:] = 0.0
                #print("error-filled measurement [entryNo,participantNo]: ",idx,",",j)
                #print(eyes[i,j])
            
            
            
    print("Total length is: ",dslen)
    
    targetsTensor = torch.from_numpy(targets)
    classlabelsTensor = torch.from_numpy(classlabels)
    imdimsTensor = torch.from_numpy(imdims)
    
    return df,filenames,eyes,targetsTensor,classlabelsTensor,imdimsTensor,dslen

class PASCALdataset(Dataset):
    def __init__(self,pascalobj,root_dir,classes,sTransform=None,imTransform = None,coordTransform = None):
        self.ABDset,self.filenames,self.eyeData,self.targets,self.classlabels,self.imdims,self.length = generate_DS(pascalobj,classes)
        self.root_dir = root_dir
        self.sTransform = sTransform
        self.imTransform = imTransform
        self.coordTransform = coordTransform
        self.deep_representation = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ViT_model = timm.create_model('vit_base_patch16_224',pretrained=True,num_classes=0, global_pool='').to(self.device)
        
    def __len__(self): 
        return len(self.ABDset)
    
    
    def __getitem__(self,idx): #for images
        if torch.is_tensor(idx):
            idx = idx.tolist() #for indexing
        
        SINGLE_PARTICIPANT = True
        
        
        filename = self.filenames[idx]
        image = Image.open(self.root_dir + filename)
        
       
        if(SINGLE_PARTICIPANT==True):
            signals = self.eyeData[idx][0]#participant zero only
        
        else: 
            signals = self.eyeData[idx,:]
        
        classL = self.classlabels[idx]
        
        targets = self.targets[idx,:]
        size = self.imdims[idx]
        
        sample = {"CLS": None, "signal": signals,"target": targets,"file": filename, "index": idx,"size":self.imdims[idx],"class":self.classlabels[idx],"mask":None,"image":image,"orgimage":image,"scaled_signal":None,"scaled_target":None,"target_centered":None}
        
        if self.sTransform: 
            sample = self.sTransform(sample)
        if self.imTransform:
            sample["orgimage"] = self.imTransform(sample["image"]) #only shoot through image 
        
        if self.coordTransform:
            sample = self.coordTransform(sample)

        if sample["orgimage"].dim()==3:
            sample["image"] = sample["orgimage"].unsqueeze(0)
            
        with torch.no_grad():    
            dImage = self.ViT_model.forward_features(sample["image"].to(self.device))
            sample["CLS"] = dImage[:,0,:] 
            #print(dImage.shape)
            del dImage

            #should interpolate here instead of later, might aswell save lines of code later on
            
        return sample
    
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

    n_train_samples = []
    training_data = []
    for classes in classesOC: #loop saves in list of lists format
        train_filename = split_dir + classlist[classes]+"_Rbbfix.txt"
        with open(train_filename) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        for i in range(len(lines)): 
            lines[i] = classlist[classes]+"_"+lines[i]+".jpg"
            
        training_data.append(lines)
        n_train_samples.append(len(lines))
        
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
         
    return train,test,n_train_samples

def get_balanced_permutation(nsamples,NUM_IN_OVERFIT,valsize=100):
    sample_perc = [x/sum(nsamples) for x in nsamples] #get percentage-distribution 
    if(len(nsamples)>1):
        classwise_nsamples = [math.ceil(x*NUM_IN_OVERFIT) for x in sample_perc] #get number of samples per class
        NUM_IN_OVERFIT = sum(classwise_nsamples) #overwrite NUM_IN_OVERFIT, as you use math.ceil
    else: 
        classwise_nsamples = [NUM_IN_OVERFIT]
        
    h = [torch.Generator() for i in range(len(classesOC))]
    for i in range(len(classesOC)):
        h[i].manual_seed(i)
    ofIDX = torch.zeros(0).to(torch.int32)
    valIDX = torch.zeros(0).to(torch.int32)
    offset = 0
    for num,instance in enumerate(nsamples):
        idx = torch.randperm(int(nsamples[num]),generator=h[num])
        t_idx = idx + offset #add per-class offset for classwise indexing
        idx = t_idx[:classwise_nsamples[num]]
        vidx = t_idx[classwise_nsamples[num]:classwise_nsamples[num]+int(valsize/len(nsamples))]
        ofIDX = torch.cat((ofIDX,idx),0)
        valIDX = torch.cat((valIDX,vidx),0)
        offset += instance
        
    return ofIDX,valIDX,NUM_IN_OVERFIT


def load_split(className,root_dir):
    #loads earlier saved splits from disk. 
    #Arg: className (e.g. "airplane")
    
    tmp_root = root_dir+"/datasets/"
    print("................. Loaded ",className," datasets from disk .................")
    train = torch.load(tmp_root+className+"Train.pt") #load earlier saved split
    test = torch.load(tmp_root+className+"Test.pt")
    return train,test


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = ["aeroplane","bicycle","boat","cat","cow","diningtable","dog","horse","motorbike","sofa"]
if(classChoice!=None):
    classesOC = classChoice
else:
    print("No selection of data provided. Used class 3, 2, 1.")
    classesOC = [3,2,1]
    #classChoice = 3

torch.manual_seed(9)
###SCRIPT PARAMETERS###
GENERATE_DATASET = True
timm_root = os.path.dirname(__file__)
if((len(classesOC)>1) and (len(classesOC)<10)):
    classString = "multiple_classes"
elif(len(classesOC)==10):
    classString = "all_classes"
else: 
    classString = classes[classesOC[0]]

BATCH_SZ = 8
# #RUN_FROM_COMMANDLINE. Class at top of programme.
NUM_IN_OVERFIT = int(sys.argv[2]) #NUM-IN-OVERFIT EQUALS LEN(TRAIN) IF OVERFIT == False
NLAYERS = int(sys.argv[3])
NHEADS = int(sys.argv[4])
EPOCHS = int(sys.argv[5])
EVAL = int(sys.argv[6])
LR = float(sys.argv[7])
#LR = 0.001

#NLAYERS = 3
#NHEADS = 1
#EPOCHS = 600
#NUM_IN_OVERFIT = 2
DROPOUT = 0.0
LR_FACTOR = 1
EPOCH_STEPS = NUM_IN_OVERFIT//BATCH_SZ
if EPOCH_STEPS == 0: 
    EPOCH_STEPS = 1
NUM_WARMUP = int(EPOCHS*(1/3))*EPOCH_STEPS #constant 30% warmup-rate
if NUM_WARMUP == 0: #debugging cases
    NUM_WARUP = 1

BETA = 1
OVERFIT=True
#EVAL = 0 #flag which is set for model evaluation, ie. final model. Set 1 if it is final model.

   
        


    
if(GENERATE_DATASET == True or GENERATE_DATASET==None):
    dataFrame = pascalET()
    #root_dir = os.path.dirname(__file__) + "/../eyeFormer/Data/POETdataset/PascalImages/"
    root_dir = os.path.join(os.path.expanduser('~'),"BA/eyeFormer/Data/POETdataset/PascalImages/")
    split_root_dir = os.path.join(os.path.expanduser('~'),"BA/eyeFormer")
    
    SignalTrans = torchvision.transforms.Compose(
        [torchvision.transforms.Lambda(boxops.tensorPad())])
        #,torchvision.transforms.Lambda(rescale_coords())]) #transforms.Lambda(tensor_pad() also a possibility, but is probably unecc.
    
    ImTrans = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224),torchvision.transforms.InterpolationMode.BILINEAR),
                                              torchvision.transforms.ToTensor()]) #should be normalized and converted to 0,1. ToTensor always does this for PIL images
    
    CoordTrans = torchvision.transforms.Compose([torchvision.transforms.Lambda(boxops.rescale_coords()),torchvision.transforms.Lambda(boxops.get_center())])
    

    fullDataset = PASCALdataset(dataFrame, root_dir, classesOC,sTransform=SignalTrans,imTransform = ImTrans,coordTransform = CoordTrans) #init dataset as torch.Dataset.
    
    trainLi,testLi,nsamples = get_split(split_root_dir,classesOC) #get Dimitrios' list of train/test-split.
    trainIDX = [fullDataset.filenames.index(i) for i in trainLi] #get indices of corresponding entries in data-structure
    testIDX = [fullDataset.filenames.index(i) for i in testLi]
    #subsample based on IDX
    train = torch.utils.data.Subset(fullDataset,trainIDX)
    test = torch.utils.data.Subset(fullDataset,testIDX)
    #save generated splits to scratch
    if(SAVEDSET == True):
        torch.save(train,timm_root+"/datasets/"+classString+"Train.pt")
        torch.save(test,timm_root+"/datasets/"+classString+"Test.pt")
        print("................. Wrote datasets for ",classString,"to disk ....................")
        

if(GENERATE_DATASET == False):
    train,test = load_split(classString,timm_root)  
    print("Succesfully read data from binary")
 

#make dataloaders of chosen split
trainloader = DataLoader(train,batch_size=BATCH_SZ,shuffle=True,num_workers=0)
#FOR DEBUGGING
#for i, data in enumerate(trainloader):
#    if i==1:
#        DEBUGSAMPLE = data

testloader = DataLoader(test,batch_size=BATCH_SZ,shuffle=True,num_workers=0)

if NUM_IN_OVERFIT==None and OVERFIT==True: #only if no cmd-argv provided
    NUM_IN_OVERFIT=375
    print("...No argument for length of DSET provided. Used N=16...")
    
g = torch.Generator()
g.manual_seed(8)

if(OVERFIT): #CREATES TRAIN AND VALIDATION-SPLIT 
     #new mode for getting representative subsample: 
    ofIDX,valIDX,NUM_IN_OVERFIT = get_balanced_permutation(nsamples,NUM_IN_OVERFIT,valsize=60)
    #print(valIDX)
    #print("valIDX shape: ",valIDX.shape)
    
    #IDX = torch.randperm(len(train),generator=g[0])#[:NUM_IN_OVERFIT].unsqueeze(1) #random permutation, followed by sampling and unsqueezing
    #ofIDX = IDX[:NUM_IN_OVERFIT].unsqueeze(1)
    #valIDX = IDX[NUM_IN_OVERFIT:NUM_IN_OVERFIT+17].unsqueeze(1) #17 because the smallest train-set has length 33=max(L)+17.
    overfitSet = torch.utils.data.Subset(train,ofIDX)
    valSet = torch.utils.data.Subset(train,valIDX)
    trainloader = DataLoader(overfitSet,batch_size=BATCH_SZ,shuffle=True,num_workers=0,generator=g)
    print("Overwrote trainloader with overfit-set of length {}".format(ofIDX.shape[0]))
    valloader = DataLoader(valSet,batch_size=BATCH_SZ,shuffle=True,num_workers=0,generator=g)
    print("Valloader of constant length {}".format(valIDX.shape[0]))


model = nn.Sequential(nn.Linear(768,2048,bias=True),nn.ReLU(),nn.Linear(2048,4,bias=True)).to(device)
    #vm.VIT_REG_MODEL(input_dim=768,output_dim=4).to(device)
#model(deep_seq,mask)
#model_opt = vm.NoamOpt(model.d_model,LR_FACTOR,NUM_WARMUP,torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
model_opt = torch.optim.Adam(model.parameters(),lr=LR,betas=(0.9,0.999),eps=1e-09)
loss_fn = nn.SmoothL1Loss(beta=BETA) #default: mean and beta=1.0



def train_one_epoch(model,loss,trainloader,negative_print=False) -> float: 
    
    running_loss = 0.
    #src_mask = generate_square_subsequent_mask(32).to(device)
    correct_count = 0
    false_count = 0
    counter = 0
    
    tIOUli_holder = []
    
    for i, data in enumerate(trainloader):
        counter += 1
        model_opt.zero_grad() #reset grads
        #CLS = data["image"][:,:,0,:].to(device)
        CLS = data["CLS"].to(device)
        outputs = model(CLS)
        
        #print(outputs.shape)
        outputs = boxops.center_to_box(outputs).to(device)
        #PASCAL CRITERIUM
        #get boxes in correct format (corners instead of cx,cy,w,h)
        target = data["scaled_target"].to(device)
        #target = boxops.center_to_box(target)

        
        noTrue,noFalse,IOU_li = vm.pascalACC(outputs,target)
        tIOUli_holder = tIOUli_holder + IOU_li
        correct_count += noTrue
        false_count += noFalse
        
        loss = loss_fn(outputs,target) #L1 LOSS
        #loss = ops.generalized_box_iou_loss(output,target)
        loss.backward()
        running_loss += loss.item() #is complete EPOCHLOSS
        #n.utils.clip_grad_value_(model.parameters(), clip_value=0.5) #experimentary
        model_opt.step()
        
    IOU_mt = sum(tIOUli_holder)/len(tIOUli_holder)    
    epochLoss = running_loss/len(trainloader.dataset) 
    epochAcc = correct_count/len(trainloader.dataset) #TP / (complete trainingset length)    
    
    return epochLoss,correct_count,false_count,target,epochAcc,model,IOU_mt


def train_one_epoch_w_val(model,loss,trainloader,valloader,negative_print=False,DEBUG=False) -> float: 
    running_loss = 0.
    #src_mask = generate_square_subsequent_mask(32).to(device)
    correct_count = 0
    false_count = 0
    counter = 0
    
    if(DEBUG):
        train__L = []
        val__L = []
        for i, data in enumerate(trainloader):
            train__L.append(data["file"][0])
        for i, data in enumerate(valloader):
            val__L.append(data["file"][0])
        
        train_s = set(train__L) 
        val_s = set(val__L)
        s = train_s & val_s 
        if len(s)!=0:
            print("!!!!WARNING: VALIDATION SET DOES CONTAIN TRAINING DATA!!!")
    
    #VALIDATION FIRST
    vIOUli_holder = []
    false_val_count = 0
    correct_val_count = 0
    val_loss = 0
    counter = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(valloader):
            counter += 1 
            target = data["scaled_target"].to(device)
            CLS = data["CLS"].to(device)
            outputs = model(CLS)
            outputs = boxops.center_to_box(outputs).to(device)
            noTrue,noFalse,IOUli_v = vm.pascalACC(outputs,target)
            vIOUli_holder = vIOUli_holder+IOUli_v 
            correct_val_count += noTrue
            false_val_count += noFalse
            
            
            loss = loss_fn(outputs,target) #L1 LOSS. Mode: "mean"
            val_loss += loss.item() #is complete EPOCHLOSS
    
    loss = 0
    
    model.train()
    tIOUli_holder = []
    for i, data in enumerate(trainloader):
        counter += 1
        model_opt.zero_grad() #reset grads
        #CLS = data["image"][:,:,0,:].to(device)
        CLS = data["CLS"].to(device)
        target = data["scaled_target"].to(device)
        outputs = model(CLS)
        #print(outputs.shape)
        outputs = boxops.center_to_box(outputs).to(device)
        #print(outputs.shape)
        #print(target.shape)
        #print("Train target: \n",target)
        #print("TRAIN output: \n",outputs)
        #PASCAL CRITERIUM
        if(DEBUG==True):
            print("OUTPUT BECOMES: ",outputs)
            print("TARGET BECOMES: ",target)
            
        noTrue,noFalse,IOUli_t = vm.pascalACC(outputs,target)
        #print("\nScaled pascalACC returns:\n",pascalACC(sOutputs,sTargets))
        #print("\nOriginal pascalACC returns:\n",pascalACC(outputs,target))
        #print("\ngetIOU function returns:\n",getIOU(outputs,target,sensitivity=0.5))
        tIOUli_holder = tIOUli_holder+IOUli_t 
        
        correct_count += noTrue
        false_count += noFalse
        
        
        loss = loss_fn(outputs,target) #SMOOTH L1. Mode: mean
        if(DEBUG==True):
            print("LOSS BECOMES: ",loss)
        #loss = ops.generalized_box_iou(outputs.to(dtype=torch.float64),target.to(dtype=torch.float64))
        running_loss += loss.item() #is complete EPOCHLOSS
        loss.backward()
        #nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5) #experimentary
        model_opt.step()
    
    if(DEBUG==True):
        print("t holder becomes: \n",tIOUli_holder)
        print("\nsum becomes",sum(tIOUli_holder))
        print("\nLen becomes",len(tIOUli_holder))
        print("\n Train MEAN BECOMES: ",sum(tIOUli_holder)/len(tIOUli_holder))
    IOU_mt = sum(tIOUli_holder)/len(tIOUli_holder)        
    
    
        
       
    if(DEBUG==True):
        print("v holder becomes: \n",vIOUli_holder)
        print("\nsum becomes",sum(vIOUli_holder))
        print("\nLen becomes{}".format(len(vIOUli_holder)))
        print("\n VAL MEAN BECOMES: ",sum(vIOUli_holder)/len(vIOUli_holder))
    IOU_mv = sum(vIOUli_holder)/len(vIOUli_holder)
    
    epochLoss = running_loss/len(trainloader.dataset) 
    epochAcc = correct_count/len(trainloader.dataset) #TP / (complete trainingset length)
    #print("correct count is: ",correct_count)
    #print("trainloader dset len is: ",len(trainloader.dataset))
    epochValLoss = val_loss/len(valloader.dataset) 
    #print("complete over-epoch val loss: ",epochValLoss)   
    epochValAcc = correct_val_count/len(valloader.dataset)
    return epochLoss,correct_count,false_count,target,epochAcc,model,epochValLoss,epochValAcc, IOU_mt, IOU_mv

def save_model(model,classString,root_dir,params,mode,epochNo):
    path = root_dir + "/" + classString+"/"
    if not os.path.exists(path):
        os.mkdir(path)
        print("Created dir in: ",path)
    path += "L_"+str(params[-1]) +"_nL_" + str(params[0]) +"_nH_" + str(params[1]) +"/" +mode+"/"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print("Created subdir in: ",path)
    torch.save(model,path+"model.pth")
    print("Wrote model from epoch {} to scratch in : {}".format(epoch,path))
    torch.save(epoch,path+"epoch_no.pth")
    return None

def load_best_model(classString,root_dir,params,mode):
    path = root_dir +"/" + classString +"/"
    path += "L_"+str(params[-1]) +"_nL_" + str(params[0]) +"_nH_" + str(params[1]) +"/" +mode+"/"
    epoch_no = torch.load(path+"epoch_no.pth")
    model = torch.load(path+"model.pth")
    print("Loaded model from epoch {}".format(epoch_no))
    return model
    

print("-----------------------------------------------------------------------------")
NPARAMS = vm.get_n_params(model)
if(OVERFIT):
    print("Model parameters:\n tL: {}\n vL: {}\nlrf: {}\nNEPOCHS: {}\n num_warmup: {}\n Dropout: {}\n Beta: {}\n NHEADS: {}\n NLAYERS: {} \n BATCH-SIZE: {} \n PARAMS: {}\n LR: {}".format(NUM_IN_OVERFIT,len(valIDX),LR_FACTOR,EPOCHS,NUM_WARMUP,DROPOUT,BETA,NHEADS,NLAYERS,BATCH_SZ,NPARAMS,LR))
    print("ACTUAL TRAINLOADER LENS",len(trainloader),len(trainloader.dataset))
else:
    print("Model parameters:\n tL: {} \n \nlrf: {}\nNEPOCHS: {}\n num_warmup: {}\n Dropout: {}\n Beta: {}\n NHEADS: {}\n NLAYERS: {}\n BATCH-SIZE: {}, PARAMS: {}\n LR: {}".format(len(trainloader.dataset),LR_FACTOR,EPOCHS,NUM_WARMUP,DROPOUT,BETA,NHEADS,NLAYERS,BATCH_SZ,NPARAMS,LR))
    
#def train_number_of_epochs(EPOCHS,model,loss,trainloader,oTrainLoader,overfit=False,negative_print=False):
epoch_number = 0
epochLoss = 0 
epochLossLI = []
epochAccLI = []
epochValLossLI = []
epochValAccLI = []
epochIOU = []
trainIOU = []
valIOU = []
torch.autograd.set_detect_anomaly(True)
patienceCounter = 0
for epoch in (pbar:=tqdm(range(EPOCHS))):
    try:        
        if(OVERFIT):
            if(epoch==0):
                BEST_MODEL = False
                epochLoss, correct_count, false_count,target,epochAcc,model,valLoss,valAcc,IOU_t,IOU_v = train_one_epoch_w_val(model,loss_fn,trainloader,valloader,negative_print=False,DEBUG=False)
            else: 
                epochLoss, correct_count, false_count,target,epochAcc,model,valLoss,valAcc,IOU_t,IOU_v = train_one_epoch_w_val(model,loss_fn,trainloader,valloader,negative_print=False)
                if(valLoss < min(epochValLossLI)):
                    print("Lower validation loss encountered in EPOCH {}, with current loss {} and earlier minimal loss {}".format(epoch,valLoss,min(epochValLossLI)))
                    best_model = deepcopy(model)
                    BEST_MODEL = True
                    best_epoch = deepcopy(epoch)
                    patienceCounter = 0
                else:
                    patienceCounter += 1
            tmpStr = f" | avg train loss {epochLoss:.4f} | train acc: {epochAcc:.4f} | avg val loss: {valLoss:.4f} | avg val acc: {valAcc:.4f} | Mean Train IOU: {IOU_t:.4f} | Mean Val IOU: {IOU_v:.4f} |"
            epochValAccLI.append(valAcc)
            epochValLossLI.append(valLoss)
            valIOU.append(IOU_v)
            if(patienceCounter>20):
                print("10-epoch patience has been passed. Stopping training")
                break
        else:
            epochLoss, correct_count, false_count,target,signal,mask,epochAcc,model,IOU_t = train_one_epoch(model,loss_fn,trainloader,negative_print=False)
            tmpStr = f" | avg train loss {epochLoss:.2f} | train acc: {epochAcc:.2f} | Epoch train IOU {IOU_t:.2f} |"
            
        pbar.set_postfix_str(tmpStr)
        epochLossLI.append(epochLoss)
        epochAccLI.append(epochAcc)
        trainIOU.append(IOU_t)
        epoch_number += 1         
        
    except KeyboardInterrupt:
        print("Manual early stopping triggered")
        break
    
if(BEST_MODEL):
    save_model(best_model,classString,timm_root,params=[NLAYERS,NHEADS],mode="val",epochNo=best_epoch)
    del model
    print("Cleared current trained sub-optimal model")


def save_split(trainloader,valloader,classString,root_dir,params):
    """
    Simple function for saving filenames of images used in train- and val-split respectively for ensuring reproducibility
    Args: 
        trainloader (pyTorch dataloader)
        valloader (pyTorch dataloader)
        classString: string, name descriptor of class of investigation
        root_dir: usually os.path.dirname(__file__)
        
    Returns: 
        None
        
    """
    path = root_dir + "/" + classString + "/L_" + str(params[-1])+"_nL_" + str(params[0]) +"_nH_" + str(params[1])+"/"
    if not os.path.exists(path):
        os.mkdir(path)
        print("Created dir in: ",path)
    trainlist = []
    vallist = []
    for i,data in enumerate(trainloader):
        trainlist.append(data["file"][0])
    for i, data in enumerate(valloader):
        vallist.append(data["file"][0])
    s = path+classString+"_train_split_len_"+str(len(trainloader.dataset))+".pth"
    torch.save(trainlist,s)
    s = path+classString+"_val_split_len_"+str(len(valloader.dataset))+".pth"
    torch.save(vallist,s)
    

def save_epochs(loss,acc,classString,root_dir,mode,params):
    path = root_dir + "/"+classString + "/"
    if not os.path.exists(path):
        os.mkdir(path)
        print("Created dir in: ",path)
    path += "L_"+str(params[-1]) +"_nL_" + str(params[0]) +"_nH_" + str(params[1]) +"/"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print("Created subdir in: ",path)
    
    if(mode=="eval"):
        path += "eval/"
        if not os.path.exists(path):
            os.mkdir(path)
            print("Created subdir in: ",path)
        torch.save(loss,path+classString+"_"+mode+"_losses.pth")
        print("Saved loss results to: ",path+classString+"_"+mode+"_losses.pth")
        torch.save(acc,path+"/"+classString+"_"+mode+"_acc.pth")
        print("Saved accuracy results to: ",path+classString+"_"+mode+"_acc.pth")
    else:
        torch.save(loss,path+classString+"_"+mode+"_losses.pth")
        print("Saved loss results to: ",path+classString+"_"+mode+"_losses.pth")
        torch.save(acc,path+"/"+classString+"_"+mode+"_acc.pth")
        print("Saved accuracy results to: ",path+classString+"_"+mode+"_acc.pth")
        
    return

def save_IOU(IOU_li,classString,root_dir,params,mode):
    path = root_dir + "/"+classString+"/"
    if not os.path.exists(path):
        print("PATH DID NOT EXIST",path)
        os.mkdir(path)
        print("Created dir in: ",path)
    path +=  "L_"+str(params[-1]) +"_nL_" + str(params[0]) +"_nH_" + str(params[1]) +"/" #
    if not os.path.exists(path):
        os.makedirs(path,exist_ok=True)
        print("Created subdir in: ",path)
    path += mode+"/"
    if not os.path.exists(path):
        os.mkdir(path)
        print("Created subdir in: ",path)
    torch.save(IOU_li,path+"epochIOU.pth")
    print("Wrote epoch IOU's to scratch in :",path)
    return None


if(EVAL==0):
    save_epochs(epochLossLI,epochAccLI,classString,timm_root,mode="VIT_REG",params=[NLAYERS,NHEADS,len(trainloader.dataset)])
    save_IOU(trainIOU,classString,timm_root,params=[NLAYERS,NHEADS,len(trainloader.dataset)],mode="result")
    #save_model(model,classString,timm_root,params=[NLAYERS,NHEADS],mode="result")

if(EVAL==1):
    save_epochs(epochLossLI,epochAccLI,classString,timm_root,mode="VIT_REG",params=[NLAYERS,NHEADS,len(trainloader.dataset)])
    save_IOU(trainIOU,classString,timm_root,params=[NLAYERS,NHEADS,len(trainloader.dataset)],mode="eval")
    #save_model(model,classString,timm_root,params=[NLAYERS,NHEADS],mode="eval")
    

if(OVERFIT==True and EVAL==0):
    save_epochs(epochValLossLI,epochValAccLI,classString,timm_root,mode="VIT_REG",params=[NLAYERS,NHEADS,len(trainloader.dataset)])
    save_split(trainloader,valloader,classString,timm_root,params=[NLAYERS,NHEADS,len(trainloader.dataset)])
    print("\nWrote train-val-split to scratch.\n")
    

def get_mean_model(trainloader): 
    """
    Batch-robust mean-calculator. 
        Input: dataloader structure of batched or unbatched input
        ---------------------------------------------------------
        Output: tensor, shape [1,4]
    """
    mean_vals = torch.zeros(1,4)
    for i, data in enumerate(trainloader): #get batch of training data
        for j in range(data["scaled_target"].shape[0]): #loop over batch-dimension
            mean_vals += data["scaled_target"][j]
    mean_vals /= len(trainloader.dataset)
    return mean_vals

    

    
def get_median_model(trainloader): #NEED TO FIX FOR BATCHES 
    """
    Batch-robust median-calculator. 
        Input: dataloader structure of batched or unbatched input
        ---------------------------------------------------------
        Output: tensor, shape [1,4]
    """
    holder_t = torch.zeros(len(trainloader.dataset),4)
    idx_space = 0
    for i, data in enumerate(trainloader):
        for j in range(data["scaled_target"].shape[0]):
            holder_t[j+idx_space] = data["scaled_target"][j]
        idx_space += j+1 #to ensure support for random and non-equal batch-sizes
    median_t,_ = torch.median(holder_t,dim=0,keepdim=True)
    #if want debug: return t_holder
    return median_t
    
#h1.remove()
#h2.remove()   
#h3.remove()
#h4.remove()
#h5.remove()
#h6.remove()




#---------------------TEST AND EVAL OF BEST MODEL FOUND-------------#
if OVERFIT:
    paramsString = timm_root +"/"+ classString +"/L_" + str(NUM_IN_OVERFIT) +"_nL_" + str(NLAYERS) +"_nH_" + str(NHEADS)+"/" #for saving to correct dirs
else:
    paramsString = timm_root +"/"+ classString + "/nL_" + str(NLAYERS) +"_nH_" + str(NHEADS)+"/" #for saving to correct dirs

#1. TEST-LOOP ON TRAIN-SET
trainsettestLosses = []
IOU_tr_li = []

#eval on TRAIN SET 
meanModel = get_mean_model(trainloader)
medianModel = get_median_model(trainloader)

if(BEST_MODEL==True):
    model = load_best_model(classString,timm_root,params=[NLAYERS,NHEADS],mode="val")
    model = model.to(device)
else: 
    print("Best model was last model. Using model with training epochs {}".format(len(epochValLossLI)))


print("Entered evaluation-phase.")
model.eval()
if(OVERFIT):
    print("Model parameters:\n tL: {}\n vL: {}\nlrf: {}\n num_warmup: {}\n Dropout: {}\n Beta: {}\n NHEADS: {}\n NLAYERS: {}\n BATCH-SIZE: {}\n LR: {}".format(NUM_IN_OVERFIT,len(valIDX),LR_FACTOR,NUM_WARMUP,DROPOUT,BETA,NHEADS,NLAYERS,BATCH_SZ,LR))
    #print("Model parameters:\n tL: {} \nlrf: {}\n num_warmup: {}\n Dropout: {}\n Beta: {}\n NHEADS: {}\n NLAYERS: {}".format(NUM_IN_OVERFIT,LR_FACTOR,NUM_WARMUP,DROPOUT,BETA,NHEADS,NLAYERS)) #running overfit-set without validation for speediness
else:
    print("Model parameters:\n tL: {} \nlrf: {}\n num_warmup: {}\n Dropout: {}\n Beta: {}\n NHEADS: {}\n NLAYERS: {} \n BATCH-SIZE: {}\ LR: {}".format(NUM_IN_OVERFIT,LR_FACTOR,NUM_WARMUP,DROPOUT,BETA,NHEADS,NLAYERS,BATCH_SZ,LR))

print("Evaluating overfit on ALL {} train instances".format(len(trainloader.dataset)))

no_overfit_correct = 0
no_overfit_false = 0
no_mean_correct = 0
no_mean_false = 0
no_med_correct = 0
no_med_false = 0
train_save_struct = []

with torch.no_grad():
    for i, data in enumerate(trainloader):
        #CLS = data["image"][:,:,0,:].to(device)
        CLS = data["CLS"].to(device)
        target = data["scaled_target"].to(device)
        size = data["size"]
        name = data["file"]
        output = model(CLS)
        output = boxops.center_to_box(output).to(device) #STILL RELATIVE (percentage)
        batchloss = loss_fn(target,output) #L1 LOSS. Mode: "mean"
       # batchloss = ops.generalized_box_iou_loss(output.to(dtype=torch.float32),target.to(dtype=torch.float32))
        
        accScores = vm.pascalACC(output,target)
        no_overfit_correct += accScores[0]
        no_overfit_false += accScores[1]
        IOU = accScores[2] #for batches is a list
        IOU_tr_li += IOU #concat lists 
        for i in range(len(IOU)):
            if(IOU[i]>0.5):
                train_save_struct.append([name,str(1),IOU[i],target,output,size]) #filename, pred-status: correct(1):false(0), IOU-value, ground-truth, prediction-value 
            else:
                train_save_struct.append([name,str(0),IOU[i],target,output,size])
        print("Filename: {}\n Target: {}\n Prediction: {}\n Loss: {}\n IOU: {}".format(data["file"],data["scaled_target"],output,batchloss,IOU))
        trainsettestLosses.append(batchloss)
        
        #fix meanModel to have as many entrys as target-tensor: 
        n_in_batch = target.shape[0]
        meanModel_tmp = meanModel.repeat(n_in_batch,1).to(device) #make n_in_batch copies along batch-dimension
        medianModel_tmp = medianModel.repeat(n_in_batch,1).to(device)
        
        accScores = vm.pascalACC(meanModel_tmp,target)
        no_mean_correct += accScores[0]
        no_mean_false += accScores[1]
        
        accScores = vm.pascalACC(medianModel_tmp,target)
        no_med_correct += accScores[0]
        no_med_false += accScores[1]
        
    print("---------------------------EVAL on ALL {} overfit-train-images---------------------------".format(len(trainloader.dataset)))    
    print("\nTransformer accuracy with PASCAL-criterium on overfit set: {}/{}, percentage: {}".format(no_overfit_correct,no_overfit_false+no_overfit_correct,no_overfit_correct/(no_overfit_false+no_overfit_correct)))    
    print("\nMean model accuracy with PASCAL-criterium on overfit set: {}/{}, percentage: {}".format(no_mean_correct,no_mean_false+no_mean_correct,no_mean_correct/(no_mean_false+no_mean_correct)))
    print("\nMedian model accuracy with PASCAL-criterium on overfit set: {}/{}, percentage: {}".format(no_med_correct,no_med_false+no_med_correct,no_med_correct/(no_med_false+no_med_correct)))
    print("\nMean IOU is {}".format(sum(IOU_tr_li)/len(IOU_tr_li)))
    
    
    if(EVAL==0):
        torch.save(train_save_struct,paramsString+classString+"_VIT_REG_"+"test_on_train_results.pth")
        print("\n   Results saved to file: ",paramsString+classString+"/"+classString+"_VIT_REG_"+"test_on_train_results.pth")
    else: 
        torch.save(train_save_struct,paramsString+"eval/"+classString+"_VIT_REG_"+"test_on_train_results.pth")
        print("\n   Results saved to file: ",paramsString+"eval/"+classString+"/"+classString+"_VIT_REG_"+"test_on_train_results.pth")

        
    
print("Evaluating overfit on ALL {} test instances".format(len(testloader.dataset)))
#2. TEST-LOOP ON TEST-SET
no_test_correct = 0 
no_test_false = 0
no_test_mean_correct = 0 
no_test_mean_false = 0
no_test_median_correct = 0
no_test_median_false = 0
testlosses = []
correct_false_list = []
IOU_te_li = []

 #freezes from here-on for some reason
model.eval()
with torch.no_grad():
    running_loss = 0 
    for i, data in enumerate(tqdm(testloader)):
        target = data["scaled_target"].to(device)
        name = data["file"]
        size = data["size"]
        #CLS = data["image"][:,:,0,:].to(device)
        CLS = data["CLS"].to(device)
        output = model(CLS)
        output = boxops.center_to_box(output).to(device)
        batchloss = loss_fn(target,output) #L1 Loss
        running_loss += batchloss.item()
        testlosses.append(batchloss.item())
        accScores = vm.pascalACC(output,target)
        IOU = accScores[2] #for batches is a list
        IOU_te_li += IOU #list concatenation
        for i in range(len(IOU)): 
            if(IOU[i]>0.5):
                correct_false_list.append([name,str(1),IOU[i],target,output,size]) #filename, pred-status: correct(1):false(0), IOU-value, ground-truth, prediction-value 
            else:
                correct_false_list.append([name,str(0),IOU[i],target,output,size])
            
        
        no_test_correct += accScores[0]        
        no_test_false += accScores[1]
        
        n_in_batch = target.shape[0]
        meanModel_tmp = meanModel.repeat(n_in_batch,1).to(device) #make n_in_batch copies along batch-dimension
        medianModel_tmp = medianModel.repeat(n_in_batch,1).to(device)

        #scale according to image-size - not necessary
        #meanModel_tmp = boxops.scale_coords_to_image(meanModel_tmp,size)
        #medianModel_tmp = boxops.scale_coords_to_image(medianModel_tmp,size)

        accScores = vm.pascalACC(meanModel_tmp,target)
        
        no_test_mean_correct += accScores[0]
        no_test_mean_false += accScores[1]
        
        accScores = vm.pascalACC(medianModel_tmp,target)
        no_test_median_correct += accScores[0]
        no_test_median_false += accScores[1]
        
        if i!=0 and i%100==0:
            #note that running_loss is not used for anything else than printing.
            print("L1-loss on every over batch {}:{}: {}\n".format(i-100,i,running_loss/100))
            running_loss = 0 
            


testmeanAcc = no_test_mean_correct/(no_test_mean_false+no_test_mean_correct)
testmedianAcc = no_test_median_correct/(no_test_median_false+no_test_median_correct)

print("---------------EVAL on ALL {} test-images---------------".format(len(testloader.dataset)))
print("\nTransformer accuracy with PASCAL-criterium: {}/{}, percentage: {}".format(no_test_correct,no_test_false+no_test_correct,no_test_correct/(no_test_false+no_test_correct)))    
print("\nMean model accuracy with PASCAL-criterium: {}/{}, percentage: {}".format(no_test_mean_correct,no_test_mean_false+no_test_mean_correct,testmeanAcc))    
print("\nMedian model accuracy with PASCAL-criterium: {}/{}, percentage: {}".format(no_test_median_correct,no_test_median_false+no_test_median_correct,testmedianAcc))    
print("\nMean IOU is {}".format(sum(IOU_te_li)/len(IOU_te_li)))
if(EVAL==0):
    torch.save(correct_false_list,paramsString+classString+"_VIT_REG_"+"test_on_test_results.pth")
    print("\n   Results saved to file: ",paramsString+classString+"/"+classString+"_VIT_REG_"+"test_on_test_results.pth")
else:    
    torch.save(correct_false_list,paramsString+"eval/"+classString+"_VIT_REG_"+"test_on_test_results.pth")
    print("\n   Results saved to file: ",paramsString+"eval/"+classString+"/"+classString+"_VIT_REG_"+"test_on_test_results.pth")














###-----------------------------------Plots from here------------------------------------------------

# full_embedded_im = output_no_cls.view(1,14,14,768)
# #SINGLE CHANNEL PLOT
# CH_NO = 0
# channel_N = full_embedded_im.squeeze(0)[:,:,CH_NO].detach().cpu().numpy()
# print(channel_N.shape)
# channel_N_T = channel_N.T

# size = (int(data["size"][0,1].item()),int(data["size"][0,0].item())) #sizes are saved as height,width - you need X,Y
# channel_N_im = Image.fromarray(channel_N_T)
# interp_im = np.array(channel_N_im.resize(size,Image.BILINEAR))

# fig,(ax1,ax2) = plt.subplots(1,2)
# ax1.imshow(interp_im)
# fig.suptitle("Usampled deep-feature extraction for channel {}".format(CH_NO))
# ax2.imshow(sampled_back_image)

# def get_first_n_features_plot(df_rep,im_rescaled,NO_CHANNELS):
#     size = im_rescaled.size
#     df_rep_s = df_rep.squeeze(0)
#     fig,axes = plt.subplots(int(math.sqrt(NO_CHANNELS)),int(math.sqrt(NO_CHANNELS)))
#     axes = axes.flatten()
#     #norm = mpl.colors.Normalize(vmin=-1, vmax=1)
#     channel_data = df_rep_s.detach().cpu().numpy()
#     print(channel_data.shape)
#     for i, a in enumerate(axes):
#         print("processing channel {}".format(i))
#         ch_tmp = channel_data[:,:,i]
#         print(ch_tmp.shape)
#         channel_N_im = Image.fromarray(ch_tmp)
#         interp_im_tmp = np.array(channel_N_im.resize(size,Image.BILINEAR))
#         print(interp_im_tmp.size)
#         a.set_xticks([]) #remove ticks
#         a.set_yticks([]) #remove ticks
#         a.imshow(interp_im_tmp)#norm=norm)
#     fig.suptitle("Unsampled deep-feature-extraction for first {} channels".format(NO_CHANNELS))
#     plt.show()
#     return None
        
    
    
# get_first_n_features_plot(full_embedded_im,sampled_back_image,9)

# def get_first_n_features_plot_dual_pane(df_rep,im_rescaled,NO_CHANNELS):
#     size = im_rescaled.size
#     df_rep_s = df_rep.squeeze(0)
#     channel_data = df_rep_s.detach().cpu().numpy()
#     fig = plt.figure(figsize=(14,5.9),constrained_layout=True)
#     subfigs = fig.subfigures(1,2,width_ratios=[1.5, 1.7])
#     ax_L = subfigs[0].subplots(int(math.sqrt(NO_CHANNELS)),int(math.sqrt(NO_CHANNELS)),sharey=True,sharex=True)
#     ax_L = ax_L.flatten()
#     for i, a in enumerate(ax_L):
#         ch_tmp = channel_data[:,:,i]
#         channel_N_im = Image.fromarray(ch_tmp)
#         interp_im_tmp = np.array(channel_N_im.resize(size,Image.BILINEAR))
#         a.set_xticks([]) #remove ticks
#         a.set_yticks([]) #remove ticks
#         a.set_title("{}".format(i),fontweight="bold", size=18,y=0.98)
#         a.imshow(interp_im_tmp)#,cmap='plasma')#norm=norm)
#     ax_R = subfigs[1].subplots(1,1)
#     ax_R.imshow(im_rescaled)
#     ax_R.set_xticks([])
#     ax_R.set_yticks([])
    
#     #fig.suptitle("Unsampled deep-feature-extraction for first {} channels".format(NO_CHANNELS))
    
#     plt.show()
#     #plt.savefig("deep_feature_representation2.pdf")
#     return None

# def plot_mean_latent_space(df_rep,im_rescaled):
#     out = torch.mean(df_rep.squeeze(0),-1)
#     im = Image.fromarray(out.detach().cpu().numpy())
#     #fig = plt.figure(figsize=(14,8))
#     org_size = (500,375)
#     fig, (ax1,ax2) = plt.subplots(1,2,constrained_layout=True,figsize=(14,8))
#     interp_im = np.array(im.resize(org_size,Image.BILINEAR))
#     ax1.imshow(interp_im)
#     ax1.axis('off')
#     ax2.imshow(im_rescaled)
#     ax2.axis('off')
#     #plt.savefig("mean_feature_map.pdf")
    
#     return None
    

# get_first_n_features_plot_dual_pane(full_embedded_im,sampled_back_image,9)
# plot_mean_latent_space(full_embedded_im,sampled_back_image)


