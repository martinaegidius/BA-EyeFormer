import torch


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
        
        sample["scaled_signal"] = outCoords
        sample["scaled_target"] = tbox
        
        return sample
    
        #return {"signal": outCoords,"size":imdims,"target":tbox}

def scale_coords_to_image(rel_box,size):
    box_t = torch.zeros(rel_box.shape[0],4,dtype=torch.float32)
    box_t[:,0] = rel_box[:,0]*size[:,1]
    box_t[:,1] = rel_box[:,1]*size[:,0]
    box_t[:,2] = rel_box[:,2]*size[:,1]
    box_t[:,3] = rel_box[:,3]*size[:,0]

    return box_t
    


class get_center(object):
    """
    Gets center x, y and bbox w, h from a batched target sequence
    Parameters
    ----------
    targets : batched torch tensor
        contains batched target x0,y0,x1,y1.

    Returns
    -------
    xywh : torch tensor
        bbox cx,cy,w,h

    """
    def __call__(self,sample):
        targets = sample["scaled_target"]
        assert targets!=None, print("FOUND NO SCALED TARGET")
        xywh = torch.zeros(4,dtype=torch.float32)
        #w = torch.zeros(targets.shape[0],dtype=torch.int32)
        #h = torch.zeros(targets.shape[0],dtype=torch.int32)
        
        xywh[0]  = torch.div(targets[2]+targets[0],2)
        xywh[1] = torch.div(targets[-1]+targets[1],2)
        xywh[2] = targets[2]-targets[0]
        xywh[3] = targets[-1]-targets[1]
        sample["target_centered"] = xywh
        return sample


def center_to_box(targets):
    """
    Scales center-prediction cx,cy,w,h back to x0,y0,y1,y2

    Parameters
    ----------
    targets : batched torch tensor [BS,4]
        Format center x, center y, width and height of box

    Returns
    -------
    box_t : batched torch tensor
        Format lower left corner, upper right corner, [x0,y0,x1,y1]
        
    """
    if targets.ndim==3:
        targets = targets.squeeze(1) #if batchsize != 1 then predictions have format [bs,1,4]
    
    box_t = torch.zeros(targets.shape[0],4,dtype=torch.float32)
    box_t[:,0] = targets[:,0]-torch.div(targets[:,2],2) #x0
    box_t[:,1] = targets[:,1]-torch.div(targets[:,3],2) #y0
    box_t[:,2] = targets[:,0]+torch.div(targets[:,2],2) #x1
    box_t[:,3] = targets[:,1]+torch.div(targets[:,3],2) #y2
    return box_t

