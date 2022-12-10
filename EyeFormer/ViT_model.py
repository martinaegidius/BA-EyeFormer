#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:58:32 2022

@author: max
"""
import torch
import math
import torch.nn as nn
from torchvision import ops

class eyeFormer_ViT(nn.Module):
        def __init__(self,input_dim=770,hidden_dim=2048,output_dim=4,dropout=0.0,n_layers = 3, num_heads = 1):
            self.d_model = input_dim
            super().__init__() #get class instance
            #self.embedding = nn.Embedding(32,self.d_model) #33 because cls needs embedding
            self.pos_encoder = PositionalEncoding(self.d_model,dropout=dropout)
            self.encoder = TransformerEncoder(num_layers=n_layers,input_dim=self.d_model,seq_len=32,num_heads=num_heads,dim_feedforward=hidden_dim) #False due to positional encoding made on batch in middle
            #make encoder - 3 pieces
            self.cls_token = nn.Parameter(torch.zeros(1,self.d_model),requires_grad=True)
            #self.transformer_encoder = nn.TransformerEncoder(encoderLayers,num_layers = 1)
            
            
            #self.decoder = nn.Linear(64,4,bias=True) 
            self.clsdecoder = nn.Linear(self.d_model,4,bias=True)
            self.DEBUG = False
            
            
        def switch_debug(self):
            self.DEBUG = not self.DEBUG
            if(self.DEBUG==True):
                string = "on"
            else: string = "off"
            print("Debugging mode turned "+string)
            
            
        def forward(self,x,src_padding_mask=None):    
            if x.dim()==1: #fix for batch-size = 1 
                x = x.unsqueeze(0)
            if src_padding_mask.dim()==1: #fix for batch-size = 1
                src_padding_mask = src_padding_mask.unsqueeze(0)
            
            bs = x.shape[0]
            #print("Registered batchsize: ",bs)

            if src_padding_mask==None: 
                src_padding_mask = torch.zeros(bs,x.shape[1]).to(dtype=torch.bool)
            
            clsmask = torch.zeros(bs,1).to(dtype=torch.bool)
            #print("\n Src_padding_mask shape: ",src_padding_mask.shape)
            #print("\n CLS-mask shape: ",clsmask.shape)
            mask = torch.cat((clsmask,src_padding_mask[:,:].reshape(bs,32)),1) #unmask cls-token
           
            x = x* math.sqrt(self.d_model) #as this in torch tutorial but dont know why
            x = torch.cat((self.cls_token.expand(x.shape[0],1,self.d_model),x),1) #concat along sequence-dimension. Copy bs times
            if self.DEBUG==True:
                print("2: scaled and cat with CLS:\n",x.shape)
            x = self.pos_encoder(x)
            if self.DEBUG==True:
                print("3: positionally encoded: \n",x.shape)
            
            #print("Src_padding mask is: ",src_padding_mask)
            #print("pos encoding shape: ",x.shape)
            output = self.encoder(x,mask)
            if self.DEBUG==True:
                print("4: Transformer encoder output:\n",output.shape)
            #print("encoder output:\n",output)
            #print("Encoder output shape:\n",output.shape)
            #print("Same as input :-)")
           
            output = self.clsdecoder(output[:,0,:]) #batch-first is true. Picks encoded cls-token-vals for the batch.
            if self.DEBUG==True:
                print("5: linear layer based on CLS-token output: \n",output.shape)
            
            
            return output
            
           

class EncoderBlock(nn.Module):
        
    """
    Inputs:
        input_dim - Dimensionality of the input
        num_heads - Number of heads to use in the attention block
        dim_feedforward - Dimensionality of the hidden layer in the MLP
        dropout - Dropout probability to use in the dropout layers
    """         
    def __init__(self,input_dim,seq_len,num_heads,dim_feedforward,dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(input_dim,num_heads,dropout,batch_first=True)
        
        #two-layer FFN
        self.linear_net = nn.Sequential(nn.Linear(input_dim,dim_feedforward),
                                        nn.Dropout(dropout),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(dim_feedforward,input_dim))
        #layer normalization: 
        self.norm1 = nn.LayerNorm((seq_len+1,input_dim))
        self.norm2 = nn.LayerNorm((seq_len+1,input_dim))
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, mask=None):
       # Attention part
       attn_out, _ = self.self_attn(x,x,x,key_padding_mask=mask)
       x = x + self.dropout(attn_out)
       x = self.norm1(x)

       # MLP part
       linear_out = self.linear_net(x)
       x = x + self.dropout(linear_out)
       x = self.norm2(x)

       return x
     

class TransformerEncoder(nn.Module):
    def __init__(self,num_layers,**block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])
        
    def forward(self,x,mask=None):
        for l in self.layers: 
            x = l(x,mask)
        return x 
    
    
     
class PositionalEncoding(nn.Module):
    ###Probably change max_len of pos-encoding
    def __init__(self,d_model,dropout = 0.0,max_len = 33):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0)/d_model))
        pe = torch.zeros(1,max_len,d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term) #first dimension is batch dimension
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Returns nn.Dropout(x+pe(x)). Parse batch-first
        
        
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
            
        Returns: 
            PosEnc(x): Tensor, shape [batchsize,seq_len,embedding_dim]
        """
        x = x + self.pe[:,:x.size(1),:] #[bs x seq_len x embedding dim]
        return self.dropout(x)
    
    
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


def boxIOU(preds,labels):
    """
    Own-implemented batch-solid IOU-calculator. 
    Returns: tensor, [BS]
    """
    BSZ = preds.shape[0]
    assert BSZ == labels.shape[0], "zeroth dimension of preds and labels are not the same"
    IOU = torch.zeros(BSZ)
    for i in range(BSZ):
        A_target = (labels[i,2]-labels[i,0])*(labels[i,3]-labels[i,1]) #(x2-x1)*(y2-y1)
        A_pred = (preds[i,2]-preds[i,0])*(preds[i,-1]-preds[i,1]) #(x2-x1)*(y2-y1)
        U_width = torch.min(labels[i,2],preds[i,2]) - torch.max(labels[i,0],preds[i,0]) #width is min(lx2,px2)-(max(lx0,px0))
        U_height = torch.min(labels[i,3],preds[i,3]) - torch.max(labels[i,1],preds[i,1])  
        A_U = U_width * U_height
        IOU[i] = A_U / (A_target+A_pred-A_U)
    return IOU      
    
def pascalACC(preds,labels): #TODO: does not work for batched input. Fix
    """
    Function for calculating the accuracy between a batch of predictions and corresponding batch of targets. 
    Returns: number of correct predictions in batch, number of false predictions in batch and a list of IOU-scores for the batch
    """
    
    BSZ = preds.shape[0]
    assert BSZ == labels.shape[0],"Batch-size dimensions between target and tensor not in corresondance!"
    
    no_corr = 0 
    no_false = 0
    IOU_li = []
    
    if preds.dim()==1:
        preds = preds.unsqueeze(0)
    if labels.dim()==1:
        labels = labels.unsqueeze(0)
        
    #no loop approach: 
        #may be more effective for small batches. But only the diagonal of IOU_tmp is of interest for you - thus many wasted calculations
    #IOU_tmp = ops.box_iou(preds,labels) #calculates pairwise 
    #print(torch.diagonal(IOU_tmp))
    
    for i in range(BSZ): #get pascal-criterium accuraccy
        pred_tmp = preds[i,:].unsqueeze(0)
        label_tmp = labels[i,:].unsqueeze(0)
        IOU = ops.box_iou(pred_tmp,label_tmp)
        IOU_li.append(IOU.item())
        if(IOU>0.5):
            no_corr += 1
        else:
            no_false += 1

    return no_corr,no_false,IOU_li


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = eyeFormer_ViT().to(device)
    model.switch_debug()
    deep_seq = torch.rand(2,32,770)
    model(deep_seq)
