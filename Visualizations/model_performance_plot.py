#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:51:25 2022

@author: max
"""

import matplotlib.pyplot as plt
#import seaborn as sns
import matplotlib.cm as cm 
import numpy as np
from matplotlib.ticker import FormatStrFormatter

plt.rc('axes', axisbelow=True)
plt.rc('axes', linewidth=2)
plt.rc('font', size=19, weight='bold')

#ViT nL = 1,3,5,6,9
model_names = ["GT","EF$_\mathbf{XS}$","EF$_\mathbf{S}$","EF$_\mathbf{M}$",	"EF$_\mathbf{L}$","EF$_\mathbf{XL}$","ViR","Dim$_\mathbf{G}$","Dim$_\mathbf{B}$","EFF"]
parameterList = [63194,5636912,16903028,28169144,33802202,50701376,1583108,15,250,31330224]
CorLoc = [0.35,0.461,0.457,0.516,0.488,0.412,0.456,0.383,0.469,0.456]


parameterList = [x/1000000 for x in parameterList]

fig, ax = plt.subplots(figsize=(11,6),dpi=150)
colors = plt.cm.tab20(np.linspace(0,1,len(CorLoc)))
#c = [x for x in range(CorLoc)]
plt.grid('on')
ax.scatter(x=parameterList,y=CorLoc,color=colors,s=2600,alpha=0.7)
#ax.set_axisbelow(True)
for i, txt in enumerate(model_names):
    ax.annotate(txt,(parameterList[i],CorLoc[i]),weight="bold",size=19,ha="center",va="center")
ax.set_xlabel("Trainable parameters [M]",weight="bold",size=18)
ax.set_ylabel("CorLoc performance",weight="bold",size=18)
x_ticks = np.arange(min(parameterList),max(parameterList),10)
labels = []
for i in range(0,len(x_ticks)): 
    labels.append(str(int(x_ticks[i])))
ax.set_xticks(x_ticks, labels=labels)
plt.setp(ax, ylim=(0.33,0.55))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax.set_ylim[0.33,0.55]
plt.tight_layout()
plt.savefig("models_overview.pdf")
plt.close()
