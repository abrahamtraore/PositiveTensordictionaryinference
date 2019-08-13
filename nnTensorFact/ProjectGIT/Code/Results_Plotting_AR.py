#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 23:13:57 2017

@author: alain
"""


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pdb
import h5py
import pandas as pd
from itertools import cycle, islice
import MethodsTSPenTestAB

plt.close("all")

def StandarddeviationToy(Testsamples,PerfTucker,PerfNMFVec,PerfNMFMean,PerfNMFMax):
    SdTucker=np.sqrt(np.multiply(np.multiply((PerfTucker/100),(1-PerfTucker/100)),1./Testsamples))
    SdNMFVec=np.sqrt(np.multiply(np.multiply((PerfNMFVec/100),(1-PerfNMFVec/100)),1./Testsamples))
    SdNMFMean=np.sqrt(np.multiply(np.multiply((PerfNMFMean/100),(1-PerfNMFMean/100)),1./Testsamples))
    SdNMFMax=np.sqrt(np.multiply(np.multiply((PerfNMFMax/100),(1-PerfNMFMax/100)),1./Testsamples))
    return SdTucker,SdNMFVec,SdNMFMean,SdNMFMax

def StandarddeviationDcase(PerfTucker,PerfNMFMean,PerfNMFMax):
    SdTucker=np.sqrt(np.multiply((PerfTucker/100),(1-PerfTucker/100))/100)
    SdNMFMean=np.sqrt(np.multiply((PerfNMFMean/100),(1-PerfNMFMean/100))/100)
    SdNMFMax=np.sqrt(np.multiply((PerfNMFMax/100),(1-PerfNMFMax/100))/100)
    return SdTucker,SdNMFMean,SdNMFMax



fontsize = 16
BASE_PATH="./FinalImagesExperiences/ToyPerformances.h5"
with h5py.File(BASE_PATH,'r') as hf:
    Training_samples=np.array(hf.get('Training_samples')).astype('float32')
    PerfTucker=np.array(hf.get('PerfTucker')).astype('float32')
    PerfNMFVec=np.array(hf.get('PerfNMFVec')).astype('float32')
    PerfNMFMean=np.array(hf.get('PerfNMFMean')).astype('float32')
    PerfNMFMax=np.array(hf.get('PerfNMFMax')).astype('float32')

data=np.zeros((5,4))
data[:,0]=PerfTucker
data[:,1]=PerfNMFVec
data[:,2]=PerfNMFMean
data[:,3]=PerfNMFMax
label = ['TDL', 'NMF vec', 'NMF mean', 'NMF max']
style = ['o-', 'v-','s-','*-']
fig = plt.figure(1)
for i, (y_arr,label) in enumerate(zip(data.T,label)):
    ax = plt.plot(Training_samples, y_arr, style[i], label = label, linewidth = 3, markersize = 14)
plt.legend(loc= (50/205,55/100))
plt.title('Learning curve', fontsize = fontsize)
plt.xlabel('Nb of examples', fontsize = fontsize)
plt.ylabel('Score', fontsize = fontsize)
plt.show()
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlim(20,205)
plt.savefig('learning_curve.png')



with h5py.File('RealtoyFinal.h5','r') as hf:
   Tensor = np.array(hf.get('Tensordata')).astype('float32')
   labels = np.array(hf.get('labels')).astype('uint32')  
   

width = 25
length = 25
base = 2
perclasse = 50
nb_classe = 8
ind_liste = [i*perclasse + base for i in range(nb_classe)]
plt.figure(2)
f, axarr = plt.subplots(4,2, sharex = True, sharey = True )
axarr = axarr.flatten()

Tensor = MethodsTSPenTestAB.rescaling(Tensor,width,length)
for i, ind in enumerate(ind_liste):
    axe = axarr[i]
    axe.imshow(Tensor[ind], aspect = 'auto', cmap = 'binary')
    axe.axis('tight')
    axe.set_xticks([])
    axe.set_yticks([])


axarr[6].set_xlabel('High-frequency classes',fontsize = 14)
axarr[7].set_xlabel('Low-frequency classes',fontsize = 14)

f.subplots_adjust(hspace=0.1, wspace = 0.01)
plt.savefig('toy_data.png')
#
#
#Base_PATH = 'Perf_toy_tucker100.h5'
#with h5py.File(Base_PATH,'r') as hf:
#    hf.keys()
#    datas = hf.get('Temporaldictionaries4')
#    TemporaldictionaryToy=np.array(datas) 
#         
#    
#plt.figure(3)
## Three subplots sharing both x/y axes
#f, axe_list = plt.subplots(8, sharex=True, sharey=True)
#t = np.linspace(0,1,30)
#for i,ax  in enumerate(axe_list):
#    ax.plot(t, TemporaldictionaryToy[:,i], linewidth = 3, markersize = 14)
#
#f.subplots_adjust(hspace=0)
#plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
#plt.setp([a.set_yticks([0,0.5]) for a in f.axes], visible= True)
#
#plt.xlabel('Time', fontsize = fontsize)
#plt.ylabel('Amplitude', fontsize = fontsize)
#
#axe_list[-1].yaxis.set_label_coords(-0.06, 4)
#axe_list[-1].set_ylabel('Amplitude')
#plt.xticks(fontsize=fontsize)
#
#plt.savefig('temporal_dico.png')



plt.close("all")
fig = plt.figure(4)
fontsize = 16
BASE_PATH="./FinalImagesExperiences/Dcase60x20.h5"
with h5py.File(BASE_PATH,'r') as hf:
    PerfTucker=np.array(hf.get('PerfTucker')).astype('float32')
    PerfNMFMean=np.array(hf.get('PerfNMFMean')).astype('float32')
    PerfNMFMax=np.array(hf.get('PerfNMFMax')).astype('float32')

data=np.zeros((10,3))
data[:,0]=PerfTucker
data[:,1]=PerfNMFMean
data[:,2]=PerfNMFMax
label = ['TDL', 'NMF mean', 'NMF max']
style = ['o-','s-','*-']
num_atoms = np.linspace(20, 200, num = 10, endpoint = True)
for i, (y_arr,label) in enumerate(zip(data.T,label)):
    ax = plt.plot(num_atoms, y_arr, style[i], label = label, linewidth = 3, markersize = 14)
plt.legend(loc= 'best')
plt.title('Performance', fontsize = fontsize)
plt.xlabel('Nb of dictionary atoms', fontsize = fontsize)
plt.ylabel('Score', fontsize = fontsize)
plt.show()
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlim(18,205)
plt.savefig('dcase_perf.png')



fig = plt.figure(5)

Base_PATH='./FinalImagesExperiences/ImagesDCaseJf180.h5'
with h5py.File(Base_PATH,'r') as hf:
    Temporaldictionary=hf.get('TemporaldictionariesJf180')
    Temporaldictionary=np.array(Temporaldictionary)

amax = Temporaldictionary.argmax(axis = 0)
sorted_amax = np.argsort(amax)

f, axe_list = plt.subplots(10, sharex=True, sharey=True)
t = np.linspace(0,1,20)
for i,ax  in enumerate(axe_list):
    ax.plot(t, Temporaldictionary[:,sorted_amax[i]], linewidth = 3, markersize = 14)

f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
plt.setp([a.set_yticks([0,0.5]) for a in f.axes], visible= True)

plt.xlabel('Normalized time', fontsize = fontsize)
plt.ylabel('Amplitude', fontsize = fontsize)

axe_list[-1].yaxis.set_label_coords(-0.06, 4)
axe_list[-1].set_ylabel('Amplitude')
plt.xticks(fontsize=fontsize)
plt.savefig('dcase_temporal.png')

