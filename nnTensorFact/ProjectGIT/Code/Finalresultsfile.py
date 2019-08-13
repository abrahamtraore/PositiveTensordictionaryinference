#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:29:01 2017

@author: Traoreabraham
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pdb
import h5py
import pandas as pd
from itertools import cycle, islice

def Rearrange_the_columns_by_maximum_energy(Matrix):  
    [nrows,ncols]=np.array(Matrix.shape,dtype=int)
    result=[]
    Energy_array=np.zeros(ncols)
    for m in range(ncols):
        Energy_array[m]=np.linalg.norm(Matrix[:,m])    
    Indexes=np.argsort(Energy_array)#This yields the indexes for energy increasing order
    Indexes=Indexes[::-1]#This yields the indexes for energy decreasing order
    for m in Indexes:
       result.append(Matrix[:,m])
    result=np.transpose(np.array(result))
    return result

def Histogram(df,data,Methodsname,Methodscolor,Xlabel,Ylabel,Fontsize,Charttitle):
    ax=df.plot(kind='bar', stacked=False, color=Methodscolor)
    ax.set_xlabel(Xlabel,fontsize=Fontsize)
    ax.set_ylabel(Ylabel,fontsize=Fontsize)
    ax.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
    ax.set_title(Charttitle)
    
##Example1: we load the performances for the toy data and plot the corresponding histogram   
BASE_PATH="./FinalImagesExperiences/ToyPerformances.h5"
with h5py.File(BASE_PATH,'r') as hf:
    Training_samples=np.array(hf.get('Training_samples')).astype('float32')
    PerfTucker=np.array(hf.get('PerfTucker')).astype('float32')
    PerfNMFVec=np.array(hf.get('PerfNMFVec')).astype('float32')
    PerfNMFMean=np.array(hf.get('PerfNMFMean')).astype('float32')
    PerfNMFMax=np.array(hf.get('PerfNMFMax')).astype('float32')
#    
Methodsname=['ONTDL','NMFPGVec', 'NMFPGMean', 'NMFPGMax']
data=np.zeros((5,4))
data[:,0]=PerfTucker
data[:,1]=PerfNMFVec
data[:,2]=PerfNMFMean
data[:,3]=PerfNMFMax
df=pd.DataFrame(data, index=Training_samples,columns=Methodsname)
Methodscolor=list(islice(cycle(['black', 'grey', 'silver', 'dimgray']), None, len(df)))
Xlabel="Training samples"
Ylabel="Score"
Fontsize=15
Charttitle="Learning curve (SVM, RBF kernel)"
Histogram(df,data,Methodsname,Methodscolor,Xlabel,Ylabel,Fontsize,Charttitle)



###Example2: we load the performances for the Dcase2013 size 60x20
#Base_PATH="/Users/Traoreabraham/Desktop/FinalImagesExperiences/Dcase60x20.h5"
#with h5py.File(Base_PATH,'r') as hf:
#    Dictionaryatoms=np.array(hf.get('Dictionaryatoms')).astype('float32')
#    PerfTucker=np.array(hf.get('PerfTucker')).astype('float32')
#    PerfNMFMax=np.array(hf.get('PerfNMFMax')).astype('float32')
#    PerfNMFMean=np.array(hf.get('PerfNMFMean')).astype('float32')
#
#Methodsname=['ONTDL', 'NMFPGMean', 'NMFPGMax']
#data=np.zeros((10,3))
#data[:,0]=PerfTucker
#data[:,1]=PerfNMFMean
#data[:,2]=PerfNMFMax
#df=pd.DataFrame(data, index=Dictionaryatoms,columns=Methodsname)
#Methodscolor=list(islice(cycle(['black', 'silver', 'dimgray']), None, len(df)))
#Xlabel="Dictionary atoms"
#Ylabel="Score"
#Fontsize=15
#Charttitle="60x20 Images"
#Histogram(df,data,Methodsname,Methodscolor,Xlabel,Ylabel,Fontsize,Charttitle)       


###Example3: we load the temporal dictionaries for 104,152,202,267 for the toy data
#Base_PATH="/Users/Traoreabraham/Desktop/FinalImagesExperiences/TemporalAtomsToy/Images104.h5"
#with h5py.File(Base_PATH,'r') as hf:
#         datas = hf.get('Temporaldictionaries1')
#         TemporaldictionaryToy104=np.array(datas) 
#         Temporaldictionaryarranged=Rearrange_the_columns_by_maximum_energy(TemporaldictionaryToy104)
#
#Base_PATH="/Users/Traoreabraham/Desktop/FinalImagesExperiences/TemporalAtomsToy/Images152.h5"
#with h5py.File(Base_PATH,'r') as hf:
#         datas = hf.get('Temporaldictionaries1')
#         TemporaldictionaryToy152=np.array(datas)  
#         Temporaldictionaryarranged=Rearrange_the_columns_by_maximum_energy(TemporaldictionaryToy152)
#

#Base_PATH="/Users/Traoreabraham/Desktop/FinalImagesExperiences/TemporalAtomsToy/Images202.h5"
#with h5py.File(Base_PATH,'r') as hf:
#         datas = hf.get('Temporaldictionaries1')
#         TemporaldictionaryToy202=np.array(datas)  
#         Temporaldictionaryarranged=Rearrange_the_columns_by_maximum_energy(TemporaldictionaryToy202)
#
#Base_PATH="/Users/Traoreabraham/Desktop/FinalImagesExperiences/TemporalAtomsToy/Images267.h5"
#with h5py.File(Base_PATH,'r') as hf:
#         datas = hf.get('Temporaldictionaries1')
#         TemporaldictionaryToy267=np.array(datas)
#         Temporaldictionaryarranged=Rearrange_the_columns_by_maximum_energy(TemporaldictionaryToy267)
#print(TemporaldictionaryToy104)
#print(TemporaldictionaryToy152)
#print(TemporaldictionaryToy202)
#print(TemporaldictionaryToy267)


###Example4: we load the temporal atoms for the Dcase2013 data set
#Base_PATH="/Users/Traoreabraham/Desktop/FinalImagesExperiences/TemporalAtomsDcase2013/ImagesDCaseJf180.h5"
#with h5py.File(Base_PATH,'r') as hf:
#    Temporaldictionary=hf.get('TemporaldictionariesJf180')
#    Temporaldictionary=np.array(Temporaldictionary)
#
#Temporaldictionaryarranged=Rearrange_the_columns_by_maximum_energy(Temporaldictionary) 

###Example5: we load the spectral atoms for the Dcase2013 data set
#Base_PATH="/Users/Traoreabraham/Desktop/FinalImagesExperiencesbis/SpectralAtomsDcase2013/SImagesJf180.h5"
#with h5py.File(Base_PATH,'r') as hf:
#    Spectraldictionary=hf.get('SpectraldictionariesJf180')
#    Spectraldictionary=np.array(Spectraldictionary)
#    
#print(Spectraldictionary)


###Example6:  we load the spectral dictionaries for 104,152,202,267 for the toy data 
#Base_PATH="/Users/Traoreabraham/Desktop/FinalImagesExperiencesbis/SpectralAtomsToy/SImages104.h5"
#with h5py.File(Base_PATH,'r') as hf:
#         datas = hf.get('Spectraldictionaries1')
#         SpectraldictionaryToy104=np.array(datas) 
#         Spectraldictionaryarranged=Rearrange_the_columns_by_maximum_energy(SpectraldictionaryToy104)
#
#Base_PATH="/Users/Traoreabraham/Desktop/FinalImagesExperiencesbis/SpectralAtomsToy/SImages152.h5"
#with h5py.File(Base_PATH,'r') as hf:
#         datas = hf.get('Spectraldictionaries1')
#         SpectraldictionaryToy152=np.array(datas)  
#         Spectraldictionaryarranged=Rearrange_the_columns_by_maximum_energy(SpectraldictionaryToy152)
#
#
#Base_PATH="/Users/Traoreabraham/Desktop/FinalImagesExperiencesbis/SpectralAtomsToy/SImages202.h5"
#with h5py.File(Base_PATH,'r') as hf:
#         datas = hf.get('Spectraldictionaries1')
#         SpectraldictionaryToy202=np.array(datas)  
#         Spectraldictionaryarranged=Rearrange_the_columns_by_maximum_energy(SpectraldictionaryToy202)
#
#Base_PATH="/Users/Traoreabraham/Desktop/FinalImagesExperiencesbis/SpectralAtomsToy/SImages267.h5"
#with h5py.File(Base_PATH,'r') as hf:
#         datas = hf.get('Spectraldictionaries1')
#         SpectraldictionaryToy267=np.array(datas)
#         Spectraldictionaryarranged=Rearrange_the_columns_by_maximum_energy(SpectraldictionaryToy267)
#print(SpectraldictionaryToy104)
#print(SpectraldictionaryToy152)
#print(SpectraldictionaryToy202)
#print(SpectraldictionaryToy267)


###Example7: we load the Toy data
#BASE_PATH='/Users/Traoreabraham/Desktop/ProjectGit/RealToyFinal/RealtoyFinal.h5'
#with h5py.File(BASE_PATH,'r') as hf:
#    Tensor = np.array(hf.get('Tensordata')).astype('float32')
#    labels = np.array(hf.get('labels')).astype('uint32')


