#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 09:24:19 2017

@author: Traoreabraham
"""

import numpy as np

from multiprocessing import Pool

import h5py

#from Librosa import librosa
import librosa
import MethodsTSPen

import matplotlib.pyplot as plt


from PIL import Image

from scipy import misc

import pdb

import sys

sys.path.append("/home/scr/etu/sin811/traorabr/sktensor/")

from sktensor import dtensor

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold

from sklearn import svm

np.random.seed(5)

plt.close("all")
Fe = 20000
N =3

t=np.linspace(0,N,N*Fe)


def rescaling(image_data,width,length):
      #This function is used to reduce the dimensions of the TFR images
      size=np.array(image_data.shape,dtype=int)
      number_of_samples=size[0]
      result=np.zeros((number_of_samples,width,length))
      for k in range(number_of_samples):
          image_of_interest=image_data[k,:,:]
          image_of_interest=Image.fromarray(image_of_interest)
          image_of_interest=misc.imresize(image_of_interest,(width,length),interp='cubic')
          image_of_interest=np.array(image_of_interest)
          result[k,:,:]=image_of_interest
      return result


def TF_line_short(t_1,t_0, f_1, f_0,scalefactor,shift1,shift2):
    #f=scalefactor*((f_1 - f_0) /(t_1 - t_0) * (t - t_0 )) + f_0*t  
    f=scalefactor*(-1200*t+2400)*t   
    sig1=np.where(np.logical_and(np.greater_equal(t,t_0+shift1),np.less_equal(t,t_1+shift1)), np.cos(2*np.pi*f),0) #motif court
    sig2=np.where(np.logical_and(np.greater_equal(t,t_0-shift2),np.less_equal(t,t_1-shift2)), np.cos(2*np.pi*f),0) #motif court
    sig=sig1+sig2
    return sig

def TF_line_short_long(t_1,t_0, f_1, f_0,scalefactor,shift1,shift2):
    #f=scalefactor*(f_1 - f_0) /(t_1 - t_0) * (t - t_0 )**2 + f_0*t 
    f=scalefactor*(-1200*t+2400)*t
    print(f)
    sig1=np.where(np.logical_and(np.greater_equal(t,t_0+shift1),np.less_equal(t,t_1+2*shift1)), np.cos(2*np.pi*f),0)
    sig2=np.where(np.logical_and(np.greater_equal(t,t_0-shift2),np.less_equal(t,t_1-shift2)), np.cos(2*np.pi*f),0)
    sig=sig1+sig2
    return sig

def TF_line_long_short(t_1,t_0, f_1, f_0,scalefactor,shift1,shift2):
    #f=scalefactor*(f_1 - f_0) /(t_1 - t_0) * (t - t_0 )**2 + f_0*t
    f=scalefactor*(-1200*t+2400)*t
    print(f)    
    sig1=np.where(np.logical_and(np.greater_equal(t,t_0+shift1),np.less_equal(t,t_1+shift1)), np.cos(2*np.pi*f),0)
    sig2=np.where(np.logical_and(np.greater_equal(t,t_0-shift2),np.less_equal(t,t_1-0.01*shift2)), np.cos(2*np.pi*f),0)
    sig=sig1+sig2
    return sig

def TF_line_long_long(t_1,t_0, f_1, f_0,scalefactor,shift1,shift2):
    #f=scalefactor*(f_1 - f_0) /(t_1 - t_0)*(t - t_0 )**2 + f_0*t 
    f=scalefactor*(-1200*t+2400)*t
    print(f)
    sig1=np.where(np.logical_and(np.greater_equal(t,t_0+shift1),np.less_equal(t,t_1+2*shift1)), np.cos(2*np.pi*f),0)
    sig2=np.where(np.logical_and(np.greater_equal(t,t_0-shift2),np.less_equal(t,t_1-0.01*shift2)), np.cos(2*np.pi*f),0)
    sig=sig1+sig2
    return sig

def Class_short_samples(t_1, t_0, f_1, f_0,t_3, t_2, f_3, f_2,scalefactors,shift1,shift2,label,nb_per_classe):    
    sig = TF_line_short(t_3, t_2, f_3, f_2,scalefactors[0],shift1,shift2)
    spec = np.abs(librosa.stft(sig,n_fft = 256, hop_length = 256))
    khi=scalefactors.size
    data_classe = np.zeros((khi*nb_per_classe,spec.shape[0],spec.shape[1]))
    labels=np.ones(khi*nb_per_classe)
    for t in range(khi):
       for i in range(nb_per_classe):
         sig = TF_line_short(t_1, t_0, f_1, f_0,scalefactors[t],shift1,shift2)
         sig = sig - np.mean(sig)
         spec = np.abs(librosa.stft(sig,n_fft = 256, hop_length = 256))
         data_classe[t*nb_per_classe+i,:,:] = spec
       labels[t*nb_per_classe:(t+1)*nb_per_classe]=label[t]*np.ones(nb_per_classe)
    return data_classe,labels


def Class_short_long_samples(t_1, t_0, f_1, f_0,t_3, t_2, f_3, f_2,scalefactors,shift1,shift2,label,nb_per_classe):    
    sig = TF_line_short_long(t_3, t_2, f_3, f_2,scalefactors[0],shift1,shift2)
    spec = np.abs(librosa.stft(sig,n_fft = 256, hop_length = 256))
    khi=scalefactors.size
    data_classe = np.zeros((khi*nb_per_classe,spec.shape[0],spec.shape[1]))
    labels=np.ones(khi*nb_per_classe)
    for t in range(khi):
       for i in range(nb_per_classe):
         sig = TF_line_short_long(t_1, t_0, f_1, f_0,scalefactors[t],shift1,shift2)
         sig = sig - np.mean(sig)
         spec = np.abs(librosa.stft(sig,n_fft = 256, hop_length = 256))
         data_classe[t*nb_per_classe+i,:,:] = spec
       labels[t*nb_per_classe:(t+1)*nb_per_classe]=label[t]*np.ones(nb_per_classe)
    return data_classe,labels

def Class_long_short_samples(t_1, t_0, f_1, f_0,t_3, t_2, f_3, f_2,scalefactors,shift1,shift2,label,nb_per_classe):    
    sig = TF_line_long_short(t_3, t_2, f_3, f_2,scalefactors[0],shift1,shift2)
    spec = np.abs(librosa.stft(sig,n_fft = 256, hop_length = 256))
    khi=scalefactors.size
    data_classe = np.zeros((khi*nb_per_classe,spec.shape[0],spec.shape[1]))
    labels=np.ones(khi*nb_per_classe)
    for t in range(khi):
       for i in range(nb_per_classe):
         sig = TF_line_long_short(t_1, t_0, f_1, f_0,scalefactors[t],shift1,shift2)
         sig = sig - np.mean(sig)
         spec = np.abs(librosa.stft(sig,n_fft = 256, hop_length = 256))
         data_classe[t*nb_per_classe+i,:,:] = spec
       labels[t*nb_per_classe:(t+1)*nb_per_classe]=label[t]*np.ones(nb_per_classe)
    return data_classe,labels

def Class_long_long_samples(t_1, t_0, f_1, f_0,t_3, t_2, f_3, f_2,scalefactors,shift1,shift2,label,nb_per_classe):    
    sig = TF_line_long_long(t_3, t_2, f_3, f_2,scalefactors[0],shift1,shift2)
    spec = np.abs(librosa.stft(sig,n_fft = 256, hop_length = 256))
    khi=scalefactors.size
    data_classe = np.zeros((khi*nb_per_classe,spec.shape[0],spec.shape[1]))
    labels=np.ones(khi*nb_per_classe)
    for t in range(khi):
       for i in range(nb_per_classe):
         sig = TF_line_long_long(t_1, t_0, f_1, f_0,scalefactors[t],shift1,shift2)
         sig = sig - np.mean(sig)
         spec = np.abs(librosa.stft(sig,n_fft = 256, hop_length = 256))
         data_classe[t*nb_per_classe+i,:,:] = spec
       labels[t*nb_per_classe:(t+1)*nb_per_classe]=label[t]*np.ones(nb_per_classe)
    return data_classe,labels


      

shift1=3/2 

shift2=1/3 

t_0, t_1, f_0, f_1 = 0.5, 1, 2000, 1000

t_2, t_3, f_2, f_3 = 2, 2.5, 1000, 2000




nb_per_classe =50

#scalefactors=np.array([2,5.5],dtype=int)
scalefactors=np.array([1,4.5],dtype=float)

label=np.array([1,2],dtype=int)

data_classe_short_samples,labels_short_samples=Class_short_samples(t_1, t_0, f_1, f_0,t_3, t_2, f_3, f_2,scalefactors,shift1,shift2,label,nb_per_classe)

label=np.array([3,4],dtype=int)

data_classe_short_long_samples,labels_short_long_samples=Class_short_long_samples(t_1, t_0, f_1, f_0,t_3, t_2, f_3, f_2,scalefactors,shift1,shift2,label,nb_per_classe)

label=np.array([5,6],dtype=int)

data_classe_long_short_samples,labels_long_short_samples=Class_long_short_samples(t_1, t_0, f_1, f_0,t_3, t_2, f_3, f_2,scalefactors,shift1,shift2,label,nb_per_classe)

label=np.array([7,8],dtype=int)

data_classe_long_long_samples,labels_long_long_samples=Class_long_long_samples(t_1, t_0, f_1, f_0,t_3, t_2, f_3, f_2,scalefactors,shift1,shift2,label,nb_per_classe)



[M,N]=np.array(np.shape(data_classe_long_long_samples),dtype=int)[1:3]
Tensor=dtensor(np.zeros((400,M,N)))
labels=np.zeros(400)
Tensor[0:100,:,:]=data_classe_short_samples
labels[0:100]=labels_short_samples
Tensor[100:200,:,:]=data_classe_short_long_samples
labels[100:200]=labels_short_long_samples
Tensor[200:300,:,:]=data_classe_long_short_samples
labels[200:300]=labels_long_short_samples
Tensor[300:400,:,:]=data_classe_long_long_samples
labels[300:400]=labels_long_long_samples

plt.imshow(Tensor[100,:,:])
plt.show()
plt.figure()
plt.imshow(Tensor[300,:,:])
plt.show()


#pdb.set_trace()
#Tensor=rescaling(Tensor,50,50)
with h5py.File('Realtoy.h5','w') as hf:
    hf.create_dataset('Tensordata', data = Tensor)
    hf.create_dataset('labels', data = labels)

#with h5py.File('/Users/Traoreabraham/Desktop/ProjectGit/Data/Realtoy.h5','r') as hf:
#    Tensor = np.array(hf.get('Tensordata')).astype('float32')
#    labels = np.array(hf.get('labels')).astype('uint32')  

#plt.imshow(Tensor[0,:,:])
#plt.show()
#pdb.set_trace()