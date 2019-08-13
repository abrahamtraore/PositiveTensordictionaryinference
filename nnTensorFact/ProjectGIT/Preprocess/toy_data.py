#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 15:24:25 2017

@author: Traoreabraham
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 14:26:00 2017
@author: alain
"""



import numpy as np


import librosa

import matplotlib.pyplot as plt

import h5py

from PIL import Image

from scipy import misc

import pdb

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
  
def Generation(rescal,X):
    result=np.copy(X)
    if(rescal==1):
        If=input("Specify the width of the output images:")
        It=input("Specify the heigth of the output images:")
        result=rescaling(result,int(If),int(It))
        return result
    if (rescal==0):
        return result
    
plt.close("all")
Fe = 20000
N = 3

def TF_line(t_1,t_0, f_1, f_0):
    f = (f_1 - f_0) / (t_1 - t_0) * (t - t_0 )**2*0.5 + f_0*t
    sig =  np.where(np.logical_and(np.greater_equal(t,t_0),np.less_equal(t,t_1)), np.cos(2*np.pi*f), 0)   

    return sig

t = np.linspace(0,N,N*Fe)
t_0, t_1, f_0, f_1 = 0.5, 1, 2000, 1000
t_2, t_3, f_2, f_3 = 2, 2.5, 1000, 2000
sig = TF_line(t_3, t_2, f_3, f_2)
spec = np.abs(librosa.stft(sig,n_fft = 256, hop_length = 256))



nb_per_classe = 500
y_0 = np.zeros((nb_per_classe,1))
data_classe_1 = np.zeros((nb_per_classe,spec.shape[0],spec.shape[1]))
for i in range(nb_per_classe):
    sig = TF_line(t_1, t_0, f_1, f_0)
    sig = sig - np.mean(sig)
    spec = np.abs(librosa.stft(sig,n_fft = 256, hop_length = 256))
    data_classe_1[i,:,:] = spec

data_classe_2 = np.zeros((nb_per_classe,spec.shape[0],spec.shape[1],))
for i in range(nb_per_classe):
    ind = np.random.randn(1)*0.2
    if t_2 + ind < 0:
        ind = - t_2
    if t_3 + ind > N:
        ind = N - t_3
    sig = TF_line(t_3 + ind, t_2 + ind, f_3, f_2)
    sig = sig - np.mean(sig)
    spec = np.abs(librosa.stft(sig,n_fft = 256, hop_length = 256))
    data_classe_2[i,:,:] = spec

Data = np.concatenate((data_classe_1, data_classe_2), axis = 0)
y_1 = np.ones((nb_per_classe,1))
y = np.concatenate((y_0,y_1), axis = 0)

print("Generation of toy_data")
rescal=input("Do you want to resize the TFR?:the answer should be 0(No) or 1(Yes):")
X=Generation(int(rescal),Data)
print(X.shape)
with h5py.File('/Users/Traoreabraham/Desktop/AutomationForEastAnglia/toy.h5','w') as hf:
    hf.create_dataset('X', data = X)
    hf.create_dataset('y', data = y)


#If=input("Specify the width of the output images:")
#It=input("Specify the heigth of the output images:")
#X=rescaling(X,int(If),int(It))

#Generation(rescal,X)
#rescal=True
#If=60
#It=60
#if(rescal==True):
#  X=rescaling(X,If,It)

#with h5py.File('/Users/Traoreabraham/Desktop/AutomationForEastAnglia/toy.h5','w') as hf:
#    hf.create_dataset('X', data = X)
#    hf.create_dataset('y', data = y)

#with h5py.File('toy.h5','r') as hf:
#    x1 = np.array(hf.get('X')).astype('float32')
#    y1 = np.array(hf.get('y')).astype('uint32')     
