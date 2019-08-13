#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 14:26:00 2017

@author: alain
"""



import numpy as np
import matplotlib
import librosa.core.audio as audio
import librosa.feature as lf
import librosa
import librosa.display
import h5py,sys, getopt

import matplotlib.pyplot as plt
import h5py
plt.close("all")
Fe = 20000
N = 3

def TF_line(t_1,t_0, f_1, f_0):
    f = (f_1 - f_0) / (t_1 - t_0) * (t - t_0 )**2*0.5 + f_0*t
    sig =  np.where(np.logical_and(np.greater_equal(t,t_0),np.less_equal(t,t_1)), np.cos(2*np.pi*f), 0)   

    return sig


class StraigthLineWithTimeShift(object):
    
    def __init__(self,t_3,t_2, f_3,f_2,noise):
        self.t_3 = t_3
        self.t_2 = t_2
        self.f_3 = f_3
        self.f_2 = f_2
        self.noise = noise
    def generate(self):

        signe  = np.random.randn(1)
        ind = np.random.rand(1)
        time_shift = np.where( signe >0, ind*(N-t_3), - ind *(t_2) )
        sig = TF_line(self.t_3 + time_shift, self.t_2 + time_shift, self.f_3, self.f_2)
        sig = sig + np.random.randn(sig.shape[0])*self.noise
        return sig
    
class StraigthLine(object):
    
    def __init__(self,t_3,t_2, f_3,f_2,noise):
        self.t_3 = t_3
        self.t_2 = t_2
        self.f_3 = f_3
        self.f_2 = f_2
        self.noise = noise
    def generate(self):


        time_shift = 0
        sig = TF_line(self.t_3 + time_shift, self.t_2 + time_shift, self.f_3, self.f_2)
        sig = sig + np.random.randn(sig.shape[0])*self.noise
        return sig
    
class StraigthLineWithTimeFrequencyShift(object):
    
    def __init__(self,t_3,t_2, f_3,f_2,noise):
        self.t_3 = t_3
        self.t_2 = t_2
        self.f_3 = f_3
        self.f_2 = f_2
        self.noise = noise
    def generate(self):
    # span is the shift of the TF line
    # it is chosen so as to avoid shifting outside the time span
        signe  = np.random.randn(1)
        ind = np.random.rand(1)
        time_shift = np.where( signe >0, ind*(N-t_3), - ind *(t_2) )
        signe  = np.random.randn(1)
        ind = np.random.rand(1)
        frequency_shift = np.where( signe >0, ind*(Fe/2-f_3), - ind *(f_2) )

        sig = TF_line(self.t_3 + time_shift, self.t_2 + time_shift, self.f_3 + frequency_shift, self.f_2 + frequency_shift)
        sig = sig + np.random.randn(sig.shape[0])*self.noise
        return sig



def build_TFR(nb_per_classe, sig_classe_1, sig_classe_2):

    sig = sig_classe_1.generate()
    spec = np.abs(librosa.stft(sig,n_fft = 256, hop_length = 256))

    y_0 = np.ones((nb_per_classe,1))
    data_classe_1 = np.zeros((nb_per_classe,spec.shape[0],spec.shape[1]))
    for i in range(nb_per_classe):
        sig = sig_classe_1.generate()
        sig = sig - np.mean(sig)
        spec = np.abs(librosa.stft(sig,n_fft = 256, hop_length = 256))
        data_classe_1[i,:,:] = spec
    
    data_classe_2 = np.zeros((nb_per_classe,spec.shape[0],spec.shape[1],))
    for i in range(nb_per_classe):
        sig = sig_classe_2.generate()
        sig = sig - np.mean(sig)
        spec = np.abs(librosa.stft(sig,n_fft = 256, hop_length = 256))
        data_classe_2[i,:,:] = spec
    
    X = np.concatenate((data_classe_1, data_classe_2), axis = 0)
    y_1 = np.ones((nb_per_classe,1))*2
    y = np.concatenate((y_0,y_1), axis = 0)
        
    return X, y



t = np.linspace(0,N,N*Fe)
t_0, t_1, f_0, f_1 = 0.5, 1, 1000, 1000
t_2, t_3, f_2, f_3 = 1, 2, 1000, 1000

nb_per_classe = 50
noise = 2


sig_classe_1 = StraigthLineWithTimeShift(t_1,t_0, f_1,f_0,noise)
sig_classe_2 = StraigthLineWithTimeShift(t_3,t_2, f_3,f_2,noise)


X, y = build_TFR(nb_per_classe, sig_classe_1, sig_classe_2)
plt.figure(1)
plt.imshow(X[-1])
plt.figure(2)
plt.imshow(X[0])

#                       
path = '../../RawData/'
with h5py.File(path + 'toy.h5','w') as hf:
    hf.create_dataset('X', data = X)
    hf.create_dataset('y', data = y)

with h5py.File(path + 'toy.h5','r') as hf:
    x1 = np.array(hf.get('X')).astype('float32')
    y1 = np.array(hf.get('y')).astype('uint32')     
