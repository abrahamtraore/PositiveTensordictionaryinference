#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 13:58:44 2017

@author: Traoreabraham
"""
import pdb
import tempEA

import logging

from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.DEBUG)

#from  Librosa import librosa
import librosa


from sklearn.model_selection import cross_val_score

from sklearn import svm

import numpy as np 

from sktensor import dtensor

from scipy import signal,optimize

def Product_with_factors(coretensor,factor_list):    
    approxim=np.copy(coretensor)    
    approxim=dtensor(approxim)    
    mode=-1    
    for factor_matrix in factor_list:
        mode=mode+1
        #print(approxim.shape)
        #print(factor_matrix.shape)
        approxim=approxim._ttm_compute(factor_matrix,mode,False)        
    return approxim

def Transform_featuresmatrix_into_tensor(matrix,firstsize,secondsize):
    size=np.array(matrix.shape,dtype=int)
    nbsamples=size[0]
    tensor=dtensor(np.zeros((nbsamples,firstsize,secondsize)))
    for k in range(nbsamples):
        tensor[k,:,:]=np.reshape(matrix[k,:],(firstsize,secondsize))
    return tensor    
    
def Transform_tensor_into_featuresmatrix(tensor):
    size=np.array(tensor.shape,dtype=int)
    number_of_samples=size[0]
    result=np.zeros((number_of_samples,size[1]*size[2]))
    for i in range(number_of_samples):
        result[i,:]=np.resize(tensor[i,:,:],np.size(tensor[i,:,:]))
    return result

def Informationtensor(labels,Jf,Jt,nbclasses,nbinstances):
    Infotensor=np.zeros((nbinstances,Jf,Jt))
    for k in range(nbinstances):
        #pdb.set_trace()
        Infotensor[k,(labels[k]-1)*int((Jf/nbclasses)):labels[k]*(int(Jf/nbclasses)),(labels[k]-1)*(int(Jt/nbclasses)):labels[k]*(int(Jt/nbclasses))]=np.ones((int(Jf/nbclasses),int(Jt/nbclasses)))
    return dtensor(Infotensor)

def train_test_validation_tensors(X,labels,ratio_test,ratio_valid,seed1,seed2):     
    firstsize=np.array(X.shape,dtype=int)[1]
    secondsize=np.array(X.shape,dtype=int)[2]
    vector_all_features=Transform_tensor_into_featuresmatrix(X)
    matrix_trainvalid,matrix_test,y_trainvalid,y_test=train_test_split(vector_all_features,labels,test_size=ratio_test,random_state=seed1)
    matrix_train,matrix_valid,y_train,y_valid=train_test_split(matrix_trainvalid,y_trainvalid,test_size=ratio_valid/(1-ratio_test),random_state=seed2)   
    Tensor_train=Transform_featuresmatrix_into_tensor(matrix_train,firstsize,secondsize)
    Tensor_test=Transform_featuresmatrix_into_tensor(matrix_test,firstsize,secondsize)
    Tensor_valid=Transform_featuresmatrix_into_tensor(matrix_valid,firstsize,secondsize)
    return Tensor_train,y_train,Tensor_test,y_test,Tensor_valid,y_valid

def Training_features_extraction(G):
    [K,Jf,Jt]=np.array(G.shape,dtype=int)
    number_train_features=K
    result=np.zeros((number_train_features,Jf*Jt))
    for k in range(number_train_features):
        feature=G[k,:,:]
        result[k,:]=np.resize(feature,np.size(feature))
    return result  

def Test_features_extraction(Testing_tensor,A_f,A_t):
    size=np.array(Testing_tensor.shape,dtype=int)
    number_testing_features=size[0]
    sizef=np.array(A_f.shape,dtype=int)
    sizet=np.array(A_t.shape,dtype=int)
    result=np.zeros((number_testing_features,sizef[1]*sizet[1]))
    beta=np.kron(A_f,A_t)
    for k in range(number_testing_features):        
        feature=optimize.nnls(beta,np.resize(Testing_tensor[k,:,:],np.size(Testing_tensor[k,:,:])))[0]
        result[k,:]=feature
    return result

def TFR_all_the_examples_regular(res,sampling_rate,length,hop_leng):
   nbexamples=np.array(np.shape(res),dtype=int)[1]
   data=res[:,0]
   f,t,Sxx=signal.spectrogram(data,fs=sampling_rate,nperseg=length,noverlap=hop_leng)
   row=f.size
   column=t.size
   RawsampleTFR=np.zeros((nbexamples,row,column))  
   RawsampleTFR[0,:,:]=Sxx
   for i in range(nbexamples-1):
     data=res[:,i+1]
     f,t,Sxx=signal.spectrogram(data,fs=sampling_rate,nperseg=length,noverlap=hop_leng)
     RawsampleTFR[i+1,:,:]=Sxx 
   return RawsampleTFR

def TFR_all_the_examples_melscale(res,sampling_rate,length,hop_leng):   
   nbexamples=np.array(np.shape(res),dtype=int)[1]
   data=res[:,0]  
   Sxx=librosa.feature.melspectrogram(y=data,sr=sampling_rate,n_fft=length,hop_length=hop_leng)
   sizes=np.array(np.shape(Sxx),dtype=int)
   row=sizes[0]
   column=sizes[1]
   RawsampleTFR=np.zeros((nbexamples,row,column))  
   RawsampleTFR[0,:,:]=Sxx
   for i in range(nbexamples-1):
     data=res[:,i+1]
     Sxx=librosa.feature.melspectrogram(y=data,sr=sampling_rate,n_fft=length,hop_length=hop_leng)
     RawsampleTFR[i+1,:,:]=Sxx 
   return RawsampleTFR

def TFR_all_examples(res,sampling_rate,length,hop_leng,grid_form):
    if(grid_form=="regular"):
      RawsampleTFR=TFR_all_the_examples_regular(res,sampling_rate,length,hop_leng)
      return RawsampleTFR
    if(grid_form=="logscale"):
      RawsampleTFR=TFR_all_the_examples_melscale(res,sampling_rate,length,hop_leng)
      return RawsampleTFR

def PenalizedTucker(Tensor_train,y_train,nbclasses,kf,kt,hyperparam,maximum_iteration,tol):     
    nbinstances=np.shape(Tensor_train)[0]
    size=np.array(Tensor_train.shape,dtype=int)
    Jf=int(kf*nbclasses)
    Jt=int(kt*nbclasses)
    if ( (Jf>size[1]) or (Jt>size[2])):
        raise AssertionError("The number of temporal or spectral dictionaries must not exceed the correspond dimension of the tensor to decompose")
    if((Jf<=size[1]) and (Jt<=size[2])):
        Infotensor=Informationtensor(y_train,Jf,Jt,nbclasses,nbinstances)
        Coretensorsize=np.array([size[0],Jf,Jt],dtype=int) 
        max_iter=maximum_iteration
        epsilon=tol
        list_of_factors,G,error_list,nb_iter=tempEA.NTD_ALS_decoupledversion(Tensor_train,Coretensorsize,max_iter,3,0,epsilon,hyperparam,Infotensor)
        A_f=list_of_factors[1]
        A_t=list_of_factors[2]
        return G,A_f,A_t,error_list,nb_iter

def Cross_val(Tensor,labels,C_values,G_values,Hyper_values,nbclasses,kf,kt,maximum_iteration,tol):
   matrix=np.zeros((len(C_values),len(G_values),len(Hyper_values)))
   for param_position in range(len(Hyper_values)):
       hyperparam=Hyper_values[param_position]
       G=PenalizedTucker(Tensor,labels,nbclasses,kf,kt,hyperparam,maximum_iteration,tol)[0]
       features=Training_features_extraction(G)
       for C_position in range(len(C_values)):
         for G_position in range(len(G_values)): 
           C_svm=C_values[C_position]
           Gamma=G_values[G_position]
           clf=svm.SVC(C=C_svm,gamma=Gamma,kernel='rbf')
           scores = cross_val_score(clf,features,labels, cv=2)
           matrix[C_position,G_position,param_position]=np.mean(scores)
   print(matrix)
   print(np.max(matrix))
   max_positions=np.argwhere(matrix==np.max(matrix))
   C_selected=C_values[max_positions[0][0]]
   G_selected=G_values[max_positions[0][1]]
   param_selected=Hyper_values[max_positions[0][2]]
   return C_selected,G_selected,param_selected

def classificaton_rate_classes(result,clf):
    length=int(len(result)/2)
    performances=[]
    confusionmatrix=[]
    for i in range(length):
        perf=np.mean(clf.predict(result[2*i])==result[2*i+1])
        print("La valeur de pref est")
        print(perf)
        confusionmatrix.append(clf.predict(result[2*i]))       
        performances.append(perf)
    return performances,confusionmatrix