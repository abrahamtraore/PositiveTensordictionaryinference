#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 13:58:44 2017

@author: Traoreabraham
"""
from PIL import Image 

from scipy import misc

import nnTF_hALS

import tempEA

import logging

import os

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.DEBUG)

from sklearn.model_selection import cross_val_score

from sklearn import svm

import numpy as np 

import sys

sys.path.append("/home/scr/etu/sin811/traorabr/")

from sktensor import dtensor

from scipy import signal,optimize

def Determine_the_number_of_examples_per_class(labels):
    Differentclasses=np.sort(np.unique(labels))
    K=np.size(Differentclasses)
    result=np.zeros(K)
    for k in range(K):
        result[k]=np.sum(labels==Differentclasses[k])        
    result=np.array(result,dtype=int)
    return result

def TransformTensorToSignals(Tensor):
    
    [K,nrows,ncols]=np.array(np.shape(Tensor),dtype=int)
    result=np.zeros((nrows,K*ncols))
    for k in range(K):
        result[:,k*ncols:(k+1)*ncols]=Tensor[k,:,:]
    return result

def Informationmatrix(nbclasses,Kd,L,array_of_examples_numbers):
    nbinstances_old=0
    nbinstances=0
    result=np.zeros((Kd,L))
    for c in range(nbclasses): 
      nbinstances=nbinstances_old+array_of_examples_numbers[c]
      result[c*int(Kd/nbclasses):(c+1)*int(Kd/nbclasses),nbinstances_old:nbinstances]=np.ones((int(Kd/nbclasses),array_of_examples_numbers[c]))
      nbinstances_old=nbinstances
    return result

def NMF_penalized(Tensor,hyperparam,Kd,nbclasses,array_of_examples_numbers): 
    [K,nrows,ncols]=np.array(np.shape(Tensor),dtype=int)
    L=K*ncols
    Signalsmatrix=TransformTensorToSignals(Tensor)    
    C=Informationmatrix(nbclasses,Kd,L,array_of_examples_numbers) 
    Signals_tilde=np.row_stack((np.sqrt(hyperparam)*Signalsmatrix,C))    
    model = tempEA.NMF(n_components=Kd, init='random', random_state=0)    
    D_tilde = model.fit_transform(Signals_tilde)    
    D=D_tilde[0:nrows,:]    
    return D

def Features_extractionPenalizedNMF(Tensor,D,pool): #The pooling is performed before yieldind the result
    
    [K,nrows,ncols]=np.array(np.shape(Tensor),dtype=int)
    
    Kd=np.array(D.shape,dtype=int)[1]
    
    Feature=np.zeros((K,Kd))
    
    for k in range(K):
        Feature[k,:]=np.mean(tempEA.NMF_decoupling_parallelized(D,Tensor[k,:,:],pool),axis=1)
    print(Feature.shape)    
    return Feature


def Cross_valNMFPenalized(Tensor,labels,C_values,G_values,Hyper_values,Kd,nbclasses,pool):
    array_of_examples_numbers=Determine_the_number_of_examples_per_class(labels)
    matrix=np.zeros((len(C_values),len(G_values),len(Hyper_values)))
    scaler = StandardScaler()      
    for param_position in range(len(Hyper_values)):
       hyperparam=Hyper_values[param_position]
       D=NMF_penalized(Tensor,hyperparam,Kd,nbclasses,array_of_examples_numbers)
       features=Features_extractionPenalizedNMF(Tensor,D,pool)
       features=scaler.fit_transform(features)
       for C_position in range(len(C_values)):
         for G_position in range(len(G_values)): 
           C_svm=C_values[C_position]
           Gamma=G_values[G_position]
           clf=svm.SVC(C=C_svm,gamma=Gamma,kernel='rbf')
           scores = cross_val_score(clf,features,labels, cv=2)
           matrix[C_position,G_position,param_position]=np.mean(scores)    
    max_positions=np.argwhere(matrix==np.max(matrix))
    C_selected=C_values[max_positions[0][0]]
    G_selected=G_values[max_positions[0][1]]
    param_selected=Hyper_values[max_positions[0][2]]
    return C_selected,G_selected,param_selected,matrix

def matrixcompletion(A):
    [nbrows,nbcols]=np.array(A.shape,dtype=int)
    result=np.zeros((nbrows,nbcols))
    for i in range(nbcols):
      if(np.linalg.norm(A[:,i])==0):
        result[:,i]=np.random.rand(nbrows)
    return result

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
  
#This function is used if we intend to perfome the cross validation only the values of C and G 
def Hyperparameter_selection(X_valid, y_valid,C_values,G_values):
   matrix=np.zeros((len(C_values),len(G_values)))
   for C_position in range(len(C_values)):
     for G_position in range(len(G_values)): 
      C_svm=C_values[C_position]
      Gamma=G_values[G_position]
      clf=svm.SVC(C=C_svm,gamma=Gamma,kernel='rbf')
      scores = cross_val_score(clf, X_valid, y_valid, cv=2)
      matrix[C_position,G_position]=np.mean(scores)
   max_positions=np.argwhere(matrix==np.max(matrix))
   selected_parameter_C=C_values[max_positions[0][0]]
   selected_parameter_G=G_values[max_positions[0][1]]
   return selected_parameter_C,selected_parameter_G

#This function is used to recover all the examples labelled with 'label_number' among a set of samples 'features' and a set of labels 'labels'
def Recover_classes_one_label(features,labels,label_number):
    #This function is used to recover all the samples belonging to one class 
    number_samples=np.size(labels)
    result=[]
    for i in range(number_samples):
        if(labels[i]==label_number):
            result.append(features[i])
    result=np.array(result)
    return result

#This function is used to cancell repeted number in an array 
def suppress_repetition_in_a_list(labels):
    labels_list=list(labels) 
    return list(set(labels_list))

def Recover_classes_all_labels(features,labels):
    #This function is used to recover all the examples with their respective labels
    list_labels=suppress_repetition_in_a_list(labels)
    number_samples=len(list_labels)
    result=[]
    class_indexes=[]
    number_of_samples_per_class=[]
    for i in range(number_samples):
        if(list_labels[i]==1):
            result.append(Recover_classes_one_label(features,labels,1))
            result.append(1)
            class_indexes.append(1)
            number_of_samples_per_class.append(np.shape(Recover_classes_one_label(features,labels,1))[0])
        if(list_labels[i]==2):
            result.append(Recover_classes_one_label(features,labels,2))
            result.append(2)
            class_indexes.append(2)
            number_of_samples_per_class.append(np.shape(Recover_classes_one_label(features,labels,2))[0])
        if(list_labels[i]==3):
            result.append(Recover_classes_one_label(features,labels,3))
            result.append(3)
            class_indexes.append(3)
            number_of_samples_per_class.append(np.shape(Recover_classes_one_label(features,labels,3))[0])
        if(list_labels[i]==4):
            result.append(Recover_classes_one_label(features,labels,4))
            result.append(4)
            class_indexes.append(4)
            number_of_samples_per_class.append(np.shape(Recover_classes_one_label(features,labels,4))[0])
        if(list_labels[i]==5):
            result.append(Recover_classes_one_label(features,labels,5))
            result.append(5)
            class_indexes.append(5)
            number_of_samples_per_class.append(np.shape(Recover_classes_one_label(features,labels,5))[0])
        if(list_labels[i]==6):
            result.append(Recover_classes_one_label(features,labels,6))
            result.append(6)
            class_indexes.append(6)
            number_of_samples_per_class.append(np.shape(Recover_classes_one_label(features,labels,6))[0])
        if(list_labels[i]==7):
            result.append(Recover_classes_one_label(features,labels,7))
            result.append(7)
            class_indexes.append(7)
            number_of_samples_per_class.append(np.shape(Recover_classes_one_label(features,labels,7))[0])
        if(list_labels[i]==8):
            result.append(Recover_classes_one_label(features,labels,8))
            result.append(8)
            class_indexes.append(8)
            number_of_samples_per_class.append(np.shape(Recover_classes_one_label(features,labels,8))[0])
        if(list_labels[i]==9):
            result.append(Recover_classes_one_label(features,labels,9))
            result.append(9)
            class_indexes.append(9)
            number_of_samples_per_class.append(np.shape(Recover_classes_one_label(features,labels,9))[0])
        if(list_labels[i]==10):
            result.append(Recover_classes_one_label(features,labels,10))
            result.append(10)
            class_indexes.append(10)
            number_of_samples_per_class.append(np.shape(Recover_classes_one_label(features,labels,10))[0])
        if(list_labels[i]==11):
            result.append(Recover_classes_one_label(features,labels,11))
            result.append(11)
            class_indexes.append(11)
            number_of_samples_per_class.append(np.shape(Recover_classes_one_label(features,labels,11))[0])
        if(list_labels[i]==12):
            result.append(Recover_classes_one_label(features,labels,12))
            result.append(12)
            class_indexes.append(12)
            number_of_samples_per_class.append(np.shape(Recover_classes_one_label(features,labels,12))[0])
        if(list_labels[i]==13):
            result.append(Recover_classes_one_label(features,labels,13))
            result.append(13)
            class_indexes.append(13)
            number_of_samples_per_class.append(np.shape(Recover_classes_one_label(features,labels,13))[0])
        if(list_labels[i]==14):
            result.append(Recover_classes_one_label(features,labels,14))
            result.append(14)
            class_indexes.append(14)
            number_of_samples_per_class.append(np.shape(Recover_classes_one_label(features,labels,14))[0])
        if(list_labels[i]==15):
            result.append(Recover_classes_one_label(features,labels,15))
            result.append(15)
            class_indexes.append(15)
            number_of_samples_per_class.append(np.shape(Recover_classes_one_label(features,labels,15))[0])
    return result,class_indexes ,number_of_samples_per_class

#This function is used to perform the product of a 3-order tensor with three matrices
def Product_with_factors(coretensor,factor_list):    
    approxim=np.copy(coretensor)    
    approxim=dtensor(approxim)    
    mode=-1    
    for factor_matrix in factor_list:
        mode=mode+1
        approxim=approxim._ttm_compute(factor_matrix,mode,False)        
    return approxim

#This function is used to turn a matrix into a tensor given the size of each fiber
def Transform_featuresmatrix_into_tensor(matrix,firstsize,secondsize):
    size=np.array(matrix.shape,dtype=int)
    nbsamples=size[0]
    tensor=dtensor(np.zeros((nbsamples,firstsize,secondsize)))
    for k in range(nbsamples):
        tensor[k,:,:]=np.reshape(matrix[k,:],(firstsize,secondsize))
    return tensor    

#This function is used to turn a tensor into a matrix   
def Transform_tensor_into_featuresmatrix(tensor):
    size=np.array(tensor.shape,dtype=int)
    number_of_samples=size[0]
    result=np.zeros((number_of_samples,size[1]*size[2]))
    for i in range(number_of_samples):
        result[i,:]=np.resize(tensor[i,:,:],np.size(tensor[i,:,:]))
    return result

#This function is used to introduce the class information in the decomposition
#The labels numerotation must begin to 1
def Informationtensor(labels,Jf,Jt,nbclasses,nbinstances):
    labelscondition=np.min(np.unique(labels))
    if(labelscondition==0):
       raise AssertionError("The minimum value of a label must be 1") 
    if(labelscondition!=0):
       Infotensor=np.zeros((nbinstances,Jf,Jt))
       for k in range(nbinstances):
          Infotensor[k,(int(labels[k])-1)*int(Jf/nbclasses):int(labels[k])*int(Jf/nbclasses),(int(labels[k])-1)*int(Jt/nbclasses):int(labels[k])*int(Jt/nbclasses)]=np.ones((int(Jf/nbclasses),int(Jt/nbclasses)))
       return dtensor(Infotensor)

#This function is used to split a tensor into a training, validation and test tensors.
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

#This function is used to extract the training features if we consider them as the fibers of the core tensor inferred by the supervised decomposition 
def Training_features_extraction(G):
    [K,Jf,Jt]=np.array(G.shape,dtype=int)
    number_train_features=K
    result=np.zeros((number_train_features,Jf*Jt))
    for k in range(number_train_features):
        feature=G[k,:,:]
        result[k,:]=np.resize(feature,np.size(feature))
    return result  

#This function is used to extract the test features by projecting each TFR image on the temporal and spectral dictionaries 
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

#def TFR_all_the_examples_melscale(res,sampling_rate,length,hop_leng):   
#   nbexamples=np.array(np.shape(res),dtype=int)[1]
#   data=res[:,0]  
#   Sxx=librosa.feature.melspectrogram(y=data,sr=sampling_rate,n_fft=length,hop_length=hop_leng)
#   sizes=np.array(np.shape(Sxx),dtype=int)
#   row=sizes[0]
#   column=sizes[1]
#   RawsampleTFR=np.zeros((nbexamples,row,column))  
#   RawsampleTFR[0,:,:]=Sxx
#   for i in range(nbexamples-1):
#     data=res[:,i+1]
#     Sxx=librosa.feature.melspectrogram(y=data,sr=sampling_rate,n_fft=length,hop_length=hop_leng)
#     RawsampleTFR[i+1,:,:]=Sxx 
#   return RawsampleTFR

#def TFR_all_the_examples_melscale(res,sampling_rate,length,hop_leng):   
#   nbexamples=np.array(np.shape(res),dtype=int)[1]
#   data=res[:,0]  
#   Sxx=librosa.feature.melspectrogram(y=data,sr=sampling_rate,n_fft=length,hop_length=hop_leng)
#   sizes=np.array(np.shape(Sxx),dtype=int)
#   row=sizes[0]
#   column=sizes[1]
#   RawsampleTFR=np.zeros((nbexamples,row,column))  
#   RawsampleTFR[0,:,:]=Sxx
#   for i in range(nbexamples-1):
#     data=res[:,i+1]
#     Sxx=librosa.feature.melspectrogram(y=data,sr=sampling_rate,n_fft=length,hop_length=hop_leng)
#     RawsampleTFR[i+1,:,:]=Sxx 
#   return RawsampleTFR

#def TFR_all_examples(res,sampling_rate,length,hop_leng,grid_form):
#    if(grid_form=="regular"):
#      RawsampleTFR=TFR_all_the_examples_regular(res,sampling_rate,length,hop_leng)
#      return RawsampleTFR
#    if(grid_form=="logscale"):
#      TFR_regular=TFR_all_the_examples_regular(res,sampling_rate,length,hop_leng)
#      RawsampleTFR=melspectrogram(TFR_regular,sampling_rate,length)
#      return RawsampleTFR

#This function is used to perform the supervised decomposition.
#We forbid the overcomplete case because the initialization is performed by taking positive part of factors inferred by the HOSVD algorithm and it is well known HOSVD can not be handled in the overcomplete case
#So, it is straighforward to lift this restriction by adopting an initialization strategy that does not hinder the overcomplete frame.
def PenalizedTucker(Tensor_train,y_train,nbclasses,kf,kt,hyperparam,maximum_iteration,tol,InitStrat,parallel_decision,pool):     
    nbinstances=np.shape(Tensor_train)[0]
    size=np.array(Tensor_train.shape,dtype=int)
    Jf=kf*nbclasses  
    Jt=kt*nbclasses
    #NTD_ALS_decoupledversion(X,Coretensorsize,max_iter,N,m,epsilon,hyperparam,Infotensor,InitStrat,parallel_decision,pool)
    #if ( (Jf>size[1]) or (Jt>size[2])):
        #raise AssertionError("The number of temporal or spectral dictionaries must not exceed the correspond dimension of the tensor to decompose")
    #if((Jf<=size[1]) and (Jt<=size[2])):
    Infotensor=Informationtensor(y_train,Jf,Jt,nbclasses,nbinstances)
    Amplitude=np.max(np.array(Tensor_train))
    Infotensor=Amplitude*Infotensor
    Coretensorsize=np.array([size[0],Jf,Jt],dtype=int) 
    max_iter=maximum_iteration
    epsilon=tol
    print("Decomposition in penalized Tucker")
    list_of_factors,G,error_list,nb_iter=tempEA.NTD_ALS_decoupledversion(Tensor_train,Coretensorsize,max_iter,3,0,epsilon,hyperparam,Infotensor,InitStrat,parallel_decision,pool)
    print("We pass the decomposition in Penalized Tucker ")
    A_f=list_of_factors[1]
    A_t=list_of_factors[2]
    return G,A_f,A_t,error_list,nb_iter


def PenalizedTuckerHals(Tensor_train,y_train,nbclasses,kf,kt,hyperparam,maximum_iteration,tol):     
    size=np.array(Tensor_train.shape,dtype=int)
    Jf=kf*nbclasses  
    Jt=kt*nbclasses
    if ( (Jf>size[1]) or (Jt>size[2])):
        raise AssertionError("The number of temporal or spectral dictionaries must not exceed the correspond dimension of the tensor to decompose")
    if((Jf<=size[1]) and (Jt<=size[2])):
        Coretensorsize=np.array([size[0],Jf,Jt],dtype=int) 
        epsilon=tol
        print("Decomposition in penalized Tucker")
        G,A,error=nnTF_hALS.nnTF2_hALS(Tensor_train,Coretensorsize,epsilon,maximum_iteration)
        print("We pass the decomposition in Penalized Tucker ")
        A_f=A[1]
        A_t=A[2]
        return G,A_f,A_t,error
    
#This function is used to perform the cross validation task by utilizing:
    #The parameter C
    #The parameter Sigma
    #The trade-off parameter Lambda
def Cross_val(Tensor,labels,C_values,G_values,Hyper_values,nbclasses,kf,kt,maximum_iteration,tol,InitStrat,parallel_decision,pool):
   matrix=np.zeros((len(C_values),len(G_values),len(Hyper_values)))
   scaler = StandardScaler()
   for param_position in range(len(Hyper_values)):
       hyperparam=Hyper_values[param_position]
       G,A_f,A_t=PenalizedTucker(Tensor,labels,nbclasses,kf,kt,hyperparam,maximum_iteration,tol,InitStrat,parallel_decision,pool)[0:3]
       #features=Training_features_extraction(G)
       features=Test_features_extraction(Tensor,A_f,A_t)
       features=scaler.fit_transform(features)
       for C_position in range(len(C_values)):
         for G_position in range(len(G_values)): 
           C_svm=C_values[C_position]
           Gamma=G_values[G_position]
           clf=svm.SVC(C=C_svm,gamma=Gamma,kernel='rbf')
           scores = cross_val_score(clf,features,labels, cv=2)
           matrix[C_position,G_position,param_position]=np.mean(scores)
   max_positions=np.argwhere(matrix==np.max(matrix))
   C_selected=C_values[max_positions[0][0]]
   G_selected=G_values[max_positions[0][1]]
   param_selected=Hyper_values[max_positions[0][2]]
   return C_selected,G_selected,param_selected,matrix

def Cross_val_AR(Tensor,labels,C_values,G_values,Hyper_values,nbclasses,kf,kt,maximum_iteration,tol):
   matrix=np.zeros((len(C_values),len(G_values),len(Hyper_values)))
   scaler = StandardScaler()
   for param_position in range(len(Hyper_values)):
       hyperparam=Hyper_values[param_position]
       G,A_f,A_t,_,_=PenalizedTucker(Tensor,labels,nbclasses,kf,kt,hyperparam,maximum_iteration,tol)
       features=Test_features_extraction(Tensor,A_f,A_t)
       features=scaler.fit_transform(features)
       for C_position in range(len(C_values)):
         for G_position in range(len(G_values)): 
           C_svm=C_values[C_position]
           Gamma=G_values[G_position]
           clf=svm.SVC(C=C_svm,gamma=Gamma,kernel='rbf')
           scores = cross_val_score(clf,features,labels, cv=2)
           matrix[C_position,G_position,param_position]=np.mean(scores)
   max_positions=np.argwhere(matrix==np.max(matrix))
   C_selected=C_values[max_positions[0][0]]
   G_selected=G_values[max_positions[0][1]]
   param_selected=Hyper_values[max_positions[0][2]]
   return C_selected,G_selected,param_selected,matrix

def Cross_valHALs(Tensor,labels,C_values,G_values,Hyper_values,nbclasses,kf,kt,maximum_iteration,tol):
   matrix=np.zeros((len(C_values),len(G_values),len(Hyper_values)))
   scaler = StandardScaler()
   for param_position in range(len(Hyper_values)):
       hyperparam=Hyper_values[param_position]
       G=PenalizedTuckerHals(Tensor,labels,nbclasses,kf,kt,hyperparam,maximum_iteration,tol)[0]
       features=Training_features_extraction(G)
       features=scaler.fit_transform(features)
       for C_position in range(len(C_values)):
         for G_position in range(len(G_values)): 
           C_svm=C_values[C_position]
           Gamma=G_values[G_position]
           clf=svm.SVC(C=C_svm,gamma=Gamma,kernel='rbf')
           scores = cross_val_score(clf,features,labels, cv=2)
           matrix[C_position,G_position,param_position]=np.mean(scores)
   max_positions=np.argwhere(matrix==np.max(matrix))
   C_selected=C_values[max_positions[0][0]]
   G_selected=G_values[max_positions[0][1]]
   param_selected=Hyper_values[max_positions[0][2]]
   return C_selected,G_selected,param_selected,matrix

#The function is used to compute the Mean Average Precision by determining the performance per class
def classificaton_rate_classes(result,clf):
    length=int(len(result)/2)
    performances=[]
    confusionmatrix=[]
    for i in range(length):
        perf=np.mean(clf.predict(result[2*i])==result[2*i+1])
        confusionmatrix.append(clf.predict(result[2*i]))       
        performances.append(perf)
    return performances,confusionmatrix

#This function is used to load the EastAngliaDataDataSet by specifying its adress
#Its output is the tensor of signal power(potentially resized) and labels arranged in the same order as the TFR matrices do in the power tensor, i.e:
def load_preprocessed_data(BASE_PATH):
  Datareload=os.listdir(BASE_PATH)
  #Datareload=Datareload[2:] #In order to delete the strings '.DS_Store', '.npz' 
  # a proper way to remove undesirable strings
  Datareload.remove('.npz')
  Datareload.remove('.DS_Store')
  K=len(Datareload)
  labels=np.zeros(K)
  result=[]
  for k in range(K):
       A=np.load(BASE_PATH+'/'+Datareload[k])
       result.append(A['arr_0'])       
       pos=Datareload[k].find(".npz")
       temp=Datareload[k][0:pos]
       Classindication=int(temp[3:])
       if((0<=Classindication) and (Classindication<8)):
           labels[k]=1
       if((8<=Classindication) and (Classindication<16)):
           labels[k]=2
       if((16<=Classindication) and (Classindication<24)):
           labels[k]=3
       if((24<=Classindication) and (Classindication<32)):
           labels[k]=4
       if((32<=Classindication) and (Classindication<40)):
           labels[k]=5
       if((40<=Classindication) and (Classindication<48)):
           labels[k]=6
       if((48<=Classindication) and (Classindication<56)):
           labels[k]=7
       if((56<=Classindication) and (Classindication<64)):
           labels[k]=8
       if((64<=Classindication) and (Classindication<72)):
           labels[k]=9
       if((72<=Classindication) and (Classindication<80)):
           labels[k]=10
  result=np.array(result)
  labels=np.array(labels,dtype=int)  
  return result,labels

#BASE_PATH='/Users/Traoreabraham/Desktop/ProjectGit/Code/RawData'
#result,labels=load_preprocessed_data(BASE_PATH)
#pdb.set_trace()
 