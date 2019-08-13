#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 13:58:44 2017

@author: Traoreabraham
"""

from PIL import Image


import NMFdecompTestAB

import tempEATSPen

from sklearn import linear_model

from sklearn.svm import LinearSVC

import math 

from scipy.optimize import lsq_linear

from scipy import misc

import tempEATSPenTestAB

import logging

import os

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.DEBUG)

from sklearn.model_selection import cross_val_score

from sklearn import svm

import numpy as np 

import sys

import pdb

sys.path.append("/home/scr/etu/sin811/traorabr/")

from sktensor import dtensor

from scipy import signal

np.random.seed(1)

def max_pool(Signal,D,Alpha,lratio):
    N=np.array(np.shape(Signal))[1]
    K=np.array(np.shape(D))[1]
    Matrix=np.zeros((K,N))
    result=np.zeros(K)
    for n in range(N):
        reg=linear_model.Lasso(alpha=Alpha*lratio,tol=0.01,max_iter=10000,random_state=5,positive=True)
        reg.fit(D,Signal[:,n])
        Matrix[:,n]=reg.coef_
    for k in range(K):
        result[k]=np.max(Matrix[k,:])        
    return result

def max_pooling_features(Tensor,D,Alpha,lratio):
    K=np.array(np.shape(Tensor),dtype=int)[0]
    Kd=np.array(np.shape(D),dtype=int)[1]
    result=np.zeros((K,Kd))
    for k in range(K):
        Signal=Tensor[k,:,:]
        result[k,:]=max_pool(Signal,D,Alpha,lratio)
    return result


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


def Features_extraction_projection(Tensor,D,penalty,pool): #The pooling is performed before yieldind the result
    
    [K,nrows,ncols]=np.array(np.shape(Tensor),dtype=int)
    
    Kd=np.array(D.shape,dtype=int)[1]
    
    Feature=np.zeros((K,Kd*ncols))
    #Feature=np.zeros((K,Kd))
    for k in range(K):        
        #Feature[k,:]=np.mean(tempEATSPen.NMF_decoupling_parallelized(D,Tensor[k,:,:],pool),axis=1)
        
        Temp=tempEATSPen.NMF_decoupling_parallelized(D,Tensor[k,:,:],penalty,pool)
        #Feature[k,:]=np.mean(Temp,axis=1)
        Feature[k,:]=np.resize(Temp,np.size(Temp))
    return Feature

def mean_pool(Signal,D,Alpha,lratio):
    N=np.array(np.shape(Signal))[1]
    K=np.array(np.shape(D))[1]
    Matrix=np.zeros((K,N))
    result=np.zeros(K)
    for n in range(N):
        reg=linear_model.Lasso(alpha=Alpha*lratio,tol=0.01,max_iter=1000,random_state=5,positive=True)
        reg.fit(D,Signal[:,n])
        Matrix[:,n]=reg.coef_
    for k in range(K):
        result[k]=np.mean(Matrix[k,:])        
    return result

def mean_pooling_features(Tensor,D,Alpha,lratio):
    K=np.array(np.shape(Tensor),dtype=int)[0]
    Kd=np.array(np.shape(D),dtype=int)[1]
    result=np.zeros((K,Kd))
    for k in range(K):
        Signal=Tensor[k,:,:]
        result[k,:]=mean_pool(Signal,D,Alpha,lratio)
    return result

def Feature_extraction_process(Tensor,D,alpha,lratio,penalty,pool,pool_decision):
    if(pool_decision==False):
      features=Features_extraction_projection(Tensor,D,penalty,pool)
      return features    
    if(pool_decision=="mean"):      
      features=mean_pooling_features(Tensor,D,alpha,lratio)
      return features
    if(pool_decision=="max"):
      features=max_pooling_features(Tensor,D,alpha,lratio)
      return features


def decomposition_single_examples(args):    
    #sol=lsq_linear(args[1],np.resize(args[0][args[2],:,:],np.size(args[0][args[2],:,:])),bounds=(0,math.inf))    
    #solution=sol.x
    reg=linear_model.Lasso(alpha=args[2],tol=0.01,max_iter=10000,random_state=5,positive=True)
    #reg.fit(D,Signal[:,n])
    reg.fit(args[1],np.resize(args[0][args[3],:,:],np.size(args[0][args[3],:,:])))
    solution=reg.coef_
    return solution
    
    
def decomposition_all_examples_parallelized(Tensor,FactorsKron,penalty,pool):
    K=np.array(Tensor.shape,dtype=int)[0]
    SIZE=np.array(FactorsKron.shape,dtype=int)[1]
    result=np.zeros((K,SIZE))
    sol=pool.map(decomposition_single_examples,[[Tensor,FactorsKron,penalty,n] for  n in range(K)])
    for k in range(K):
        result[k,:]=sol[k]
    return  result  

 
    
def NMF_penalized_CD(Tensor,lamb,Alpha,Lratio,Kd,nbclasses,array_of_examples_numbers): 
    [K,nrows,ncols]=np.array(np.shape(Tensor),dtype=int)
    L=K*ncols
    Signalsmatrix=TransformTensorToSignals(Tensor)    
    C=Informationmatrix(nbclasses,Kd,L,array_of_examples_numbers) 
    Signals_tilde=np.row_stack((Signalsmatrix,np.sqrt(lamb)*C))        
    nrow=np.array(Signals_tilde.shape,dtype=int)[0]
    W=np.random.rand(nrow,Kd)
    H=np.random.rand(Kd,L)
    D_tilde,A,nb_iter=NMFdecompTestAB._fit_coordinate_descent(Signals_tilde,W,H,tol=1e-8,max_iter=500,alpha=Alpha,l1_ratio=Lratio,regularization='mixed_transformation',update_H=True ,verbose=0, shuffle=False,random_state=2)                       
    D=D_tilde[0:nrows,:]    
    return D

def NMF_penalized_PG(Tensor,W,H,lambdainfo,Alpha,Lratio,Kd,nbclasses,array_of_examples_numbers): 
    [K,nrows,ncols]=np.array(np.shape(Tensor),dtype=int)
    L=K*ncols
    Signalsmatrix=TransformTensorToSignals(Tensor)    
    C=Informationmatrix(nbclasses,Kd,L,array_of_examples_numbers) 
    Signals_tilde=np.row_stack((Signalsmatrix,np.sqrt(lambdainfo)*C))        
    D_tilde,A,n_iter=NMFdecompTestAB._fit_projected_gradient(Signals_tilde,W,H,1e-8,10,500,Alpha,Lratio,None,0.1,0.1) 
    return D_tilde,A,n_iter

def Cross_valNMF_PG_DCase(Tensor_train,Tensor_valid,y_train,y_valid,C_values,G_values,Lambdainfo,Alpha,Lratio,Kd,nbclasses,pool,pool_decision):
      matrix=np.zeros((len(C_values),len(G_values),len(Lambdainfo),len(Lratio),len(Alpha)))
      scaler = StandardScaler() 
      array_of_examples_numbers=Determine_the_number_of_examples_per_class(y_train)      
      [K,nrows,ncols]=np.array(np.shape(Tensor_train),dtype=int)
      L=K*ncols
      Signalsmatrix=TransformTensorToSignals(Tensor_train)    
      C=Informationmatrix(nbclasses,Kd,L,array_of_examples_numbers) 
      Signals_tilde=np.row_stack((Signalsmatrix,C))        
      nrow=np.array(Signals_tilde.shape,dtype=int)[0]
      W_old=np.random.rand(nrow,Kd)
      H_old=np.random.rand(Kd,L) 
      for Lambdainfo_position in range(len(Lambdainfo)):
        for Alpha_position in range(len(Alpha)):
          for Lratio_position in range(len(Lratio)):
            lambdainfo=Lambdainfo[Lambdainfo_position] 
            alpha=Alpha[Alpha_position]
            lratio=Lratio[Lratio_position]           
            D_tilde,A,n_iter=NMF_penalized_PG(Tensor_train,W_old,H_old,lambdainfo,alpha,lratio,Kd,nbclasses,array_of_examples_numbers)
            D=D_tilde[0:nrows,:]
            penalty=alpha*lratio
            Training_features=Feature_extraction_process(Tensor_train,D,alpha,lratio,penalty,pool,pool_decision)                             
            Training_features=scaler.fit_transform(Training_features)
            Valid_features=Feature_extraction_process(Tensor_valid,D,alpha,lratio,penalty,pool,pool_decision)
            Valid_features=scaler.transform(Valid_features)
            for C_position in range(len(C_values)):
              for G_position in range(len(G_values)): 
               C_svm=C_values[C_position]
               Gamma=G_values[G_position]
               clf=svm.SVC(C=C_svm,gamma=Gamma,kernel='rbf')
               clf.fit(Training_features,y_train)
               y_predict=clf.predict(Valid_features)
               scores=np.array(y_predict==y_valid)
               matrix[C_position,G_position,Lambdainfo_position,Lratio_position,Alpha_position]=np.mean(scores) 
            W_old=D_tilde
            H_old=A
      max_positions=np.argwhere(matrix==np.max(matrix))
      C_selected=C_values[max_positions[0][0]]
      G_selected=G_values[max_positions[0][1]]
      Lambdainfo_selected=Lambdainfo[max_positions[0][2]]
      Lratio_selected=Lratio[max_positions[0][3]]
      Alpha_selected=Alpha[max_positions[0][4]]
      return C_selected,G_selected,Lambdainfo_selected,Lratio_selected,Alpha_selected,matrix
  

def Cross_valNMF_CD_DCase(Tensor_train,Tensor_valid,y_train,y_valid,C_values,G_values,Lambdainfo,Alpha,Lratio,Kd,nbclasses,pool,pool_decision):
      matrix=np.zeros((len(C_values),len(G_values),len(Lambdainfo),len(Lratio),len(Alpha)))
      scaler = StandardScaler() 
      array_of_examples_numbers=Determine_the_number_of_examples_per_class(y_train)
      for Lambdainfo_position in range(len(Lambdainfo)):
        for Alpha_position in range(len(Alpha)):
          for Lratio_position in range(len(Lratio)):
            lambdainfo=Lambdainfo[Lambdainfo_position] 
            alpha=Alpha[Alpha_position]
            lratio=Lratio[Lratio_position]            
            D=NMF_penalized_CD(Tensor_train,lambdainfo,alpha,lratio,Kd,nbclasses,array_of_examples_numbers)
            penalty=alpha*lratio
            Training_features=Feature_extraction_process(Tensor_train,D,alpha,lratio,penalty,pool,pool_decision)                             
            Training_features=scaler.fit_transform(Training_features)
            Valid_features=Feature_extraction_process(Tensor_valid,D,alpha,lratio,penalty,pool,pool_decision)
            Valid_features=scaler.transform(Valid_features)
            for C_position in range(len(C_values)):
              for G_position in range(len(G_values)): 
               C_svm=C_values[C_position]
               Gamma=G_values[G_position]
               clf=svm.SVC(C=C_svm,gamma=Gamma,kernel='rbf')
               clf.fit(Training_features,y_train)
               y_predict=clf.predict(Valid_features)
               scores=np.array(y_predict==y_valid)
               matrix[C_position,G_position,Lambdainfo_position,Lratio_position,Alpha_position]=np.mean(scores)    

      max_positions=np.argwhere(matrix==np.max(matrix))
      C_selected=C_values[max_positions[0][0]]
      G_selected=G_values[max_positions[0][1]]
      Lambdainfo_selected=Lambdainfo[max_positions[0][2]]
      Lratio_selected=Lratio[max_positions[0][3]]
      Alpha_selected=Alpha[max_positions[0][4]]
      return C_selected,G_selected,Lambdainfo_selected,Lratio_selected,Alpha_selected,matrix

def Cross_valTSPenDCase(Tensor_train,y_train,Tensor_valid,y_valid,K,seed,C_values,G_values,Lambda_info,Alpha,Lratio,T,nbclasses,kf,kt,maximum_iteration,max_iter_update,tol,InitStrat,parallel_decision,pool):        
    
        matrix=np.zeros((len(C_values),len(G_values),len(Lambda_info),len(Alpha),len(Lratio)))
        
        scaler = StandardScaler() 
        for param_pos_info in range(len(Lambda_info)):             
          for Alpha_pos in range(len(Alpha)):
            for Lratio_pos in range(len(Lratio)):
              alpha=Alpha[Alpha_pos]
              lratio=Lratio[Lratio_pos]
              lambdaG=lratio*alpha
              lambdaAf=(1-lratio)*alpha
              lambdaAt=(1-lratio)*alpha
              lambdaBf=(1-lratio)*alpha
              lambdaBt=(1-lratio)*alpha
              lambdainfo=Lambda_info[param_pos_info]                   
                                                               
              G,A_f,A_t,B_f,B_t,error_list,nb_iter=PenalizedTuckerTSPen(Tensor_train,y_train,nbclasses,kf,kt,maximum_iteration,max_iter_update,tol,InitStrat,lambdainfo,lambdaG,lambdaAf,lambdaAt,lambdaBf,lambdaBt,T,parallel_decision,pool)                                         
              penalty=lambdaG                      
              Training_features=decomposition_all_examples_parallelized(Tensor_train,np.kron(A_f,A_t),penalty,pool)                       
              Training_features=scaler.fit_transform(Training_features)                    
              Valid_features=decomposition_all_examples_parallelized(Tensor_valid,np.kron(A_f,A_t),penalty,pool)                      
              Valid_features=scaler.transform(Valid_features)                                            
              for C_position in range(len(C_values)):
                 for G_position in range(len(G_values)): 
                                               
                           C_svm=C_values[C_position]
                           Gamma=G_values[G_position]
                           clf=svm.SVC(C=C_svm,gamma=Gamma,kernel='rbf')
                           clf.fit(Training_features,y_train)
                           y_predict=clf.predict(Valid_features)
                           scores=np.mean(y_predict==y_valid)
                           matrix[C_position,G_position,param_pos_info,Alpha_pos,Lratio_pos]=np.mean(scores)
                           
        max_positions=np.argwhere(matrix==np.max(matrix))
        C_selected=C_values[max_positions[0][0]]
        G_selected=G_values[max_positions[0][1]]
        Lambda_info_selected=Lambda_info[max_positions[0][2]]
        Alpha_selected=Alpha[max_positions[0][3]]
        Lratio_selected=Lratio[max_positions[0][4]]
        return C_selected,G_selected,Lambda_info_selected,Alpha_selected,Lratio_selected,matrix

def Cross_valTSPenDCaseLinear(Tensor_train,y_train,Tensor_valid,y_valid,K,seed,C_values,G_values,Lambda_info,Alpha,Lratio,T,nbclasses,kf,kt,maximum_iteration,max_iter_update,tol,InitStrat,parallel_decision,pool):        
    
        matrix=np.zeros((len(C_values),len(G_values),len(Lambda_info),len(Alpha),len(Lratio)))
       
        scaler = StandardScaler() 
        for param_pos_info in range(len(Lambda_info)):             
          for Alpha_pos in range(len(Alpha)):
            for Lratio_pos in range(len(Lratio)):
              alpha=Alpha[Alpha_pos]
              lratio=Lratio[Lratio_pos]
              lambdaG=lratio*alpha
              lambdaAf=(1-lratio)*alpha
              lambdaAt=(1-lratio)*alpha
              lambdaBf=(1-lratio)*alpha
              lambdaBt=(1-lratio)*alpha
              lambdainfo=Lambda_info[param_pos_info]                   
              G,A_f,A_t,B_f,B_t,error_list,nb_iter=PenalizedTuckerTSPen(Tensor_train,y_train,nbclasses,kf,kt,maximum_iteration,max_iter_update,tol,InitStrat,lambdainfo,lambdaG,lambdaAf,lambdaAt,lambdaBf,lambdaBt,T,parallel_decision,pool)                                         
              penalty=lambdaG                      
              Training_features=decomposition_all_examples_parallelized(Tensor_train,np.kron(A_f,A_t),penalty,pool)                       
              Training_features=scaler.fit_transform(Training_features)                    
              Valid_features=decomposition_all_examples_parallelized(Tensor_valid,np.kron(A_f,A_t),penalty,pool)                      
              Valid_features=scaler.transform(Valid_features)                                            
              for C_position in range(len(C_values)):
                 for G_position in range(len(G_values)):                                                  
                           C_svm=C_values[C_position]
                           Gamma=G_values[G_position]
                           clf=LinearSVC(C=C_svm)
                           clf.fit(Training_features,y_train)
                           y_predict=clf.predict(Valid_features)
                           scores=np.mean(y_predict==y_valid)
                           matrix[C_position,G_position,param_pos_info,Alpha_pos,Lratio_pos]=np.mean(scores)
                           
        max_positions=np.argwhere(matrix==np.max(matrix))
        C_selected=C_values[max_positions[0][0]]
        G_selected=G_values[max_positions[0][1]]
        Lambda_info_selected=Lambda_info[max_positions[0][2]]
        Alpha_selected=Alpha[max_positions[0][3]]
        Lratio_selected=Lratio[max_positions[0][4]]
        return C_selected,G_selected,Lambda_info_selected,Alpha_selected,Lratio_selected,matrix


def Cross_valNMF_PG(Tensor,labels,C_values,G_values,Lambdainfo,Alpha,Lratio,Kd,nbclasses,pool,pool_decision):
    array_of_examples_numbers=Determine_the_number_of_examples_per_class(labels)
    matrix=np.zeros((len(C_values),len(G_values),len(Lambdainfo),len(Lratio),len(Alpha)))
    scaler = StandardScaler()      
    for Lambdainfo_position in range(len(Lambdainfo)):
     for Alpha_position in range(len(Alpha)):
       for Lratio_position in range(len(Lratio)):
         lambdainfo=Lambdainfo[Lambdainfo_position] 
         alpha=Alpha[Alpha_position]
         lratio=Lratio[Lratio_position]
         D=NMF_penalized_PG(Tensor,lambdainfo,alpha,lratio,Kd,nbclasses,array_of_examples_numbers)
         features=Feature_extraction_process(Tensor,D,alpha,lratio,pool,pool_decision)
         features=scaler.fit_transform(features)
         for C_position in range(len(C_values)):
           for G_position in range(len(G_values)): 
             C_svm=C_values[C_position]
             Gamma=G_values[G_position]
             clf=svm.SVC(C=C_svm,gamma=Gamma,kernel='rbf')
             scores = cross_val_score(clf,features,labels, cv=2)
             matrix[C_position,G_position,Lambdainfo_position,Lratio_position,Alpha_position]=np.mean(scores)    
    max_positions=np.argwhere(matrix==np.max(matrix))
    C_selected=C_values[max_positions[0][0]]
    G_selected=G_values[max_positions[0][1]]
    Lambdainfo_selected=Lambdainfo[max_positions[0][2]]
    Lratio_selected=Lratio[max_positions[0][3]]
    Alpha_selected=Alpha[max_positions[0][4]]
    return C_selected,G_selected,Lambdainfo_selected,Lratio_selected,Alpha_selected,matrix

def Features_extractionPenalizedNMF(Tensor,D,pool): #The pooling is performed before yieldind the result
    
    [K,nrows,ncols]=np.array(np.shape(Tensor),dtype=int)
    
    Kd=np.array(D.shape,dtype=int)[1]
    
    Feature=np.zeros((K,Kd))
    
    for k in range(K):
        
        Feature[k,:]=np.mean(tempEATSPenTestAB.NMF_decoupling_parallelized(D,Tensor[k,:,:],pool),axis=1)
    
    
    return Feature


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

def Test_features_extraction(Testing_tensor,A_f,A_t):
    size=np.array(Testing_tensor.shape,dtype=int)
    number_testing_features=size[0]
    sizef=np.array(A_f.shape,dtype=int)
    sizet=np.array(A_t.shape,dtype=int)
    result=np.zeros((number_testing_features,sizef[1]*sizet[1]))
    beta=np.kron(A_f,A_t)
    for k in range(number_testing_features):        
        #feature=optimize.nnls(beta,np.resize(Testing_tensor[k,:,:],np.size(Testing_tensor[k,:,:])))[0]
        #result[k,:]=feature
        feature=lsq_linear(beta,np.resize(Testing_tensor[k,:,:],np.size(Testing_tensor[k,:,:])),bounds=(0,math.inf))
        result[k,:]=feature.x
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

#This function is used to perform the supervised decomposition.
#We forbid the overcomplete case because the initialization is performed by taking positive part of factors inferred by the HOSVD algorithm and it is well known HOSVD can not be handled in the overcomplete case
#So, it is straighforward to lift this restriction by adopting an initialization strategy that does not hinder the overcomplete frame.
def PenalizedTucker(Tensor_train,y_train,nbclasses,kf,kt,hyperparam,maximum_iteration,tol,InitStrat):     
    nbinstances=np.shape(Tensor_train)[0]
    size=np.array(Tensor_train.shape,dtype=int)
    Jf=kf*nbclasses  
    Jt=kt*nbclasses
    #if ( (Jf>size[1]) or (Jt>size[2])):
        #raise AssertionError("The number of temporal or spectral dictionaries must not exceed the correspond dimension of the tensor to decompose")
    #if((Jf<=size[1]) and (Jt<=size[2])):
    Infotensor=Informationtensor(y_train,Jf,Jt,nbclasses,nbinstances)
    Amplitude=np.max(np.array(Tensor_train))
    Infotensor=Amplitude*Infotensor
    Coretensorsize=np.array([size[0],Jf,Jt],dtype=int) 
    max_iter=maximum_iteration
    epsilon=tol
    list_of_factors,G,error_list,nb_iter=tempEATSPenTestAB.NTD_ALS_decoupledversion(Tensor_train,Coretensorsize,max_iter,3,0,epsilon,hyperparam,Infotensor,InitStrat)
    A_f=list_of_factors[1]
    A_t=list_of_factors[2]
    return G,A_f,A_t,error_list,nb_iter


def PenalizedTuckerTSPen(Tensor_train,y_train,nbclasses,kf,kt,maximum_iteration,max_iter_update,tol,InitStrat,lambdainfo,lambdaG,lambdaAf,lambdaAt,lambdaBf,lambdaBt,T,parallel_decision,pool):     
    nbinstances=np.shape(Tensor_train)[0]
    size=np.array(Tensor_train.shape,dtype=int)
    Jf=kf*nbclasses  
    Jt=kt*nbclasses
    Infotensor=Informationtensor(y_train,Jf,Jt,nbclasses,nbinstances)
    Amplitude=np.max(np.array(Tensor_train))
    Infotensor=Amplitude*Infotensor
    Coretensorsize=np.array([size[0],Jf,Jt],dtype=int) 
    max_iter=maximum_iteration
    epsilon=tol
    list_of_factors,G,error_list,nb_iter=tempEATSPen.NTD_ALS_decoupledversionTSPen(Tensor_train,Coretensorsize,max_iter,max_iter_update,3,0,epsilon,lambdainfo,lambdaG,lambdaAf,lambdaAt,lambdaBf,lambdaBt,T,Infotensor,InitStrat,parallel_decision,pool)
    A_f=list_of_factors[1]
    A_t=list_of_factors[2]
    B_f=list_of_factors[4]
    B_t=list_of_factors[5]
    return G,A_f,A_t,B_f,B_t,error_list,nb_iter


#This function is used to perform the cross validation task by utilizing:
    #The parameter C
    #The parameter Sigma
    #The trade-off parameter Lambda
def Cross_val(Tensor,labels,C_values,G_values,Hyper_values,nbclasses,kf,kt,maximum_iteration,tol,InitStrat):
   matrix=np.zeros((len(C_values),len(G_values),len(Hyper_values)))
   scaler = StandardScaler()
   for param_position in range(len(Hyper_values)):
       hyperparam=Hyper_values[param_position]
       G,A_f,A_t=PenalizedTucker(Tensor,labels,nbclasses,kf,kt,hyperparam,maximum_iteration,tol,InitStrat)[0:3]
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


def Cross_valTSPen(Tensor,labels,C_values,G_values,Lambda_info,LambdaG,Lambda_Af,Lambda_At,Lambda_Bf,Lambda_Bt,T,nbclasses,kf,kt,maximum_iteration,max_iter_update,tol,InitStrat,parallel_decision,pool):
   matrix=np.zeros((len(C_values),len(G_values),len(Lambda_info),len(LambdaG),len(Lambda_Af),len(Lambda_At),len(Lambda_Bf),len(Lambda_Bt)))
   scaler = StandardScaler()
   for param_pos_info in range(len(Lambda_info)):
    for param_pos_G in range(len(LambdaG)):
     for param_pos_Af in range(len(Lambda_Af)):
         for param_pos_At in range(len(Lambda_At)):
             for param_pos_Bf in range(len(Lambda_Bf)):
                 for param_pos_Bt in range(len(Lambda_Bt)):             
                    lambdainfo=Lambda_info[param_pos_info]
                    lambdaG=LambdaG[param_pos_G]
                    lambdaAf=Lambda_Af[param_pos_Af]
                    lambdaAt=Lambda_At[param_pos_At]
                    lambdaBf=Lambda_Bf[param_pos_Bf]
                    lambdaBt=Lambda_Bt[param_pos_Bt]                    
                    G,A_f,A_t,B_f,B_t,error_list,nb_iter=PenalizedTuckerTSPen(Tensor,labels,nbclasses,kf,kt,maximum_iteration,max_iter_update,tol,InitStrat,lambdainfo,lambdaG,lambdaAf,lambdaAt,lambdaBf,lambdaBt,T,parallel_decision,pool)[0:3]                    
                    features=Test_features_extraction(Tensor,A_f,A_t)
                    features=scaler.fit_transform(features)
                    for C_position in range(len(C_values)):
                      for G_position in range(len(G_values)): 
                         C_svm=C_values[C_position]
                         Gamma=G_values[G_position]
                         clf=svm.SVC(C=C_svm,gamma=Gamma,kernel='rbf')
                         scores = cross_val_score(clf,features,labels, cv=2)
                         matrix[C_position,G_position,param_pos_info,param_pos_Af,param_pos_At,param_pos_Bf,param_pos_Bf]=np.mean(scores)
                    max_positions=np.argwhere(matrix==np.max(matrix))
   C_selected=C_values[max_positions[0][0]]
   G_selected=G_values[max_positions[0][1]]
   Lambda_info_selected=Lambda_info[max_positions[0][2]]
   Lambda_G_selected=LambdaG[max_positions[0][3]]   
   Lambda_Af_selected=Lambda_Af[max_positions[0][4]]
   Lambda_At_selected=Lambda_At[max_positions[0][5]]
   Lambda_Bf_selected=Lambda_Af[max_positions[0][6]]
   Lambda_Bt_selected=Lambda_At[max_positions[0][7]]   
   return C_selected,G_selected,Lambda_info_selected,Lambda_G_selected,Lambda_Af_selected,Lambda_At_selected,Lambda_Bf_selected,Lambda_Bt_selected,matrix



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
 