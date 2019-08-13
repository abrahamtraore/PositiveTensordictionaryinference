#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 15:48:14 2017

@author: Traoreabraham
"""

import numpy as np

from multiprocessing import Pool

import Methods

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold

import pdb

from sklearn import svm

import h5py

import logging

logging.basicConfig(filename='test_log.log',level=logging.DEBUG,format='%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s')

#A complete example of a toy classification problem

#We import the data processed locally on the computer
BASE_PATH="/home/scr/etu/sin811/traorabr/RawDataToy/toy.h5"
#with h5py.File('/Users/Traoreabraham/Desktop/ProjectGit/Data/toy.h5','r') as hf:
with h5py.File(BASE_PATH,'r') as hf:
    TensorTFR= np.array(hf.get('X')).astype('float32')
    labels= np.array(hf.get('y')).astype('uint32')    
    labels=labels+1
    labels=labels[:,0]
#sampling_rate=500
#window_length=50
#overlap_points=5
#grid_form="regular" #can also be "logscale"
##We compute the TFR of all the examples.
##The TFR can be computed with regular grid or with logscale grid.
#   #If the parameter grid_form is "regular", the TFR is computed on regular grid.
#   #If the parameter grid_form is "logscale", the TFR is computed on logarithmic scale.
#TensorTFR=Methods.TFR_all_examples(Data,sampling_rate,window_length,overlap_points,grid_form)

#We split the global tensor into training, test and validation tensors
#The ratio are considered on the total nimber of examples
#seed1,seed2 intervene in the splitting tool scikit to ensure reproductibility

seed=5

K=5

skf=StratifiedKFold(n_splits=K,random_state=seed,shuffle=True)#skf=StratifiedKFold(n_splits=5,random_state=17,shuffle=True)

Perf=np.zeros(K)   

k=-1

firstsize=np.array(TensorTFR.shape,dtype=int)[1]

secondsize=np.array(TensorTFR.shape,dtype=int)[2]

vector_all_features=Methods.Transform_tensor_into_featuresmatrix(TensorTFR)

for trainvalid_index, test_index in skf.split(vector_all_features,labels):
    
    matrix_trainvalid,matrix_test=vector_all_features[trainvalid_index],vector_all_features[test_index]
    
    y_trainvalid,y_test=labels[trainvalid_index],labels[test_index]
    
    skf1=StratifiedKFold(n_splits=3,random_state=seed,shuffle=True)
    
    for train_index, test_index in skf1.split(matrix_trainvalid,y_trainvalid):
    
         matrix_train,matrix_valid=matrix_trainvalid[train_index],matrix_trainvalid[test_index]
         
         y_train,y_valid=y_trainvalid[train_index],y_trainvalid[test_index]
         
    print("Point I")
    
    Tensor_train=Methods.Transform_featuresmatrix_into_tensor(matrix_train,firstsize,secondsize)
    Tensor_test=Methods.Transform_featuresmatrix_into_tensor(matrix_test,firstsize,secondsize)
    Tensor_valid=Methods.Transform_featuresmatrix_into_tensor(matrix_valid,firstsize,secondsize)
    #We perform the Cross_validation to determine the best values for the hyperparameters
    #The definition of kf,kt amounts to the definition Jf and Jt because in our framework, Jf=nbclasses*kf. The same definition holds for Jt
    C_values=np.power(10,np.linspace(-2,2,1),dtype=float)#5
    G_values=np.power(10,np.linspace(-1,1,1),dtype=float)#3
    Hyper_values=np.power(10,np.linspace(-1,1,1),dtype=float)#3
    nbclasses=2
    kt=20
    kf=20
    print("The size of the tensor is")
    print(Tensor_train.shape)
    print(kt*nbclasses)
    print(kf*nbclasses)
    tol=np.power(10,-3,dtype=float)
    InitStrat="Multiplicative" #The other possibility is "Tucker2HOSVD", but for this latter initialization, we are in the undercomplete case
    maximum_iterations=10
    parallel_decision=True
    pool=Pool(5)
    #C_svm,G_svm,hyperparam,matrixscore=Methods.Cross_val(Tensor_valid,y_valid,C_values,G_values,Hyper_values,nbclasses,kf,kt,maximum_iterations,tol,InitStrat)
    C_svm,G_svm,hyperparam,matrixscore=Methods.Cross_val(Tensor_train,y_train,C_values,G_values,Hyper_values,nbclasses,kf,kt,maximum_iterations,tol,InitStrat,parallel_decision,pool)

    print("Point II")
    #We perform the decomposition of the training tensor
    #The decomposition yields:
       #The error related to each updating error_list;
       #The temporal and spectral dictionary components A_f and A_t;
       #The number of iterations and the activation coefficients G;
       #We dimension purpose, we can reduce the size of the tensor to be decomposed.This can be done locally on the computer
    #G,A_f,A_t,error_list,nb_iter=Methods.PenalizedTucker(Tensor_train,y_train,nbclasses,kf,kt,hyperparam,maximum_iterations,tol,InitStrat)
    G,A_f,A_t,error,nbiter=Methods.PenalizedTucker(Tensor_train,y_train,nbclasses,kf,kt,hyperparam,maximum_iterations,tol,InitStrat,parallel_decision,pool)    
    print("Point III")
    #We define the training features. They are obtained by vectorizing the matrices G[k,:,:] and normalized.
    #Training_features=Methods.Training_features_extraction(G)
    Training_features=Methods.Test_features_extraction(Tensor_train,A_f,A_t)
    
    scaler=StandardScaler()
    
    Training_features=scaler.fit_transform(Training_features)

    #We define the test features and normalize them
    Test_features=Methods.Test_features_extraction(Tensor_test,A_f,A_t)
    Test_features=scaler.transform(Test_features)

    #We define the classifier and perform the classification task
    clf=svm.SVC(C=C_svm,gamma=G_svm,kernel='rbf')
    clf.fit(Training_features,y_train)
    #Recover_classes_all_labels is designed to recover the classes present in an array with the help of the labels. 
    #There are three outputs:
      #result corresponds to the examples followed by their classes label
      #indexes contains the classes present in the array
      #number_of_samples_per_class yields the number of examples per class.
    result,indexes,number_of_samples_per_class=Methods.Recover_classes_all_labels(Test_features,y_test)
    #The function classificaton_rate_classes is designed in order to determine the performances per classe in order to compute the MAP
    #There are two outputs:
       #performances represent the proportion of well classified examples per class.
       #confusion_matrix represents the prediction of the classifier for each example in each class. It is usefull to understand which example has not been well classified
    performances,confusionmatrix=Methods.classificaton_rate_classes(result,clf)
    k=k+1
    Perf[k]=np.mean(performances)
    print(Perf[k])
#    print("The Mean Average precision is")
#    print(np.mean(performances))
#    print("The number of examples are:")
#    print(np.array(TensorTFR.shape)[0])
#    print("The sizes of the TFR images are:")
#    print(np.array(TensorTFR.shape)[1:3])
#    print("The numbers of temporal and spectral dictionary atoms respectively are:")
#    print(np.array([kf*nbclasses,kt*nbclasses]))
print(Perf)
pdb.set_trace()