#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 22:07:36 2017

@author: Traoreabraham
"""

import numpy as np

from multiprocessing import Pool

#from Librosa import librosa

import MethodsTSPen

import h5py

import pdb

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold

from sklearn import svm

np.random.seed(1)

import sys

sys.path.append("/home/scr/etu/sin811/traorabr/sktensor/")

from sktensor import dtensor


BASE_PATH="/home/scr/etu/sin811/traorabr/RealToy/Realtoy.h5"
with h5py.File(BASE_PATH,'r') as hf:
    Tensor = np.array(hf.get('Tensordata')).astype('float32')
    labels = np.array(hf.get('labels')).astype('uint32')

eta=20

width=30

length=30

TensorTFR=dtensor(MethodsTSPen.rescaling(Tensor,width,length))

TensorTFR=TensorTFR+eta*np.random.rand(400,width,length)


seed=5

K=5

skf=StratifiedKFold(n_splits=K,random_state=seed,shuffle=True)#skf=StratifiedKFold(n_splits=5,random_state=17,shuffle=True)

Perf=np.zeros(K)   

k=-1

firstsize=np.array(TensorTFR.shape,dtype=int)[1]

secondsize=np.array(TensorTFR.shape,dtype=int)[2]

vector_all_features=MethodsTSPen.Transform_tensor_into_featuresmatrix(TensorTFR)

for trainvalid_index, test_index in skf.split(vector_all_features,labels):
    
    matrix_trainvalid,matrix_test=vector_all_features[trainvalid_index],vector_all_features[test_index]
    
    y_trainvalid,y_test=labels[trainvalid_index],labels[test_index]
    
    skf1=StratifiedKFold(n_splits=3,random_state=seed,shuffle=True)
    
    for train_index, test_index in skf1.split(matrix_trainvalid,y_trainvalid):
    
         matrix_train,matrix_valid=matrix_trainvalid[train_index],matrix_trainvalid[test_index]
         
         y_train,y_valid=y_trainvalid[train_index],y_trainvalid[test_index]
         
    print("Point I")
    
    Tensor_train=MethodsTSPen.Transform_featuresmatrix_into_tensor(matrix_train,firstsize,secondsize)
    Tensor_test=MethodsTSPen.Transform_featuresmatrix_into_tensor(matrix_test,firstsize,secondsize)
    Tensor_valid=MethodsTSPen.Transform_featuresmatrix_into_tensor(matrix_valid,firstsize,secondsize)
    #We perform the Cross_validation to determine the best values for the hyperparameters
    #The definition of kf,kt amounts to the definition Jf and Jt because in our framework, Jf=nbclasses*kf. The same definition holds for Jt   
    C_values=np.power(10,np.linspace(-3,4,4),dtype=float)
    G_values=np.power(10,np.linspace(-3,4,4),dtype=float)
    Lambda_info=np.power(10,np.linspace(-4,1,3),dtype=float)
    LambdaG=np.power(10,-4,dtype=float)*np.power(10,np.linspace(-4,3,1),dtype=float)
    Lambda_Af=(1-np.power(10,-4,dtype=float))*np.power(10,np.linspace(-4,3,1),dtype=float)
    Lambda_At=(1-np.power(10,-4,dtype=float))*np.power(10,np.linspace(-4,3,1),dtype=float)
    Lambda_Bf=(1-np.power(10,-4,dtype=float))*np.power(10,np.linspace(-4,3,1),dtype=float)
    Lambda_Bt=(1-np.power(10,-4,dtype=float))*np.power(10,np.linspace(-4,3,1),dtype=float)
    T=2
    nbclasses=10
    kt=2 #2
    kf=1 #2
    
    tol=np.power(10,-8,dtype=float)
    maximum_iterations=10
    pool=Pool(3)
    InitStrat="Multiplicative"#"Tucker2HOSVD"
    parallel_decision=True
    C_svm,G_svm,lambdainfo,lambdaG,lambdaAf,lambdaAt,lambdaBf,lambdaBt,matrix=MethodsTSPen.Cross_valTSPen(Tensor_valid,y_valid,C_values,G_values,Lambda_info,LambdaG,Lambda_Af,Lambda_At,Lambda_Bf,Lambda_Bt,T,nbclasses,kf,kt,maximum_iterations,tol,InitStrat,parallel_decision,pool)
    #We perform the decomposition of the training tensor
    #The decomposition yields:
       #The error related to each updating error_list;
       #The temporal and spectral dictionary components A_f and A_t;
       #The number of iterations and the activation coefficients G;
       #We dimension purpose, we can reduce the size of the tensor to be decomposed.This can be done in the following way:
          #Tensor_train=dtensor(Preprocessing.rescaling(Tensor_train,If,It)) where If and It correspond to the desired sizes.



     #We perform the decomposition of the training tensor
     #The decomposition yields:
       #The error related to each updating error_list;
       #The temporal and spectral dictionary components A_f and A_t;
       #The number of iterations and the activation coefficients G;
       #We dimension purpose, we can reduce the size of the tensor to be decomposed.This can be done locally on the computer

    #G,A_f,A_t,error_list,nb_iter=MethodsTSPen.PenalizedTuckerGD(Tensor_train,y_train,nbclasses,kf,kt,maximum_iterations,tol,InitStrat,lambdainfo,lambdaAf,lambdaAt,lambdaBf,lambdaBt,T)
    #G,A_f,A_t,error_list,nb_iter=MethodsTSPen.PenalizedTuckerGD(Tensor_train,y_train,nbclasses,kf,kt,maximum_iterations,tol,InitStrat,lambdainfo,lambdaG,lambdaAf,lambdaAt,lambdaBf,lambdaBt,T,pool)
    G,A_f,A_t,error_list,nb_iter=MethodsTSPen.PenalizedTuckerTSPen(Tensor_train,y_train,nbclasses,kf,kt,maximum_iterations,tol,InitStrat,lambdainfo,lambdaG,lambdaAf,lambdaAt,lambdaBf,lambdaBt,T,parallel_decision,pool)
    
    #We define the training features. They are obtained by vectorizing the matrices G[k,:,:] and normalized.
    
    Training_features=MethodsTSPen.Test_features_extraction(Tensor_train,A_f,A_t)
    

    scaler=StandardScaler()

    Training_features=scaler.fit_transform(Training_features)

    #We define the test features and normalize them
    Test_features=MethodsTSPen.Test_features_extraction(Tensor_test,A_f,A_t)

    Test_features=scaler.transform(Test_features)

    #We define the classifier and perform the classification task
    clf=svm.SVC(C=C_svm,gamma=G_svm,kernel='rbf')
    clf.fit(Training_features,y_train)
    #Recover_classes_all_labels is designed to recover the classes present in an array with the help of the labels. 
    #There are three outputs:
      #result corresponds to the examples followed by their classes label
      #indexes contains the classes present in the array
      #number_of_samples_per_class yields the number of examples per class.
    result,indexes,number_of_samples_per_class=MethodsTSPen.Recover_classes_all_labels(Test_features,y_test)
    #The function classificaton_rate_classes is designed in order to determine the performances per classe in order to compute the MAP
    #There are two outputs:
      #performances represent the proportion of well classified examples per class.
      #confusion_matrix represents the prediction of the classifier for each example in each class. It is usefull to understand which example has not been well classified
    performances,confusionmatrix=MethodsTSPen.classificaton_rate_classes(result,clf)
    print(np.mean(performances))
    k=k+1
    Perf[k]=np.mean(performances)
    
pdb.set_trace()
    
    
    