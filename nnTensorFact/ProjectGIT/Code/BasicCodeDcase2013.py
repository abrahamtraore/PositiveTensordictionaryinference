#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:10:43 2017

@author: Traoreabraham
"""

import h5py
from multiprocessing import Pool
from sklearn.model_selection import StratifiedKFold
import sys
import pdb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import Methods
sys.path.append("/home/scr/etu/sin811/traorabr/sktensor/")
from sktensor import dtensor


#filename="/Users/Traoreabraham/Desktop/ProjectGit/DCase2013/dcase13-cqtrain.h5"
filename="/home/scr/etu/sin811/traorabr/DCase2013/dcase13-cqtrain.h5"
with h5py.File(filename,'r') as hf:
        data = hf.get('x')
        Tensor_trainvalid = np.array(data)
        data = hf.get('y')
        y_trainvalid = np.array(data)
        y_trainvalid=y_trainvalid+1
        data = hf.get('set')
        set = np.array(data)
           
#filename="/Users/Traoreabraham/Desktop/ProjectGit/DCase2013/dcase13-cqtest.h5"
filename="/home/scr/etu/sin811/traorabr/DCase2013/dcase13-cqtest.h5"
with h5py.File(filename,'r') as hf:
        data = hf.get('x')
        Tensor_test = np.array(data)
        data = hf.get('y')
        y_test = np.array(data)
        y_test=y_test+1
        data = hf.get('set')
        set = np.array(data)


If=70

It=60

Tensor_trainvalid=dtensor(Methods.rescaling(Tensor_trainvalid,If,It))

Tensor_test=dtensor(Methods.rescaling(Tensor_test,If,It))

[firstsize,secondsize]=np.array(Tensor_test.shape,dtype=int)[1:3]

vector_all_features=Methods.Transform_tensor_into_featuresmatrix(Tensor_trainvalid)

K=5

seed=43

skf=StratifiedKFold(n_splits=K,random_state=seed,shuffle=True)#skf=StratifiedKFold(n_splits=5,random_state=17,shuffle=True)

Perf=np.zeros(K)   

k=-1

print(Tensor_trainvalid.shape)



for train_index, valid_index in skf.split(vector_all_features,y_trainvalid):
    
     matrix_train, matrix_valid = vector_all_features[train_index],vector_all_features[valid_index]
    
     y_train, y_valid=y_trainvalid[train_index],y_trainvalid[valid_index]
    
     Tensor_train=Methods.Transform_featuresmatrix_into_tensor(matrix_train,firstsize,secondsize)
    
     Tensor_valid=Methods.Transform_featuresmatrix_into_tensor(matrix_valid,firstsize,secondsize) 
     #We perform the Cross_validation to determine the best values for the hyperparameters
     #The definition of kf,kt amounts to the definition Jf and Jt because in our framework, Jf=nbclasses*kf. The same definition holds for Jt
     C_values=np.power(10,np.linspace(-3,7,6),dtype=float)
     G_values=np.power(10,np.linspace(-3,4,4),dtype=float)
     Hyper_values=np.power(10,np.linspace(-6,1,8),dtype=float)
     nbclasses=10
     kt=3
     kf=3
     tol=np.power(10,-10,dtype=float)
     maximum_iterations=10
     parallel_decision=True
     pool=Pool(5)
     InitStrat="Multiplicative"      #"Tucker2HOSVD"   #"Multiplicative"
    
     Tensor_train=Methods.Transform_featuresmatrix_into_tensor(matrix_train,firstsize,secondsize)
    
     Tensor_valid=Methods.Transform_featuresmatrix_into_tensor(matrix_valid,firstsize,secondsize) 
     
     C_svm,G_svm,hyperparam,matrixscore=Methods.Cross_val(Tensor_valid,y_valid,C_values,G_values,Hyper_values,nbclasses,kf,kt,maximum_iterations,tol,InitStrat,parallel_decision,pool)

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
         
     G,A_f,A_t,error,nbiter=Methods.PenalizedTucker(Tensor_train,y_train,nbclasses,kf,kt,hyperparam,maximum_iterations,tol,InitStrat,parallel_decision,pool)
    
     #We define the training features. They are obtained by vectorizing the matrices G[k,:,:] and normalized.
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
     print(np.mean(performances))
pdb.set_trace()