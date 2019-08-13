#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 09:57:42 2017

@author: Traoreabraham
"""


import h5py
import pdb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import Methods
import sys
sys.path.append("/home/scr/etu/sin811/traorabr/sktensor/")
from sktensor import dtensor
#filename="/Users/Traoreabraham/Desktop/DCaseData/dcase16-cqt.h5"
BASE_PATH='/home/scr/etu/sin811/traorabr/RawDCase/dcase16-cqtrain.h5'

with h5py.File(BASE_PATH,'r') as hf:
        data = hf.get('x')
        x = np.array(data)    
        data = hf.get('y')
        y = np.array(data) 
        data = hf.get('set')
        set = np.array(data)
        
##filename="/Users/Traoreabraham/Desktop/DCaseData/dcase16-cqt.h5.test"
BASE_PATH='/home/scr/etu/sin811/traorabr/RawDCase/dcase16-cqtest.h5'
with h5py.File(BASE_PATH,'r') as hf:
        data=hf.get('X_test')
        Tensor_test=dtensor(np.array(data))
        data=hf.get('y_test')
        y_test=np.array(data,dtype=int)
        y_test=y_test+1
print("Spot I")

scaler=StandardScaler()

folds=np.array([0,1,2,3],dtype=int)

Perf=np.zeros(4)

for numfold in folds:

   indlearn = np.where(set[:,numfold]==1)

   indval = np.where(set[:,numfold]==2)

   Tensor_train = x[indlearn,:,:]

   [img_rows,img_cols]=np.array(np.shape(Tensor_train),dtype=int)[2:4]

   Tensor_train = dtensor(Tensor_train[0,0:Tensor_train.shape[1],0:img_rows,0:img_cols])

   Tensor_val = x[indval,:,:]

   Tensor_val = dtensor(Tensor_val[0,0:Tensor_val.shape[1],0:img_rows,0:img_cols])

   y_train=y[indlearn][:,numfold]

   y_train=y_train+1

   y_train=np.array(y_train,dtype=int)

   y_val=y[indval][:,numfold]

   y_val=y_val+1

   y_val=np.array(y_val,dtype=int)


   #These lines yield the possibility to rescal the TFR images
   If=60
   It=60
   Tensor_train=Methods.rescaling(Tensor_train,If,It)
   Tensor_train=dtensor(Tensor_train)
   Tensor_val=Methods.rescaling(Tensor_val,If,It)
   Tensor_val=dtensor(Tensor_val)
   Tensor_test=Methods.rescaling(Tensor_test,If,It)
   Tensor_test=dtensor(Tensor_test)
   #We perform the Cross_validation to determine the best values for the hyperparameters
   #The definition of kf,kt amounts to the definition Jf and Jt because in our framework, Jf=nbclasses*kf. The same definition holds for Jt
   #C_values=np.power(10,np.linspace(-2,2,2),dtype=float)
   #G_values=np.power(10,np.linspace(-1,1,2),dtype=float)
   #Hyper_values=np.power(10,np.linspace(-1,1,2),dtype=float)
   nbclasses=15
   kt=3
   kf=3
   hyperparam=np.power(10,-3,dtype=float)
   tol=np.power(10,-3,dtype=float)
   maximum_iterations=10
   
   C_values=np.power(10,np.linspace(-4,4,9),dtype=float)
   G_values=np.power(10,np.linspace(-2,2,4),dtype=float)
   Hyper_values=np.power(10,np.linspace(-10,-1,10),dtype=float)
   C_svm,G_svm,hyperparam,matrixscore=Methods.Cross_valHALs(Tensor_val,y_val,C_values,G_values,Hyper_values,nbclasses,kf,kt,maximum_iterations,tol)
   #We perform the decomposition of the training tensor
   #The decomposition yields:
     #The error related to each updating error_list;
     #The temporal and spectral dictionary components A_f and A_t;
     #The number of iterations and the activation coefficients G;
     #We dimension purpose, we can reduce the size of the tensor to be decomposed.This can be done in the following way:
        #Tensor_train=dtensor(Preprocessing.rescaling(Tensor_train,If,It)) where If and It correspond to the desired sizes.
   G,A_f,A_t,error_list=Methods.PenalizedTuckerHals(Tensor_train,y_train,nbclasses,kf,kt,hyperparam,maximum_iterations,tol)
   print("Spot II")
   #We define the training features. They are obtained by vectorizing the matrices G[k,:,:]
   #Training_features=Methods.Training_features_extraction(G)

   Training_features=Methods.Training_features_extraction(G)
   
   Training_features=scaler.fit_transform(Training_features)
   print("Spot III")
   #We define the test features
   #If we resize the training tensor, it is obligatory to resize the test tensor for dimensionality coherence
     #This is done via the rescaling function defined in Preprocessing
   Test_features=Methods.Test_features_extraction(Tensor_test,A_f,A_t)
   
   Test_features=scaler.fit_transform(Test_features)
   print("Spot IV")
   #We define the classifier and perform the classification task
   clf=svm.SVC(C=C_svm,gamma=G_svm,kernel='rbf')
   clf.fit(Training_features,y_train)
   #Recover_classes_all_labels is designed to recover the classes present in an array with the help of the labels. 
   #There are three outputs:
     #result corresponds to the examples followed by their classes label
     #indexes contains the classes present in the array
     #number_of_samples_per_class yields the number of examples per class.
   result,indexes,number_of_samples_per_class=Methods.Recover_classes_all_labels(Test_features,y_test)
   print("Spot V")
   #The function classificaton_rate_classes is designed in order to determine the performances per classe in order to compute the MAP
   #There are two outputs:
     #performances represent the proportion of well classified examples per class.
     #confusion_matrix represents the prediction of the classifier for each example in each class. It is usefull to understand which example has not been well classified
   performances,confusionmatrix=Methods.classificaton_rate_classes(result,clf)
   print("Spot VI")
   Perf[numfold]=np.mean(performances)
   print(np.mean(performances))
pdb.set_trace()