#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 21:57:34 2017

@author: Traoreabraham
"""
import numpy as np

from multiprocessing import Pool 

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn import svm

import Methods

#from dtensor import dtensor

import pdb
    
BASE_PATH="/home/scr/etu/sin811/traorabr/RawDataEastAnglia"
#This tensor contains the preprocessed data.
TensorTFR,labels=Methods.load_preprocessed_data(BASE_PATH)
print("The adress is the good one")
#We split the global tensor into training, test and validation tensors
#The ratio are considered on the total nimber of examples
#seed1,seed2 intervene in the splitting tool scikit to ensure reproductibility


seed=2

[firstsize,secondsize]=np.array(TensorTFR.shape,dtype=int)[1:3]

vector_all_features=Methods.Transform_tensor_into_featuresmatrix(TensorTFR)  
   
   
seed=5

K=5

skf=StratifiedKFold(n_splits=K,random_state=seed,shuffle=True)#skf=StratifiedKFold(n_splits=5,random_state=17,shuffle=True)

Perf=np.zeros(K)   

k=-1

firstsize=np.array(TensorTFR.shape,dtype=int)[1]

secondsize=np.array(TensorTFR.shape,dtype=int)[2]

vector_all_features=Methods.Transform_tensor_into_featuresmatrix(TensorTFR)

for train_index, test_index in skf.split(vector_all_features,labels):
    
    matrix_train, matrix_test = vector_all_features[train_index],vector_all_features[test_index]
    
    y_train, y_test=labels[train_index],labels[test_index]
    
    Tensor_train=Methods.Transform_featuresmatrix_into_tensor(matrix_train,firstsize,secondsize)
    
    Tensor_test=Methods.Transform_featuresmatrix_into_tensor(matrix_test,firstsize,secondsize) 
    #Tensor_train,y_train,Tensor_test,y_test,Tensor_valid,y_valid=Methods.train_test_validation_tensors(TensorTFR,labels,ratio_test,ratio_valid,seed1,seed2)

    #We perform the Cross_validation to determine the best values for the hyperparameters
    #The definition of kf,kt amounts to the definition Jf and Jt because in our framework, Jf=nbclasses*kf. The same definition holds for Jt
    #C_values=np.power(10,np.linspace(-2,2,2),dtype=float)
    #G_values=np.power(10,np.linspace(-1,1,2),dtype=float)
    #Hyper_values=np.power(10,np.linspace(-1,1,2),dtype=float)
    nbclasses=10
    #window_length=200
    #overlap_points=185
    #kt=2 
    #kf=2
    #perf=69.04%
    tol=np.power(10,-6,dtype=float)
    
    maximum_iterations=10

    C_values=np.power(10,np.linspace(-5,4,5),dtype=float)
    #np.power(10,np.linspace(-5,5,10),dtype=float)
    G_values=np.power(10,np.linspace(-4,3,4),dtype=float)
    #np.power(10,np.linspace(-4,4,8),dtype=float)
    Hyper_values=np.power(10,np.linspace(-10,-1,5))    

    Kd=25
    
    pool=Pool(3)
    
    C_svm,G_svm,hyperparam,matrix=Methods.Cross_valNMFPenalized(Tensor_train,y_train,C_values,G_values,Hyper_values,Kd,nbclasses,pool)
    

    #We perform the decomposition of the training tensor
    #The decomposition yields:
      #The error related to each updating error_list;
      #The temporal and spectral dictionary components A_f and A_t;
      #The number of iterations and the activation coefficients G;
    #We dimension purpose, we can reduce the size of the tensor to be decomposed.This can be done in the following way:
        #Tensor_train=dtensor(Preprocessing.rescaling(Tensor_train,If,It)) where If and It correspond to the desired sizes.
    array_of_examples_numbers=Methods.Determine_the_number_of_examples_per_class(y_train)
    D=Methods.NMF_penalized(Tensor_train,hyperparam,Kd,nbclasses,array_of_examples_numbers)
    
    
    
    #We define the training features. They are obtained by vectorizing the matrices G[k,:,:]
    #Training_features=Methods.Training_features_extraction(G)
    #Training_features=Methods.Training_features_extraction(G)

    Training_features=Methods.Features_extractionPenalizedNMF(Tensor_train,D,pool)
    scaler=StandardScaler()
    Training_features=scaler.fit_transform(Training_features)

    #C_values=np.power(10,np.linspace(-2,5,4),dtype=float)
    #G_values=np.power(10,np.linspace(-2,2,5),dtype=float)


    #We define the test features
     #If we resize the training tensor, it is obligatory to resize the test tensor for dimensionality coherence
     #This is done via the rescaling function defined in Preprocessing
    Test_features=Methods.Features_extractionPenalizedNMF(Tensor_test,D,pool)
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
print(Perf)
pdb.set_trace()