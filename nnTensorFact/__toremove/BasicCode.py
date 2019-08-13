#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 14:12:08 2017

@author: Traoreabraham
"""
#This file allows to understand how to perform easily a classification task with the defined framework.
#The commentaries above the functions should be read in order to have an overview about how they work and what they intend to do.
import numpy as np

import Methods

import pdb

import Preprocessing

from sklearn import svm

import logging

logging.basicConfig(filename='test_log.log',level=logging.DEBUG,format='%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s')

#A complete example of a toy classification problem

#We define the matrix data: 100 examples split into 10 classes whose rows correspond to the signal vectors
Data=np.random.rand(10000,500)
labels=np.zeros(500)
for i in range(10):
    labels[50*i:50*(i+1)]=(i+1)*np.ones(50)

sampling_rate=500
window_length=50
overlap_points=5
grid_form="regular" #can also be "logscale"
#We compute the TFR of all the examples.
#The TFR can be computed with regular grid or with logscale grid.
   #If the parameter grid_form is "regular", the TFR is computed on regular grid.
   #If the parameter grid_form is "logscale", the TFR is computed on logarithmic scale.
TensorTFR=Methods.TFR_all_examples(Data,sampling_rate,window_length,overlap_points,grid_form)

#We split the global tensor into training, test and validation tensors
#The ratio are considered on the total nimber of examples
#seed1,seed2 intervene in the splitting tool scikit to ensure reproductibility
ratio_test=0.20 #The number of test exmaples represents 20% of the total(500)
ratio_valid=0.10 #The number of validation exmaples represents 10% of the total(500)
seed1=5
seed2=5
Tensor_train,y_train,Tensor_test,y_test,Tensor_valid,y_valid=Methods.train_test_validation_tensors(TensorTFR,labels,ratio_test,ratio_valid,seed1,seed2)
#We perform the Cross_validation to determine the best values for the hyperparameters
#The definition of kf,kt amounts to the definition Jf and Jt because in our framework, Jf=nbclasses*kf. The same definition holds for Jt
C_values=np.power(10,np.linspace(-2,2,5),dtype=float)
G_values=np.power(10,np.linspace(-1,1,3),dtype=float)
Hyper_values=np.power(10,np.linspace(-1,1,3),dtype=float)
nbclasses=10
kt=2
kf=2
tol=np.power(10,-3,dtype=float)
maximum_iterations=10
C_svm,G_svm,hyperparam=Methods.Cross_val(Tensor_valid,y_valid,C_values,G_values,Hyper_values,nbclasses,kf,kt,maximum_iterations,tol)
#We perform the decomposition of the training tensor
#The decomposition yields:
    #The error related to each updating error_list;
    #The temporal and spectral dictionary components A_f and A_t;
    #The number of iterations and the activation coefficients G;
    #We dimension purpose, we can reduce the size of the tensor to be decomposed.This can be done in the following way:
        #Tensor_train=dtensor(Preprocessing.rescaling(Tensor_train,If,It)) where If and It correspond to the desired sizes.
G,A_f,A_t,error_list,nb_iter=Methods.PenalizedTucker(Tensor_train,y_train,nbclasses,kf,kt,hyperparam,maximum_iterations,tol)
pdb.set_trace()
#We define the training features. They are obtained by vectorizing the matrices G[k,:,:]
Training_features=Methods.Training_features_extraction(G)
#We define the test features
#If we resize the training tensor, it is obligatory to resize the test tensor for dimensionality coherence
    #This is done via the rescaling function defined in Preprocessing
Test_features=Methods.Test_features_extraction(Tensor_test,A_f,A_t)
#We define the classifier and perform the classification task
clf=svm.SVC(C=C_svm,gamma=G_svm,kernel='rbf')
clf.fit(Training_features,y_train)
#Recover_classes_all_labels is designed to recover the classes present in an array with the help of the labels. 
#There are three outputs:
    #result corresponds to the examples followed by their classes label
    #indexes contains the classes present in the array
    #number_of_samples_per_class yields the number of examples per class.
result,indexes,number_of_samples_per_class=Preprocessing.Recover_classes_all_labels(Test_features,y_test)
#The function classificaton_rate_classes is designed in order to determine the performances per classe in order to compute the MAP
#There are two outputs:
    #performances represent the proportion of well classified examples per class.
    #confusion_matrix represents the prediction of the classifier for each example in each class. It is usefull to understand which example has not been well classified
performances,confusionmatrix=Methods.classificaton_rate_classes(result,clf)
print(np.mean(performances))
pdb.set_trace()