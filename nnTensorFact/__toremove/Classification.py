#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 09:19:45 2017

@author: Traoreabraham
"""
import os

import Preprocessing

from sktensor import dtensor

import numpy as np

from sklearn.model_selection import StratifiedKFold

import Tuckerdecomposition

from sklearn import svm

#The variable random_state is used to ensure reproductibility and its different values correspond to diffrent splittings of the samples
#Shuffle=True means the samples are shuffle before the splitting
#random_state=3  
#random_state=5  
#random_state=7
#random_state=154 
filename='EastAngliaDataSet'
Datafilename=os.listdir(filename) #corresponds to the data file names
res,labels=Preprocessing.split_all_the_data_and_stack(Datafilename,filename)
X=dtensor(Preprocessing.TFR_all_the_examples(res))
width=35
length=35 
Tensor_train_reshaped=dtensor(Preprocessing.rescaling(X,width,length))
firstsize=np.array(X.shape,dtype=int)[1]
secondsize=np.array(X.shape,dtype=int)[2]
vector_all_features=Preprocessing.Transform_tensor_into_featuresmatrix(X)
skf=StratifiedKFold(n_splits=5,random_state=154,shuffle=True)  
for train_index, test_index in skf.split(vector_all_features,labels):  
    matrix_train, matrix_test = vector_all_features[train_index],vector_all_features[test_index]
    y_train, y_test=labels[train_index],labels[test_index] 
Tensor_train=Preprocessing.Transform_featuresmatrix_into_tensor(matrix_train,firstsize,secondsize)
width=35
length=35 
Tensor_train_reshaped=dtensor(Preprocessing.rescaling(Tensor_train,width,length))
size=np.array(Tensor_train_reshaped.shape,dtype=int)
Coretensorsize=np.array([size[0],round(1*size[1]/4),round(1*size[2]/4)],dtype=int)
max_iter=50             
epsilon=0.00001
N=3
m=0
list_of_factors,G,error_list,nb_iter=Tuckerdecomposition.NTD_ALS_decoupledversion(Tensor_train_reshaped,Coretensorsize,max_iter,N,m,epsilon)
A_f=list_of_factors[1]
A_t=list_of_factors[2]
range_of_values_C=np.linspace(0.01,10000,300)
range_of_values_G=np.linspace(0.00001,10,10)
Training_feature=Preprocessing.Training_features_extraction(G)
C_svm,G_svm=Preprocessing.Hyperparameter_selection(Training_feature, y_train,range_of_values_C,range_of_values_G)
Tensor_test=Preprocessing.Transform_featuresmatrix_into_tensor(matrix_test,firstsize,secondsize)
Tensor_test_reshaped=dtensor(Preprocessing.rescaling(Tensor_test,width,length))
Test_features=Preprocessing.Test_features_extraction(Tensor_test_reshaped,A_f,A_t)
clf=svm.SVC(C=C_svm,gamma=G_svm,kernel='rbf')
clf.fit(Training_feature,y_train)
result,indexes ,number_of_samples_per_class=Preprocessing.Recover_classes_all_labels(Test_features,y_test)
performances,confusionmatrix=Preprocessing.classificaton_rate_classes(result,clf)
#print(np.mean(performances))

