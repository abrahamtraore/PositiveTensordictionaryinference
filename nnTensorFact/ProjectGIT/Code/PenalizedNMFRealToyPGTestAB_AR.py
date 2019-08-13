#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 18:08:34 2017

@author: Traoreabraham
"""

#81.49%

import numpy as np
from multiprocessing import Pool
import MethodsTSPenTestAB
import MethodsTSPen
import pdb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import svm
import h5py
import sys


from sktensor import dtensor

BASE_PATH="toy.h5"
with h5py.File(BASE_PATH,'r') as hf:
    Tensor = np.array(hf.get('Tensordata')).astype('float32')
    labels = np.array(hf.get('labels')).astype('uint32')  



width = 25
length = 25
nb_train = 240
Kd = 32
seed = 2
pool_decision = False
T=2
nbclasses=6

seeds=np.array([1,2,3,4,5],dtype=int)
Perf=np.zeros(5)
C_values=np.power(10,np.linspace(-3,4,4),dtype=float)
G_values=np.power(10,np.linspace(-3,4,4),dtype=float)
Lambda_info=np.power(10,np.linspace(-4,1,1),dtype=float)
LambdaG=np.power(10,-4,dtype=float)*np.power(10,np.linspace(-3,3,1),dtype=float)
Lambda_Af=(1-np.power(10,-3,dtype=float))*np.power(10,np.linspace(-4,3,1),dtype=float)
Lambda_At=(1-np.power(10,-3,dtype=float))*np.power(10,np.linspace(-4,3,1),dtype=float)
Lambda_Bf=(1-np.power(10,-3,dtype=float))*np.power(10,np.linspace(-4,3,1),dtype=float)
Lambda_Bt=(1-np.power(10,-3,dtype=float))*np.power(10,np.linspace(-4,3,1),dtype=float)
Alpha= [1, 10]
Lratio=[0.001, 0.01]



    
tol=np.power(10,-8,dtype=float)
maximum_iterations=10
max_iter_update= 100
pool=Pool(3)
InitStrat="Multiplicative"
parallel_decision=True
nb_val_split = 2
sum_matrix = np.zeros((len(C_values),len(G_values),len(Lambda_info),len(Alpha),len(Lratio)))

Tensor=dtensor(MethodsTSPenTestAB.rescaling(Tensor,width,length))
Tensor=Tensor
labels=labels
nb_data = np.array(Tensor.shape,dtype=int)[0]
firstsize = np.array(Tensor.shape,dtype=int)[1]
secondsize = np.array(Tensor.shape,dtype=int)[2]
ratio_test = 1 - nb_train/nb_data
for i in range(5):
    
    np.random.seed(seeds[i])
  
    TensorTFR=dtensor(MethodsTSPenTestAB.rescaling(Tensor,width,length))

    vector_all_features = MethodsTSPenTestAB.Transform_tensor_into_featuresmatrix(TensorTFR)
    matrix_trainvalid, matrix_test, y_trainvalid, y_test = train_test_split( vector_all_features, labels, test_size= ratio_test, random_state=42)
    
    Tensor_test = MethodsTSPenTestAB.Transform_featuresmatrix_into_tensor(matrix_test,firstsize,secondsize)
    # 2 fold CV
    skf1 = StratifiedKFold(n_splits= nb_val_split,random_state=seed,shuffle=True)
    for train_index, valid_index in skf1.split(matrix_trainvalid,y_trainvalid):
        matrix_train, matrix_valid = matrix_trainvalid[train_index], matrix_trainvalid[valid_index]
        y_train, y_valid = y_trainvalid[train_index], y_trainvalid[valid_index]
        Tensor_train = MethodsTSPenTestAB.Transform_featuresmatrix_into_tensor(matrix_train,firstsize,secondsize)
        Tensor_valid = MethodsTSPenTestAB.Transform_featuresmatrix_into_tensor(matrix_valid,firstsize,secondsize)
        matrix=MethodsTSPenTestAB.Cross_valNMF_AR(Tensor_train, y_train, Tensor_valid,y_valid,C_values,G_values,Lambda_info,Alpha,Lratio,Kd,nbclasses,pool,pool_decision)

        sum_matrix += matrix

    max_positions=np.argwhere(sum_matrix==np.max(sum_matrix))
    C_svm=C_values[max_positions[0][0]]
    G_svm=G_values[max_positions[0][1]]
    lambda_info=Lambda_info[max_positions[0][2]]
    alpha=Alpha[max_positions[0][3]]
    lratio=Lratio[max_positions[0][4]]
    lambdaG=lratio*alpha
    lambda_Af=(1-lratio)*alpha
    lambda_At=(1-lratio)*alpha
    lambda_Bf=(1-lratio)*alpha
    lambda_Bt=(1-lratio)*alpha
        
    Tensor_train = MethodsTSPenTestAB.Transform_featuresmatrix_into_tensor(matrix_trainvalid,firstsize,secondsize)
    y_train = y_trainvalid
    array_of_examples_numbers = MethodsTSPenTestAB.Determine_the_number_of_examples_per_class(y_train)
    D = MethodsTSPenTestAB.NMF_penalized_PG(Tensor_train,lambda_info,alpha,lratio,Kd,nbclasses,array_of_examples_numbers)


    Training_features=MethodsTSPenTestAB.Feature_extraction_process(Tensor_train,D,alpha,lratio,pool,pool_decision)
    Test_features=MethodsTSPenTestAB.Feature_extraction_process(Tensor_test,D,alpha,lratio,pool,pool_decision)


    scaler=StandardScaler()
    Training_features=scaler.fit_transform(Training_features)
    Test_features=scaler.transform(Test_features)

    clf=svm.SVC(C=C_svm,gamma=G_svm,kernel='rbf')
    clf.fit(Training_features,y_train)
    y_pred = clf.predict(Test_features)
    y_pred_app = clf.predict(Training_features)
    bc_test = sum(y_pred  == y_test)/len(y_test)
    bc_app = sum(y_pred_app  == y_train)/len(y_pred_app)
    print('bc_app : ',bc_app)
    print('bc_test: ',bc_test)
    Perf[i]= bc_test


    with h5py.File('Perf_toy_NMF' + str(nb_train)+ '.h5','w') as hf:
        hf.create_dataset('Performances'+str(i), data=Perf)
        hf.create_dataset('SVMPenaltyparam'+str(i), data=C_svm)
        hf.create_dataset('SVMVariance'+str(i), data=G_svm)
pool.terminate()
