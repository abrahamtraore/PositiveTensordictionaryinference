#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:45:27 2017

@author: alain
"""

import h5py
import numpy as np
#import matplotlib.pylab as plt
import Methods
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
import MethodsTSPenTestAB


pathfile = '../../RawData/'
with h5py.File(pathfile + 'toy.h5','r') as hf:
    X = np.array(hf.get('X')).astype('float32')
    y = np.array(hf.get('y')).astype('uint32')
    
y = np.squeeze(y)

ratio_test = 0.30 #The number of test exmaples represents 20% of the total(500)
ratio_valid = 0.30 #The number of validation exmaples represents 10% of the total(500)
seed1 = 5
seed2 = 5
Tensor_train,y_train,Tensor_test,y_test,Tensor_valid,y_valid = Methods.train_test_validation_tensors(X,y,ratio_test,ratio_valid,seed1,seed2)
index = np.argsort(y_train)
Tensor_train= Tensor_train[index]
y_train = y_train[index]
nbclasses = 2
kt = 3
kf = 3
T = 2
tol = np.power(10,-3,dtype=float)
maximum_iterations = 10

pool=Pool(2)
InitStrat="Multiplicative"#"Tucker2HOSVD"
parallel_decision=True
lambdainfo,lambdaG,lambdaAf,lambdaAt,lambdaBf,lambdaBt = 0.001, 0.001, 0.001, 0.001, 0.001, 0.001
G,A_f,A_t,error_list,nb_iter=MethodsTSPenTestAB.PenalizedTuckerTSPen(Tensor_train,y_train,nbclasses,kf,kt,maximum_iterations,tol,InitStrat,lambdainfo,lambdaG,lambdaAf,lambdaAt,lambdaBf,lambdaBt,T,parallel_decision,pool)

G_a = Methods.Test_features_extraction(Tensor_train,A_f,A_t)
G_t = Methods.Test_features_extraction(Tensor_test,A_f,A_t)

scaler = StandardScaler()
x_train = scaler.fit_transform(G_a)
x_test = scaler.transform(G_t)

C = 1  # SVM regularization parameter
lin_svc = svm.LinearSVC(C=C).fit(x_train, y_train)
class_train = lin_svc.score(x_train,y_train)
class_test = lin_svc.score(x_test,y_test)
print('train perf:', class_train)
print('test perf:', class_test)

#plt.imshow(G_a,aspect = 0.1,interpolation='none')

