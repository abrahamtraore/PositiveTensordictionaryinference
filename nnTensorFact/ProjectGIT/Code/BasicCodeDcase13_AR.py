#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 09:04:49 2017

@author: Traoreabraham
"""

import numpy as np
import h5py

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn import svm
import logging
import itertools
import os
import MethodsTSPen as Methods

np.random.seed(1)

logging.basicConfig(filename='test_log.log',level=logging.DEBUG,format='%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s')

if os.getcwd().find('alain') > 0:
    BASE_PATH = "/home/alain/recherche/casacnn/dcase2013/"
else:
    BASE_PATH = "/home/arakoto/recherche/nnTensorFact/RawDatadcase2013/"

filename = 'dcase13-cqt.h5'
with h5py.File(BASE_PATH + filename,'r') as hf:
    TensorTFR = np.array(hf.get('x')).astype('float32')
    labels = np.array(hf.get('y')).astype('uint32') + 1

with h5py.File(BASE_PATH + filename + '.test', 'r') as hf:
        data=hf.get('x')
        Tensor_test=(np.array(data))
        data=hf.get('y')
        y_test=np.array(data)
        y_test=y_test+1

seed=2

width = 60
length = 60
TensorTFR = Methods.rescaling(TensorTFR,width,length)
Tensor_test = Methods.rescaling(Tensor_test,width,length)

[firstsize,secondsize]=np.array(TensorTFR.shape,dtype=int)[1:3]

matrix_train = Methods.Transform_tensor_into_featuresmatrix(TensorTFR)
matrix_test = Methods.Transform_tensor_into_featuresmatrix(Tensor_test)




#skf= ShuffleSplit(n_splits=K,test_size = 0.2, random_state = seed)#skf=StratifiedKFold(n_splits=5,random_state=17,shuffle=True)


tol=np.power(10,-6,dtype=float)

maximum_iterations=10

nbclasses=10

ktvec = [2, 5, 7, 10]
kfvec = [2, 5, 7, 10]
n_h_val, n_C_val, n_G_val = 10,10,6    
init_strategy = 'Multiplicative'

C_values=np.power(10,np.linspace(-2,9,n_C_val),dtype=float)
G_values=np.power(10,np.linspace(-7,-2,n_G_val),dtype=float)
Hyper_values=np.logspace(-11,-6,n_h_val)
lambdainfo, lambdaG, lambdaAf ,lambdaAt, lambdaBf ,lambdaBt = 0.01, 0.01, 10, 10, 10, 10

for kt, kf in zip(ktvec, kfvec):

    outfile = 'resultat_dcase13_kt{:}_kf{:}.npz'.format(kt,kf)
    resultat = np.zeros((2,n_h_val,n_C_val, n_G_val))
    y_train = labels
    Tensor_train=Methods.Transform_featuresmatrix_into_tensor(matrix_train,firstsize,secondsize)
    Tensor_test=Methods.Transform_featuresmatrix_into_tensor(matrix_test,firstsize,secondsize) 
    
    for i_h, lambdainfo  in enumerate((Hyper_values)):
        
 #       def PenalizedTuckerTSPen(Tensor_train,y_train,nbclasses,kf,kt,maximum_iteration,tol,InitStrat,lambdainfo,lambdaG,lambdaAf,lambdaAt,lambdaBf,lambdaBt,T,parallel_decision,pool):     

        G,A_f,A_t,error,_=Methods.PenalizedTuckerTSPen(Tensor_train,y_train,nbclasses,kf,kt,maximum_iterations,tol, InitStrat = init_strategy,
                                                       lambdainfo = lambdainfo,lambdaG = lambdaG,
                                                       lambdaAf = lambdaAf,lambdaAt =  lambdaAt,
                                                       lambdaBf = lambdaBf,lambdaBt = lambdaBt,
                                                       T = 10, parallel_decision = False,pool = 0)
    
        Training_features=Methods.Test_features_extraction(Tensor_train,A_f,A_t)
        Test_features=Methods.Test_features_extraction(Tensor_test,A_f,A_t)
        
        scaler=StandardScaler()
        
        Training_features=scaler.fit_transform(Training_features)
        Test_features=scaler.transform(Test_features)
          
        for i_C, C_svm in enumerate(C_values):
            for i_G, G_svm in enumerate(G_values):
                print(lambdainfo, C_svm,G_svm)      
                clf=svm.SVC(C=C_svm,gamma=G_svm,kernel='rbf', decision_function_shape='ovo')
                clf.fit(Training_features,y_train)
                y_pred_app = clf.predict(Training_features)
                y_pred_test = clf.predict(Test_features)
                bc_train = sum(y_pred_app == y_train)/len(y_train)
                bc_test = sum(y_pred_test == y_test)/len(y_test)
                print('{:2.2e} C = {:f} sig = {:2.2e}  {:2.3f} {:2.3f}'.format(lambdainfo,C_svm,G_svm, bc_train, bc_test))
                resultat[0,i_h,i_C,i_G] = bc_train
                resultat[1,i_h,i_C,i_G] = bc_test
                np.savez(outfile, resultat = np.array(resultat))

import numpy as np
a = np.load('resultat_dcase13_kt5_kf5.npz')
resultat = a['resultat']
print(resultat)
resultat.shape[0]
for i in range(resultat.shape[0]):
    print(''.join('{:2.4f} '.format(k) for k in (resultat[i]))) 

