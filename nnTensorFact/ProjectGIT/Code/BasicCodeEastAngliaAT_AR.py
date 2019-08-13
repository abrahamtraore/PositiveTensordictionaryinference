#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 09:04:49 2017

@author: Traoreabraham
"""

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn import svm
import logging
import itertools
import os
import MethodsTSPen as Methods



np.random.seed(1)

logging.basicConfig(filename='test_log.log',level=logging.DEBUG,format='%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s')

if os.getcwd().find('alain') > 0:
    BASE_PATH = "/home/alain/recherche/nnTensorFact/RawDataEastAnglia"
else:
    BASE_PATH = "/home/arakoto/recherche/nnTensorFact/RawDataEastAnglia"


TensorTFR,labels=Methods.load_preprocessed_data(BASE_PATH)
print("The adress is the good one")


seed=2
[firstsize,secondsize]=np.array(TensorTFR.shape,dtype=int)[1:3]
vector_all_features=Methods.Transform_tensor_into_featuresmatrix(TensorTFR)

K = 5
skf=StratifiedKFold(n_splits=K,random_state=seed,shuffle=True)#skf=StratifiedKFold(n_splits=5,random_state=17,shuffle=True)

Perf=np.zeros(K)   

k=0
tol=np.power(10,-6,dtype=float)
maximum_iterations=10
nbclasses=10


ktvec = [5]
kfvec = [5]
n_h_val, n_C_val, n_G_val = 10,10,6    
init_strategy = 'Multiplicative'



C_values=np.power(10,np.linspace(-2,9,n_C_val),dtype=float)
G_values=np.power(10,np.linspace(-4,-1,n_G_val),dtype=float)
Hyper_values=np.logspace(-11,-6,n_h_val)
lambdainfo, lambdaG, lambdaAf ,lambdaAt, lambdaBf ,lambdaBt = 0.01, 0.01, 10, 10, 10, 10


for kt, kf in zip(ktvec, kfvec):
    outfile = 'resultat_EA_kt{:}_kf{:}.npz'.format(kt,kf)
    print(outfile)
    resultat = np.zeros((2,n_h_val,n_C_val, n_G_val,K))
    for train_index, test_index in skf.split(vector_all_features,labels):
        
        matrix_train, matrix_test = vector_all_features[train_index],vector_all_features[test_index]
        y_train, y_test=labels[train_index],labels[test_index]
        Tensor_train=Methods.Transform_featuresmatrix_into_tensor(matrix_train,firstsize,secondsize)
        Tensor_test=Methods.Transform_featuresmatrix_into_tensor(matrix_test,firstsize,secondsize) 
    
        for i_h, lambdainfo  in enumerate((Hyper_values)):

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
                    print('{:d} {:2.2e} C = {:f} sig = {:2.2e}  {:2.3f} {:2.3f}'.format(k, lambdainfo,C_svm,G_svm, bc_train, bc_test))
                    resultat[0,i_h,i_C,i_G,k] = bc_train
                    resultat[1,i_h,i_C,i_G,k] = bc_test
                    np.savez(outfile, resultat = resultat)
        k +=1

#
#a = np.load('resultat.npz')
#resultat = a['resultat']
#resultat.shape[0]
#for i in range(resultat.shape[0]):
#    print(''.join('{:2.4f} '.format(k) for k in (resultat[i]))) 

