#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:44:50 2017

@author: Traoreabraham
"""

import h5py
from multiprocessing import Pool
from sklearn.svm import LinearSVC
import pdb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
import MethodsTSPen
scaler=StandardScaler()
import sys

sys.path.append("/home/scr/etu/sin811/traorabr/sktensor/")

from sktensor import dtensor

#filename="/Users/Traoreabraham/Desktop/ProjectGit/DCase2013/dcase13-cqtrain.h5"
filename="/home/scr/etu/sin811/traorabr/DCase2013/dcase13-cqtrain.h5"
#filename='/home/arakoto/recherche/nnTensorFact/RawDatadcase2013/dcase13-cqt.h5'

with h5py.File(filename,'r') as hf:
         data = hf.get('x')
         Tensor_train = np.array(data)
         data = hf.get('y')
         y_train = np.array(data)
         y_train=y_train+1
         data = hf.get('set')
         set = np.array(data)
            
#filename="/Users/Traoreabraham/Desktop/ProjectGit/DCase2013/dcase13-cqtest.h5"
filename="/home/scr/etu/sin811/traorabr/DCase2013/dcase13-cqtest.h5"
#filename='/home/arakoto/recherche/nnTensorFact/RawDatadcase2013/dcase13-cqt.h5.test'

with h5py.File(filename,'r') as hf:
         data = hf.get('x')
         Tensor_test = np.array(data)
         data = hf.get('y')
         y_test = np.array(data)
         y_test=y_test+1
         data = hf.get('set')
         set = np.array(data)


#import matplotlib.pyplot as plt
#
#If=60
#It=60
#Tensor_train=dtensor(MethodsTSPen.rescaling(Tensor_train,If,It))
#Signal=Tensor_train[0,:,:]
#Kd=40
#W=np.random.rand(60,Kd)
#H=np.random.rand(Kd,60)
#Alpha=np.power(10,-2,dtype=int)
#Lratio=np.power(10,-2,dtype=int)
#DPG,APG,n_iter=NMFdecompTestAB._fit_projected_gradient(Signal,W,H,1e-8,30,200,Alpha,Lratio,None,0.1,0.1)
#
#DCD,ACD,nb_iter=NMFdecompTestAB._fit_coordinate_descent(Signal,W,H,tol=1e-8,max_iter=30,alpha=Alpha,l1_ratio=Lratio,regularization='mixed_transformation',update_H=True,verbose=0,shuffle=False,random_state=2)                 
#
#plt.imshow(DCD)
#plt.show()
#plt.imshow(ACD)
#plt.show()
#plt.imshow(DPG)
#plt.show()
#plt.imshow(APG)
#plt.show()
#
#print("The final error for PG")
#print(np.linalg.norm(Signal-np.dot(DPG,APG)))
#print("The final error for CD")
#print(np.linalg.norm(Signal-np.dot(DCD,ACD)))
#
#pdb.set_trace()

scaler=StandardScaler()

If=60

It=20

#These values represent the former performances
#Kt=10,Kd=20,Perf=48%,np.max_score=47%
#Kt=10,Kd=40,Perf=61%,np.max_score=55%
#Kt=10,Kd=60,Perf=64%,np.max_score=53%
#Kt=10,Kd=80,Perf=55%,np.max_score=49%
#Kt=10,Kd=100,Perf=52%,np.max_score=43%
#Kt=10,Kd=120,Perf=51%,np.max_score=0.39%
#These are the parameters for the former values
#C_values=np.power(10,np.arange(-3,4,2),dtype=float)
#G_values=np.power(10,np.arange(-3,1,1),dtype=float)
#Lambda_info=np.power(10,np.arange(-4,1,1),dtype=float)#max=2
##The arrays Alpha and Lratio should have the same lenghts or at least one of them must be a length-1 array
#Alpha=np.power(10,np.arange(-3,0,1),dtype=float)
#Lratio=np.power(10,np.arange(-3,0,1),dtype=float)


#These values represent the former performances
#Alpha= [0.01,0.1,1,10,20]
#Lratio=[0.5,0.6, 0.7,0.8,0.9,0.95,0.99]
#Kt=10,Kd=20,Perf=49%,np.max_score=49%
#Kt=10,Kd=40,Perf=53%,np.max_score=53%
#Kt=10,Kd=60,Perf=%,np.max_score=%
#Kt=10,Kd=80,Perf=%,np.max_score=%
#Kt=10,Kd=100,Perf=%,np.max_score=%
#Kt=10,Kd=120,Perf=%,np.max_score=%

Tensor_train=dtensor(MethodsTSPen.rescaling(Tensor_train,If,It))

Tensor_test=dtensor(MethodsTSPen.rescaling(Tensor_test,If,It))

vector_all_features=MethodsTSPen.Transform_tensor_into_featuresmatrix(Tensor_train)


[firstsize,secondsize]=np.array(Tensor_train[0,:,:].shape)

K=5

seed=2

skf=StratifiedKFold(n_splits=K,random_state=seed,shuffle=True)

#C_values=np.power(10,np.arange(-3,4,2),dtype=float)
#G_values=np.power(10,np.arange(-3,4,2),dtype=float)
#Lambda_info=np.power(10,np.arange(-3,4,2),dtype=float)
##The arrays Alpha and Lratio should have the same lenghts or at least one of them must be a length-1 array
#Alpha=np.power(10,np.arange(-3,1,2),dtype=float)
#Lratio=np.power(10,np.arange(-3,1,2),dtype=float)#The values of Lratio should be greater than 0 and strictly inferior to 1


#C_values=np.power(10,np.arange(-3,4,2),dtype=float)
#G_values=np.power(10,np.arange(-3,1,1),dtype=float)
#Lambda_info=np.power(10,np.arange(-4,1,1),dtype=float)
#The arrays Alpha and Lratio should have the same lenghts or at least one of them must be a length-1 array
#Alpha= [0.01,0.1,1,10,20]
#Lratio=[0.5,0.6, 0.7,0.8,0.9,0.95,0.99]

#Alpha= [0.001,0.01,0.1,1,5,10,15,20]
#Lratio=[0.001,0.01,0.1,0.5,0.6, 0.7,0.8,0.9,0.95,0.99]
#These values represent the former performances

#Kt=10,Kd=20,Perf=49%,np.max_score=49%,alpha=5,lambda_info=0.0001,lratio=0.001
#Kt=10,Kd=40,Perf=61%,np.max_score=55%,alpha=0.01,lambda_info=0.001,lratio=0.001
#Kt=10,Kd=60,Perf=%,np.max_score=%
#Kt=10,Kd=80,Perf=%,np.max_score=%
#Kt=10,Kd=100,Perf=%,np.max_score=%
#Kt=10,Kd=120,Perf=%,np.max_score=%

#LambdaG=Lratio*Alpha
#Lambda_Af=(1-Lratio)*Alpha
#Lambda_At=(1-Lratio)*Alpha
#Lambda_Bf=(1-Lratio)*Alpha
#Lambda_Bt=(1-Lratio)*Alpha



C_values=np.power(10,np.arange(-3,4,2),dtype=float)
G_values=np.power(10,np.arange(-5,3,1),dtype=float)
Lambda_info=np.power(10,np.arange(-3,4,2),dtype=float)
Alpha= [0.01,0.1,1,10,20]
Lratio=[0.001,0.01,0.1,0.25,0.5,0.75]


T=2
nbclasses=10
kt=1
kf=18
tol=np.power(10,-8,dtype=float)
#maximum_iteration=10
#max_iter_update=int(maximum_iteration/2)

maximum_iteration=10

max_iter_update=100

pool=Pool(8)
InitStrat="Multiplicative"#"Tucker2HOSVD"
parallel_decision=True

##mean_score_old=np.zeros((len(C_values),len(G_values),len(Lambda_info),len(LambdaG),len(Lambda_Af),len(Lambda_At),len(Lambda_Bf),len(Lambda_Bt)))
##mean_score=np.zeros((len(C_values),len(G_values),len(Lambda_info),len(LambdaG),len(Lambda_Af),len(Lambda_At),len(Lambda_Bf),len(Lambda_Bt)))
#mean_score=np.zeros((len(C_values),len(G_values),len(Lambda_info),len(Alpha),len(Lratio)))
#k=-1 
#for train_index, valid_index in skf.split(vector_all_features,y_train):
#    k=k+1
#    matrixtrain,matrixvalid=vector_all_features[train_index],vector_all_features[valid_index]         
#    ytrain,yvalid=y_train[train_index],y_train[valid_index]         
#    Tensortrain=MethodsTSPen.Transform_featuresmatrix_into_tensor(matrixtrain,firstsize,secondsize)
#    Tensorvalid=MethodsTSPen.Transform_featuresmatrix_into_tensor(matrixvalid,firstsize,secondsize) 
#    
#    #matrix=MethodsTSPen.Cross_valTSPenDCaseLinear(Tensortrain,ytrain,Tensorvalid,yvalid,K,seed,C_values,np.array([5]),Lambda_info,Alpha,Lratio,T,nbclasses,kf,kt,maximum_iteration,max_iter_update,tol,InitStrat,parallel_decision,pool)[5]    
#    matrix=MethodsTSPen.Cross_valTSPenDCase(Tensortrain,ytrain,Tensorvalid,yvalid,K,seed,C_values,G_values,Lambda_info,Alpha,Lratio,T,nbclasses,kf,kt,maximum_iteration,max_iter_update,tol,InitStrat,parallel_decision,pool)[5]
#    
#    mean_score=mean_score+matrix
#    print("The fold"+" "+str(k)+" "+"is finished")

#max_positions=np.argwhere(mean_score==np.max(mean_score))
#C_svm=C_values[max_positions[0][0]]
#G_svm=G_values[max_positions[0][1]]
#lambda_info=Lambda_info[max_positions[0][2]]
#alpha=Alpha[max_positions[0][3]]
#lratio=Lratio[max_positions[0][4]]

##Jt=140, 
#alpha=0.1
#lambda_info=10.0
#lratio=0.01

##Jt=180
alpha=0.01
lambda_info=1000.0
lratio=0.1

lambdaG=lratio*alpha
lambda_Af=(1-lratio)*alpha
lambda_At=(1-lratio)*alpha
lambda_Bf=(1-lratio)*alpha
lambda_Bt=(1-lratio)*alpha
   

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
    
G,A_f,A_t,B_f,B_t,error_list,nb_iter=MethodsTSPen.PenalizedTuckerTSPen(Tensor_train,y_train,nbclasses,kf,kt,maximum_iteration,max_iter_update,tol,InitStrat,lambda_info,lambdaG,lambda_Af,lambda_At,lambda_Bf,lambda_Bt,T,parallel_decision,pool)

with h5py.File('/home/scr/etu/sin811/traorabr/ExampleDictionaryV2/Parameters'+'.'+'h5' ,'w') as hf:
     hf.create_dataset('Spectraldictionaries', data = A_f)
     hf.create_dataset('Temporaldictionaries', data = A_t)
#     hf.create_dataset('Activationcoefficients'+str(seeds[i]), data=G)
#           hf.create_dataset('Performances'+str(seeds[i]), data=np.mean(Perf))
#           hf.create_dataset('SVMPenaltyparam'+str(seeds[i]), data=C_svm)
#           hf.create_dataset('SVMVariance'+str(seeds[i]), data=G_svm)

pool.terminate() 
pdb.set_trace()
