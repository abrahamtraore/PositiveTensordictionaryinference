#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 13:02:48 2017

@author: Traoreabraham
"""
import NMFdecompTestAB

import h5py
from multiprocessing import Pool 

from sklearn.model_selection import StratifiedKFold
import MethodsTSPen
#import sys
import pdb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
#sys.path.append("/home/scr/etu/sin811/traorabr/sktensor/")
from sktensor import dtensor


def NMF_penalized_PG(Tensor,W,H,lambdainfo,Alpha,Lratio,Kd,nbclasses,array_of_examples_numbers): 
    [K,nrows,ncols]=np.array(np.shape(Tensor),dtype=int)
    L=K*ncols
    Signalsmatrix=MethodsTSPen.TransformTensorToSignals(Tensor)    
    C=MethodsTSPen.Informationmatrix(nbclasses,Kd,L,array_of_examples_numbers) 
    Signals_tilde=np.row_stack((Signalsmatrix,np.sqrt(lambdainfo)*C))        
    #nrow=np.array(Signals_tilde.shape,dtype=int)[0]
    #W=np.random.rand(nrow,Kd)
    #H=np.random.rand(Kd,L)    
    #D_tilde,A,nb_iter=NMFV2._fit_coordinate_descent(Signals_tilde,W,H,tol=1e-4,max_iter=200,alpha=Alpha,l1_ratio=Lratio,regularization='mixed_transformation',update_H=True ,verbose=0, shuffle=False,random_state=1)                      
    D_tilde,A,n_iter=NMFdecompTestAB._fit_projected_gradient(Signals_tilde,W,H,1e-8,20,500,Alpha,Lratio,None,0.1,0.1)
    #D=D_tilde[0:nrows,:] 
    return D_tilde,A,n_iter

def Cross_valNMF_PG_DCase(Tensor_train,Tensor_valid,y_train,y_valid,C_values,G_values,Lambdainfo,Alpha,Lratio,Kd,nbclasses,pool,pool_decision):
      matrix=np.zeros((len(C_values),len(G_values),len(Lambdainfo),len(Lratio),len(Alpha)))
      scaler = StandardScaler() 
      array_of_examples_numbers=MethodsTSPen.Determine_the_number_of_examples_per_class(y_train)      
      [K,nrows,ncols]=np.array(np.shape(Tensor_train),dtype=int)
      L=K*ncols
      Signalsmatrix=MethodsTSPen.TransformTensorToSignals(Tensor_train)    
      C=MethodsTSPen.Informationmatrix(nbclasses,Kd,L,array_of_examples_numbers) 
      Signals_tilde=np.row_stack((Signalsmatrix,C))        
      nrow=np.array(Signals_tilde.shape,dtype=int)[0]
      W_old=np.random.rand(nrow,Kd)
      H_old=np.random.rand(Kd,L) 
      for Lambdainfo_position in range(len(Lambdainfo)):
        for Alpha_position in range(len(Alpha)):
          for Lratio_position in range(len(Lratio)):
            lambdainfo=Lambdainfo[Lambdainfo_position] 
            alpha=Alpha[Alpha_position]
            lratio=Lratio[Lratio_position]           
            D_tilde,A,n_iter=NMF_penalized_PG(Tensor_train,W_old,H_old,lambdainfo,alpha,lratio,Kd,nbclasses,array_of_examples_numbers)
            D=D_tilde[0:nrows,:]
            penalty=alpha*lratio
            Training_features=MethodsTSPen.Feature_extraction_process(Tensor_train,D,alpha,lratio,penalty,pool,pool_decision)                             
            Training_features=scaler.fit_transform(Training_features)
            Valid_features=MethodsTSPen.Feature_extraction_process(Tensor_valid,D,alpha,lratio,penalty,pool,pool_decision)
            Valid_features=scaler.transform(Valid_features)
            for C_position in range(len(C_values)):
              for G_position in range(len(G_values)): 
               C_svm=C_values[C_position]
               Gamma=G_values[G_position]
               clf=svm.SVC(C=C_svm,gamma=Gamma,kernel='rbf')
               clf.fit(Training_features,y_train)
               y_predict=clf.predict(Valid_features)
               scores=np.array(y_predict==y_valid)
               matrix[C_position,G_position,Lambdainfo_position,Lratio_position,Alpha_position]=np.mean(scores) 
            W_old=D_tilde
            H_old=A
      max_positions=np.argwhere(matrix==np.max(matrix))
      C_selected=C_values[max_positions[0][0]]
      G_selected=G_values[max_positions[0][1]]
      Lambdainfo_selected=Lambdainfo[max_positions[0][2]]
      Lratio_selected=Lratio[max_positions[0][3]]
      Alpha_selected=Alpha[max_positions[0][4]]
      return C_selected,G_selected,Lambdainfo_selected,Lratio_selected,Alpha_selected,matrix

#filename="/Users/Traoreabraham/Desktop/ProjectGit/DCase2013/dcase13-cqtrain.h5"
filename="/home/scr/etu/sin811/traorabr/DCase2013/dcase13-cqtrain.h5"
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
with h5py.File(filename,'r') as hf:
        data = hf.get('x')
        Tensor_test = np.array(data)
        data = hf.get('y')
        y_test = np.array(data)
        y_test=y_test+1
        data = hf.get('set')
        set = np.array(data)

scaler=StandardScaler()

If=60

It=20

Tensor_train=dtensor(MethodsTSPen.rescaling(Tensor_train,If,It))

Tensor_test=dtensor(MethodsTSPen.rescaling(Tensor_test,If,It))

vector_all_features=MethodsTSPen.Transform_tensor_into_featuresmatrix(Tensor_train)

#[firstsize,secondsize]=np.array(np.shape(Tensor_test[0,:,:]),dtype=int)

#vector_all_features=MethodsTSPen.Transform_tensor_into_featuresmatrix(Tensor_train)

seed=2  #Trueseed=2


#Matrix=MethodsTSPen.TransformTensorToSignals(Tensor_train)
##print("The fitting for PG error is")
##print(np.linalg.norm(Matrix-np.dot(DPG,APG))) 
##print("The number of for PG is")
##print(n_iterPG)
#
#[K,nrows,ncols]=np.array(np.shape(Tensor_train),dtype=int)
#L=K*ncols
#Kd=10
#Winit1=np.random.rand(nrows,Kd)
#Hinit1=np.random.rand(Kd,L)
#
#Winit2=np.copy(Winit1)
#Hinit2=np.copy(Hinit1)
#
#print(Matrix.shape)
#print(Winit1.shape)
#print(Hinit1.shape)
#
##import NMFdecompTestAB
#import NMFreal
#Alpha=30
#Lratio=0.5
#DCD,ACD,n_iterCD=NMFreal._fit_coordinate_descent(Matrix, Winit1,Hinit1,tol=1e-8, max_iter=2000,
#                                                 alpha=Alpha,l1_ratio=Lratio, regularization=None,
#                                                 update_H=True,verbose=0, shuffle=False,random_state=3)
# 
#DPG,APG,n_iterPG=NMFreal._fit_projected_gradient(Matrix, Winit2, Hinit2,1e-8,000,2000, Alpha,Lratio,
#                                                 None,0.1,0.1)
#
#import matplotlib.pyplot as plt
#plt.imshow(DCD)
#plt.show()
#plt.imshow(DPG)
#plt.show()
#print(np.linalg.norm(DCD))
#print(np.linalg.norm(DPG))
#pdb.set_trace()
#pdb.set_trace()
#==============================================================================
# C_values=np.power(10,np.arange(-3,4,2),dtype=float)
# G_values=np.power(10,np.arange(-3,4,2),dtype=float)
# Lambdainfo=np.power(10,np.arange(-4,2,2),dtype=float)#max=2
# #The arrays Alpha and Lratio should have the same lenghts or at least one of them must be a length-1 array
# Alpha=np.power(10,np.arange(-2,3,2),dtype=float)
# Lratio=np.power(10,np.arange(-6,-1,2),dtype=float)
# Kd=10 
#==============================================================================
  
#C_values=np.power(10,np.arange(-3,4,2),dtype=float)
#G_values=np.power(10,np.arange(-3,4,2),dtype=float)
#Lambdainfo=np.power(10,np.arange(-4,5,2),dtype=float)#max=2
##The arrays Alpha and Lratio should have the same lenghts or at least one of them must be a length-1 array
#Alpha=np.power(10,np.arange(-2,3,2),dtype=float)
#Lratio=np.power(10,np.arange(-6,-1,2),dtype=float)

#Kd=10,Perf=26%
#Kd=20,Perf=23%
#Kd=30,Perf=33%
#Kd=40,Perf=35%
#Kd=50,Perf=20%, max_validationscore=32%
#Kd=60,Perf=33%,maxvalidationscore=29%
#Kd=70,Perf=28%,maxvalidationscore=33%



#C_values=np.power(10,np.arange(-3,4,2),dtype=float)
#G_values=np.power(10,np.arange(-3,1,1),dtype=float)
#Lambdainfo=np.power(10,np.arange(-4,1,1),dtype=float)#max=2
#The arrays Alpha and Lratio should have the same lenghts or at least one of them must be a length-1 array
#Alpha=np.power(10,np.arange(-3,0,1),dtype=float)
#Lratio=np.power(10,np.arange(-3,0,1),dtype=float)



C_values=np.power(10,np.arange(-3,4,2),dtype=float)
G_values=np.power(10,np.arange(-5,3,1),dtype=float)
Lambdainfo=np.power(10,np.arange(-3,4,2),dtype=float)
Alpha= [0.01,0.1,1,10,20]
Lratio=[0.001,0.01,0.1,0.25,0.5,0.75]
 

Kd=160
pool=Pool(10)
pool_decision=True  
nbclasses=10
K=5

skf=StratifiedKFold(n_splits=K,random_state=seed,shuffle=True)
[firstsize,secondsize]=np.array(np.shape(Tensor_test[0,:,:]),dtype=int)
mean_score=np.zeros((len(C_values),len(G_values),len(Lambdainfo),len(Lratio),len(Alpha)))


for train_index, valid_index in skf.split(vector_all_features,y_train):
    matrixtrain,matrixvalid=vector_all_features[train_index],vector_all_features[valid_index]         
    ytrain,yvalid=y_train[train_index],y_train[valid_index]         
    Tensortrain=MethodsTSPen.Transform_featuresmatrix_into_tensor(matrixtrain,firstsize,secondsize)
    Tensorvalid=MethodsTSPen.Transform_featuresmatrix_into_tensor(matrixvalid,firstsize,secondsize)    
    matrix=Cross_valNMF_PG_DCase(Tensortrain,Tensorvalid,ytrain,yvalid,C_values,G_values,Lambdainfo,Alpha,Lratio,Kd,nbclasses,pool,pool_decision)[5]
    mean_score=mean_score+matrix


#We selected the values that minimise the error  

max_positions=np.argwhere(mean_score==np.max(mean_score))
C_svm=C_values[max_positions[0][0]]
G_svm=G_values[max_positions[0][1]]
lambdainfo=Lambdainfo[max_positions[0][2]]
lratio=Lratio[max_positions[0][3]]
alpha=Alpha[max_positions[0][4]]


#C_svm=10
#G_svm=5
#lambdainfo=np.power(10,-3,dtype=float)
#lratio=np.power(10,-2,dtype=float)
#alpha=np.power(10,1,dtype=float)

##We learn the dictionary
array_of_examples_numbers=MethodsTSPen.Determine_the_number_of_examples_per_class(y_train)
[K,nrows,ncols]=np.array(np.shape(Tensor_train),dtype=int)
L=K*ncols
Signalsmatrix=MethodsTSPen.TransformTensorToSignals(Tensor_train) 
C=MethodsTSPen.Informationmatrix(nbclasses,Kd,L,array_of_examples_numbers) 
Signals_tilde=np.row_stack((Signalsmatrix,C))        
nrow=np.array(Signals_tilde.shape,dtype=int)[0]
W=np.random.rand(nrow,Kd)
H=np.random.rand(Kd,L)
D_tildePG,APG,n_iterPG=NMF_penalized_PG(Tensor_train,W,H,lambdainfo,alpha,lratio,Kd,nbclasses,array_of_examples_numbers)
D=D_tildePG[0:nrows,:]
penalty=lratio*alpha
Training_features=MethodsTSPen.Feature_extraction_process(Tensor_train,D,alpha,lratio,penalty,pool,pool_decision)     
Training_features=scaler.fit_transform(Training_features)
Test_features=MethodsTSPen.Feature_extraction_process(Tensor_test,D,alpha,lratio,penalty,pool,pool_decision)
Test_features=scaler.transform(Test_features)
clf=svm.SVC(C=C_svm,gamma=G_svm,kernel='rbf')
clf.fit(Training_features,y_train)
result,indexes,number_of_samples_per_class=MethodsTSPen.Recover_classes_all_labels(Test_features,y_test)
performances,confusionmatrix=MethodsTSPen.classificaton_rate_classes(result,clf)
print("The value of Alpha is")
print(alpha)
print("The value of lambda_info is")
print(lambdainfo)
print("The value of lratio")
print(lratio)
print("Le maximum du score pendant la phase de validation est")
print(np.max(mean_score/5))
print("La performance en test est")
print(np.mean(performances))
pool.terminate()  
pdb.set_trace()
#The final test sketch
#Kd=20, Alpha=, lambdainfo=, lratio=, max_score=%, Perf=% pending in local
#Kd=40, Alpha=, lambdainfo=, lratio=, max_score=%, Perf=% pending in NMFPGPool
#Kd=60, Alpha=, lambdainfo=, lratio=, max_score=%, Perf=%  
#Kd=80, Alpha=, lambdainfo=, lratio=, max_score=%, Perf=%  
#Kd=100, Alpha=, lambdainfo=, lratio=, max_score=%, Perf=% 
#Kd=120, Alpha=, lambdainfo=, lratio=, max_score=%, Perf=% pending in NMFPGPoolbis
#Kd=140, Alpha=, lambdainfo=, lratio=, max_score=%, Perf=%
#Kd=160, Alpha=, lambdainfo=, lratio=, max_score=%, Perf=%
#Kd=180, Alpha=, lambdainfo=, lratio=, max_score=%, Perf=%
#Kd=200, Alpha=, lambdainfo=, lratio=, max_score=%, Perf=%
#Kd=220, Alpha=, lambdainfo=, lratio=, max_score=%, Perf=%