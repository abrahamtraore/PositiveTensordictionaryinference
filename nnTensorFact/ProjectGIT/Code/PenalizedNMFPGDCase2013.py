#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 19:27:05 2017

@author: Traoreabraham
"""


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


filename="/Users/Traoreabraham/Desktop/ProjectGit/DCase2013/dcase13-cqtrain.h5"
#filename="/home/scr/etu/sin811/traorabr/DCase2013/dcase13-cqtrain.h5"
with h5py.File(filename,'r') as hf:
        data = hf.get('x')
        Tensor_train = np.array(data)
        data = hf.get('y')
        y_train = np.array(data)
        y_train=y_train+1
        data = hf.get('set')
        set = np.array(data)
           
filename="/Users/Traoreabraham/Desktop/ProjectGit/DCase2013/dcase13-cqtest.h5"
#filename="/home/scr/etu/sin811/traorabr/DCase2013/dcase13-cqtest.h5"
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
 

Kd=220
pool=Pool(10)
pool_decision="max" #False,#"mean",#"max"
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
    matrix=MethodsTSPen.Cross_valNMF_PG_DCase(Tensortrain,Tensorvalid,ytrain,yvalid,C_values,G_values,Lambdainfo,Alpha,Lratio,Kd,nbclasses,pool,pool_decision)[5]
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
D_tildePG,APG,n_iterPG=MethodsTSPen.NMF_penalized_PG(Tensor_train,W,H,lambdainfo,alpha,lratio,Kd,nbclasses,array_of_examples_numbers)
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
#Kd=20, Alpha=1, lambdainfo=0.001, lratio=0.25, max_score=25%, Perf=17%
#Kd=40, Alpha=1, lambdainfo=0.001, lratio=0.75, max_score=24%, Perf=24%
#Kd=60, Alpha=, lambdainfo=, lratio=, max_score=%, Perf=%  pending in NMFProjected
#Kd=80, Alpha=, lambdainfo=, lratio=, max_score=%, Perf=%  pending in NMFCoordinateDescent
#Kd=100, Alpha=, lambdainfo=, lratio=, max_score=%, Perf=% pending in local
#Kd=120, Alpha=, lambdainfo=, lratio=, max_score=%, Perf=%
#Kd=140, Alpha=, lambdainfo=, lratio=, max_score=%, Perf=%
#Kd=160, Alpha=, lambdainfo=, lratio=, max_score=%, Perf=%
#Kd=180, Alpha=, lambdainfo=, lratio=, max_score=%, Perf=%
#Kd=200, Alpha=, lambdainfo=, lratio=, max_score=%, Perf=%
#Kd=220, Alpha=, lambdainfo=, lratio=, max_score=%, Perf=%






















##C_values=np.power(10,np.arange(-3,4,2),dtype=float)
##G_values=np.power(10,np.arange(-3,4,2),dtype=float)
##Lambdainfo=np.power(10,np.arange(-4,5,2),dtype=float)#max=2
###The arrays Alpha and Lratio should have the same lenghts or at least one of them must be a length-1 array
##Alpha=np.power(10,np.arange(-2,3,2),dtype=float)
##Lratio=np.power(10,np.arange(-6,-1,2),dtype=float)  
##Kd=10,perf=18%
##Kd=20,perf=27%,max_score=30%
##Kd=30,perf=19%,max_score=28%
##Kd=40,perf=12%,max_score=28%
##Kd=50,perf=17%,max_score=28%
##Kd=60,perf=21%,max_score=26%
##Kd=70,perf=22%,max_score=27.%
#
#
#C_values=np.power(10,np.arange(-3,4,2),dtype=float)
#G_values=np.power(10,np.arange(-3,1,1),dtype=float)
#Lambdainfo=np.power(10,np.arange(-4,1,1),dtype=float)#max=2
##The arrays Alpha and Lratio should have the same lenghts or at least one of them must be a length-1 array
#Alpha=np.power(10,np.arange(-3,0,1),dtype=float)
#Lratio=np.power(10,np.arange(-3,0,1),dtype=float)
#


#Kd=120
##PG
##Kd=20,perf=27%,max_score=30%
##Kd=40,perf=16%,max_score=28%
##Kd=60,perf=15%,max_score=25%
##Kd=80,perf=11%,max_score=24%
##Kd=100,perf=17%,max_score=27%
##Kd=120,perf=18%,max_score=26%
#nbclasses=10


#pool=Pool(3)
#pool_decision=False   
#
#K=5
#seed=2
#  
#skf=StratifiedKFold(n_splits=K,random_state=seed,shuffle=True)
#[firstsize,secondsize]=np.array(np.shape(Tensor_test[0,:,:]),dtype=int)
#mean_score=np.zeros((len(C_values),len(G_values),len(Lambdainfo),len(Lratio),len(Alpha)))
##We perform the K-fold validation to tune the parameters
#for train_index, valid_index in skf.split(vector_all_features,y_train):
#    matrixtrain,matrixvalid=vector_all_features[train_index],vector_all_features[valid_index]         
#    ytrain,yvalid=y_train[train_index],y_train[valid_index]         
#    Tensortrain=MethodsTSPen.Transform_featuresmatrix_into_tensor(matrixtrain,firstsize,secondsize)
#    Tensorvalid=MethodsTSPen.Transform_featuresmatrix_into_tensor(matrixvalid,firstsize,secondsize) 
#   
#    matrix=MethodsTSPen.Cross_valNMF_PG_DCase(Tensortrain,Tensorvalid,ytrain,yvalid,C_values,G_values,Lambdainfo,Alpha,Lratio,Kd,nbclasses,pool,pool_decision)[5]
#   
#    mean_score=mean_score+matrix
#
#
##We selected the values that minimise the error  
#max_positions=np.argwhere(mean_score==np.max(mean_score))
#C_svm=C_values[max_positions[0][0]]
#G_svm=G_values[max_positions[0][1]]
#lambdainfo=Lambdainfo[max_positions[0][2]]
#lratio=Lratio[max_positions[0][3]]
#alpha=Alpha[max_positions[0][4]]
#
##We learn the dictionary
#array_of_examples_numbers=MethodsTSPen.Determine_the_number_of_examples_per_class(y_train)
#D=MethodsTSPen.NMF_penalized_PG(Tensor_train,lambdainfo,alpha,lratio,Kd,nbclasses,array_of_examples_numbers)
#penalty=lratio*alpha
##We extract the features by projected each TFR on the learned dictionary
#Training_features=MethodsTSPen.Feature_extraction_process(Tensor_train,D,alpha,lratio,penalty,pool,pool_decision)     
#Training_features=scaler.fit_transform(Training_features)
#Test_features=MethodsTSPen.Feature_extraction_process(Tensor_test,D,alpha,lratio,penalty,pool,pool_decision)
#Test_features=scaler.transform(Test_features)
#clf=svm.SVC(C=C_svm,gamma=G_svm,kernel='rbf')
#clf.fit(Training_features,y_train)
#    
#
#    #Recover_classes_all_labels is designed to recover the classes present in an array with the help of the labels. 
#    #There are three outputs:
#    #result corresponds to the examples followed by their classes label
#    #indexes contains the classes present in the array
#    #number_of_samples_per_class yields the number of examples per class.
#result,indexes,number_of_samples_per_class=MethodsTSPen.Recover_classes_all_labels(Test_features,y_test)
#    #The function classificaton_rate_classes is designed in order to determine the performances per classe in order to compute the MAP
#    #There are two outputs:
#      #performances represent the proportion of well classified examples per class.
#      #confusion_matrix represents the prediction of the classifier for each example in each class. It is usefull to understand which example has not been well classified
#performances,confusionmatrix=MethodsTSPen.classificaton_rate_classes(result,clf)
#print(np.mean(performances))
#print(np.max(mean_score))
#pool.terminate()  
#pdb.set_trace()