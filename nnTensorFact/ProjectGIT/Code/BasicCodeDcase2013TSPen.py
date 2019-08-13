#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:10:43 2017

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

filename="/Users/Traoreabraham/Desktop/ProjectGit/DCase2013/dcase13-cqtrain.h5"
#filename="/home/scr/etu/sin811/traorabr/DCase2013/dcase13-cqtrain.h5"
#filename='/home/arakoto/recherche/nnTensorFact/RawDatadcase2013/dcase13-cqt.h5'

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
kf=1
tol=np.power(10,-8,dtype=float)
#maximum_iteration=10
#max_iter_update=int(maximum_iteration/2)

maximum_iteration=10

max_iter_update=100

pool=Pool(8)
InitStrat="Multiplicative"#"Tucker2HOSVD"
parallel_decision=True


mean_score=np.zeros((len(C_values),len(G_values),len(Lambda_info),len(Alpha),len(Lratio)))
k=-1 
for train_index, valid_index in skf.split(vector_all_features,y_train):
    k=k+1
    matrixtrain,matrixvalid=vector_all_features[train_index],vector_all_features[valid_index]         
    ytrain,yvalid=y_train[train_index],y_train[valid_index]         
    Tensortrain=MethodsTSPen.Transform_featuresmatrix_into_tensor(matrixtrain,firstsize,secondsize)
    Tensorvalid=MethodsTSPen.Transform_featuresmatrix_into_tensor(matrixvalid,firstsize,secondsize)     
    matrix=MethodsTSPen.Cross_valTSPenDCase(Tensortrain,ytrain,Tensorvalid,yvalid,K,seed,C_values,G_values,Lambda_info,Alpha,Lratio,T,nbclasses,kf,kt,maximum_iteration,max_iter_update,tol,InitStrat,parallel_decision,pool)[5]    
    mean_score=mean_score+matrix
    print("The fold"+" "+str(k)+" "+"is finished")

max_positions=np.argwhere(mean_score==np.max(mean_score))
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
  

G,A_f,A_t,B_f,B_t,error_list,nb_iter=MethodsTSPen.PenalizedTuckerTSPen(Tensor_train,y_train,nbclasses,kf,kt,maximum_iteration,max_iter_update,tol,InitStrat,lambda_info,lambdaG,lambda_Af,lambda_At,lambda_Bf,lambda_Bt,T,parallel_decision,pool)
#with h5py.File("/Users/Traoreabraham/Desktop/ImagesDCJf"+str(10*kf)+".h5",'w') as hf:        
     #hf.create_dataset('SpectraldictionariesJf'+str(10*kf),data = A_f)
           
#with h5py.File("/Users/Traoreabraham/Desktop/ImagesDCaseJf"+str(10*kf)+".h5",'w') as hf:        
          #hf.create_dataset('TemporaldictionariesJf'+str(10*kf), data = A_t)          
#pdb.set_trace()   


    #We define the training features. They are obtained by vectorizing the matrices G[k,:,:] and normalized.
    
  #Training_features=MethodsTSPen.Test_features_extraction(Tensor_train,A_f,A_t)
#penalty=np.power(10,-4,dtype=float)*np.power(10,-3,dtype=float)
penalty=lambdaG
Training_features=MethodsTSPen.decomposition_all_examples_parallelized(Tensor_train,np.kron(A_f,A_t),penalty,pool)
Training_features=scaler.fit_transform(Training_features)

      #We define the test features and normalize them
      #Test_features=MethodsTSPen.Test_features_extraction(Tensor_test,A_f,A_t)
      #Test_features=MethodsTSPenTestAB.decomposition_all_examples_parallelized(Tensor_test,np.kron(A_f,A_t),pool)
Test_features=MethodsTSPen.decomposition_all_examples_parallelized(Tensor_test,np.kron(A_f,A_t),penalty,pool)     
Test_features=scaler.transform(Test_features)

      #We define the classifier and perform the classification task
clf=svm.SVC(C=C_svm,gamma=G_svm,kernel='rbf')
#clf=LinearSVC(C=C_svm)
clf.fit(Training_features,y_train)
    #clf=KNeighborsClassifier(n_neighbors=5)
    #clf.fit(Training_features,y_train)

    #Recover_classes_all_labels is designed to recover the classes present in an array with the help of the labels. 
    #There are three outputs:
    #result corresponds to the examples followed by their classes label
    #indexes contains the classes present in the array
    #number_of_samples_per_class yields the number of examples per class.
result,indexes,number_of_samples_per_class=MethodsTSPen.Recover_classes_all_labels(Test_features,y_test)
    #The function classificaton_rate_classes is designed in order to determine the performances per classe in order to compute the MAP
    #There are two outputs:
      #performances represent the proportion of well classified examples per class.
      #confusion_matrix represents the prediction of the classifier for each example in each class. It is usefull to understand which example has not been well classified
performances,confusionmatrix=MethodsTSPen.classificaton_rate_classes(result,clf)
print("The value of Alpha is")
print(alpha)
print("The value of lambda_info is")
print(lambda_info)
print("The value of lratio")
print(lratio)
print("Le maximum du score pendant la phase de validation est")
print(np.max(mean_score/K))
print("La performance en test est")
print(np.mean(performances))

pool.terminate() 
pdb.set_trace()
#The final test sketch
#Jf=20, Alpha=0.01, lambda_info=0.001, lratio=0.1, max_score=52%, Perf=58%
#Jt=40, Alpha=1, lambda_info=0.001, lratio=0.001, max_score=56%, Perf=59%
#Jt=60, Alpha=10, lambda_info=10.0, lratio=0.001, max_score=51%, Perf=57%
#Jt=80, Alpha=0.01, lambda_info=0.001, lratio=0.01, max_score=54%, Perf=68%                                                                   
#Jt=100,Alpha=0.01, lambda_info=0.1, lratio=0.001, max_score=56%, Perf=52%
#Jt=120,Alpha=0.01, lambda_info=10, lratio=0.01, max_score=53%, Perf=58%                                                           
#Jt=140, Alpha=0.1, lambda_info=10.0, lratio=0.01, max_score=49%, Perf=58%                                                         
#Jt=160 , Alpha=0.1, lambda_info=1000.0, lratio=0.01, max_score=51%, Perf=67%       
#Jt=180 ,Alpha=0.01, lambda_info=1000.0, lratio=0.1, max_score=56%, Perf=67%                                                    
#Jt=200 ,Alpha=0.01, lambda_info=1000.0, lratio=0.25, max_score=47%, Perf=51%
#Jt=220, Alpha=0.1, lambda_info=1000.0, lratio=0.001, max_score=48%, Perf=53%     