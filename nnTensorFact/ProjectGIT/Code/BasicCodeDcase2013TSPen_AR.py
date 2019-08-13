#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:10:43 2017

@author: Traoreabraham
"""

import h5py
from multiprocessing import Pool
import pdb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
import MethodsTSPen
scaler=StandardScaler()
import sys, os, getopt

sys.path.append("/home/scr/etu/sin811/traorabr/sktensor/")

from sktensor import dtensor


kf = 2
kt = 2
opts, args = getopt.getopt(sys.argv[1:],"f:t:")
for opt,arg in opts:
    if opt == '-f':  # 
        kf= int(arg)
    if opt == '-t':  #
        kt= int(arg)

if str.find(os.getcwd(),'alain')>0:
    pathdata = '/home/alain/'
elif  str.find(os.getcwd(),'arakoto')>0:
    pathdata = '/home/arakoto/'


filename=pathdata + 'recherche/nnTensorFact/RawDatadcase2013/dcase13-cqt.h5'
with h5py.File(filename,'r') as hf:
         data = hf.get('x')
         Tensor_train = np.array(data)
         data = hf.get('y')
         y_train = np.array(data)
         y_train=y_train+1
         data = hf.get('set')
         set = np.array(data)
            
filename=pathdata + 'recherche/nnTensorFact/RawDatadcase2013/dcase13-cqt.h5.test'
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


[firstsize,secondsize]=np.array(Tensor_train[0,:,:].shape)

K=5

seed=2

skf=StratifiedKFold(n_splits=K,random_state=seed,shuffle=True)


C_values=np.power(10,np.arange(-3,4,2),dtype=float)
G_values=np.power(10,np.arange(-5,3,1),dtype=float)
Lambda_info=np.power(10,np.arange(-3,4,2),dtype=float)
Alpha= [0.01,0.1,1,10,20]
Lratio=[0.001, 0.01, 0.1, 0.25,0.5,0.75]





T=2
nbclasses=10


tol=np.power(10,-8,dtype=float)
maximum_iteration=10
max_iter_update= 100
pool=Pool(32)
InitStrat="Multiplicative"#"Tucker2HOSVD"
parallel_decision=False

#mean_score_old=np.zeros((len(C_values),len(G_values),len(Lambda_info),len(LambdaG),len(Lambda_Af),len(Lambda_At),len(Lambda_Bf),len(Lambda_Bt)))
#mean_score=np.zeros((len(C_values),len(G_values),len(Lambda_info),len(LambdaG),len(Lambda_Af),len(Lambda_At),len(Lambda_Bf),len(Lambda_Bt)))
mean_score=np.zeros((len(C_values),len(G_values),len(Lambda_info),len(Alpha),len(Lratio)))
k=-1 
print('kf:',kf,'kt:',kt)
for train_index, valid_index in skf.split(vector_all_features,y_train):
    k=k+1
    matrixtrain,matrixvalid=vector_all_features[train_index],vector_all_features[valid_index]         
    ytrain,yvalid=y_train[train_index],y_train[valid_index]         
    Tensortrain=MethodsTSPen.Transform_featuresmatrix_into_tensor(matrixtrain,firstsize,secondsize)
    Tensorvalid=MethodsTSPen.Transform_featuresmatrix_into_tensor(matrixvalid,firstsize,secondsize)   
   
    #matrix=MethodsTSPen.Cross_valTSPenDCase(Tensortrain,ytrain,Tensorvalid,yvalid,K,seed,C_values,G_values,Lambda_info,LambdaG,Lambda_Af,Lambda_At,Lambda_Bf,Lambda_Bt,T,nbclasses,kf,kt,maximum_iteration,tol,InitStrat,parallel_decision,pool)[8]  
    matrix=MethodsTSPen.Cross_valTSPenDCase(Tensortrain,ytrain,Tensorvalid,yvalid,K,seed,C_values,G_values,Lambda_info,Alpha,Lratio,T,nbclasses,kf,kt,maximum_iteration,max_iter_update,tol,InitStrat,parallel_decision,pool)[5]
    mean_score=mean_score+matrix
    print(matrix)
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
print("Le maximum du score pendant la phase de validation est")
print(np.max(mean_score/K))
print("La performance en test est")
print(np.mean(performances)) 
y_pred = clf.predict(Test_features)
y_pred_app = clf.predict(Training_features)
bc_test = sum(y_pred  == y_test)/len(y_test)
bc_app = sum(y_pred_app  == y_pred_app)/len(y_pred_app)
print('bc_app : ',bc_app)
print('bc_test: ',bc_test)
pool.terminate()  


filename = 'T_dcase13_kf' + str(kf) + 'kt' + str(kt) + '_If' + str(If) 
np.savez(filename, bc_app = bc_app, bc_test = bc_test, mean_score = mean_score, performances = performances)
