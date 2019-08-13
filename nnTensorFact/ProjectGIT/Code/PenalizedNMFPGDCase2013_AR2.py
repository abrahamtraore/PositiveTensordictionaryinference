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

import sys, getopt,os
dtype = 'torch.FloatTensor'

Kd = 10
opts, args = getopt.getopt(sys.argv[1:],"k:")
for opt,arg in opts:
    if opt == '-k':  # wasserstein or not
        Kd= int(arg)

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

It=60

Tensor_train=dtensor(MethodsTSPen.rescaling(Tensor_train,If,It))

Tensor_test=dtensor(MethodsTSPen.rescaling(Tensor_test,If,It))

vector_all_features=MethodsTSPen.Transform_tensor_into_featuresmatrix(Tensor_train)


seed=2  #Trueseed=2





C_values=np.power(10,np.arange(-3,4,2),dtype=float)
G_values=np.power(10,np.arange(-3,1,1),dtype=float)
Lambdainfo=np.power(10,np.arange(-4,1,1),dtype=float)#max=2
#The arrays Alpha and Lratio should have the same lenghts or at least one of them must be a length-1 array
Alpha=np.power(10,np.arange(-3,0,1),dtype=float)
Lratio=np.power(10,np.arange(-3,0,1),dtype=float)

#CD
#Kd=,Perf=
#PG
#Kd=,Perf=
 


pool=Pool(3)
pool_decision= True   
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
    print(matrix)
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
y_pred = clf.predict(Test_features)
y_pred_app = clf.predict(Training_features)



bc_test = sum(y_pred  == y_test)/len(y_test)
bc_app = sum(y_pred_app  == y_pred_app)/len(y_pred_app)

result,indexes,number_of_samples_per_class=MethodsTSPen.Recover_classes_all_labels(Test_features,y_test)
    #The function classificaton_rate_classes is designed in order to determine the performances per classe in order to compute the MAP
    #There are two outputs:
      #performances represent the proportion of well classified examples per class.
      #confusion_matrix represents the prediction of the classifier for each example in each class. It is usefull to understand which example has not been well classified
performances,confusionmatrix=MethodsTSPen.classificaton_rate_classes(result,clf)
print(performances,confusionmatrix)
print(np.mean(performances))
print(np.max(mean_score))
print('bc_app : ',bc_app)
print('bc_test: ',bc_test)
pool.terminate()  

filename = 'dcase13_Kd' + str(Kd) + 'Pool_' + str(pool_decision)
np.savez(filename, bc_app = bc_app, bc_test = bc_test, mean_score = mean_score, performances = performances)










