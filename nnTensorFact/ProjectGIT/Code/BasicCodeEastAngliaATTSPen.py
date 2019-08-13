#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 13:15:04 2017

@author: Traoreabraham
"""

import gc

from multiprocessing import Pool

import numpy as np

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

import MethodsTSPen

import pdb

from sklearn import svm

import logging

np.random.seed(1)

logging.basicConfig(filename='test_log.log',level=logging.DEBUG,format='%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s')

#A complete example of a toy classification problem

##We define the matrix data: 100 examples split into 10 classes whose rows correspond to the signal vectors
#filename="/home/scr/etu/sin811/traorabr/Data/EastAngliaDataSet"
##filename="/Users/Traoreabraham/Desktop/ProjectGit/Data/EastAngliaDataSet"
#Datafilename=os.listdir(filename) #corresponds to the data file names
#Data,labels=Preprocessing.split_all_the_data_and_stack(Datafilename,filename)
#sampling_rate=22050
#window_length=2000
#overlap_points=500
#grid_form="logscale" #can also be "logscale"
##We compute the TFR of all the examples.
##The TFR can be computed with regular grid or with logscale grid.
#   #If the parameter grid_form is "regular", the TFR is computed on regular grid.
#   #If the parameter grid_form is "logscale", the TFR is computed on logarithmic scale.
#print("First point")
#print(grid_form)
#TensorTFR=Methods.TFR_all_examples(Data,sampling_rate,window_length,overlap_points,grid_form)

#This is the adress where are stored the preprocessed data.



BASE_PATH="/home/scr/etu/sin811/traorabr/RawDataEastAnglia"

#This tensor contains the preprocessed data.
TensorTFR,labels=MethodsTSPen.load_preprocessed_data(BASE_PATH)
print(TensorTFR.shape)

#We split the global tensor into training, test and validation tensors
#The ratio are considered on the total nimber of examples
#seed1,seed2 intervene in the splitting tool scikit to ensure reproductibility

seed=2   

#Values at the end:
#C_svm,G_svm,hyperparam=(56.23; 0.02; 0.1 x 0.0005)

#seed=2 value for direct projection


[firstsize,secondsize]=np.array(TensorTFR.shape,dtype=int)[1:3]

vector_all_features=MethodsTSPen.Transform_tensor_into_featuresmatrix(TensorTFR)

K=5   #4

skf=StratifiedKFold(n_splits=K,random_state=seed,shuffle=True)#skf=StratifiedKFold(n_splits=5,random_state=17,shuffle=True)

Perf=np.zeros(K)   

k=-1

for train_index, test_index in skf.split(vector_all_features,labels):
    
    matrix_train, matrix_test = vector_all_features[train_index],vector_all_features[test_index]
    
    y_train, y_test=labels[train_index],labels[test_index]
    
    Tensor_train=MethodsTSPen.Transform_featuresmatrix_into_tensor(matrix_train,firstsize,secondsize)
    
    Tensor_test=MethodsTSPen.Transform_featuresmatrix_into_tensor(matrix_test,firstsize,secondsize)
    
    #Tensor_train,y_train,Tensor_test,y_test,Tensor_valid,y_valid=Methods.train_test_validation_tensors(TensorTFR,labels,ratio_test,ratio_valid,seed1,seed2)

    #We perform the Cross_validation to determine the best values for the hyperparameters
    #The definition of kf,kt amounts to the definition Jf and Jt because in our framework, Jf=nbclasses*kf. The same definition holds for Jt
    #C_values=np.power(10,np.linspace(-2,2,2),dtype=float)
    #G_values=np.power(10,np.linspace(-1,1,2),dtype=float)
    #Hyper_values=np.power(10,np.linspace(-1,1,2),dtype=float)
    nbclasses=10
    kt=6  #2
    kf=6  #2
    tol=np.power(10,-8,dtype=float)
    maximum_iterations=4
    InitStrat="Multiplicative" #"Tucker2HOSVD"
    C_values=np.power(10,np.linspace(-5,4,5),dtype=float)
    G_values=np.power(10,np.linspace(-4,3,4),dtype=float)
    LambdaG=np.power(10,np.linspace(-1,3,1),dtype=float)
    Lambda_info=np.power(10,np.linspace(-4,1,3),dtype=float)
    Lambda_Af=np.power(10,np.linspace(-4,1,1),dtype=float)
    Lambda_At=np.power(10,np.linspace(-4,1,1),dtype=float)
    Lambda_Bf=np.power(10,np.linspace(-4,1,1),dtype=float)
    Lambda_Bt=np.power(10,np.linspace(-4,1,1),dtype=float)
    
#    C_values=np.power(10,np.linspace(-5,4,5),dtype=float)
#    G_values=np.power(10,np.linspace(-4,3,4),dtype=float)
#    Lambda_info=np.power(10,np.linspace(-4,1,1),dtype=float)
#    Lambda_Af=np.power(10,np.linspace(-4,1,1),dtype=float)
#    Lambda_At=np.power(10,np.linspace(-4,1,1),dtype=float)
#    Lambda_Bf=np.power(10,np.linspace(-4,1,1),dtype=float)
#    Lambda_Bt=np.power(10,np.linspace(-4,1,1),dtype=float)    
    T=4
    parallel_decision=True
    pool=Pool(3)
    #C_svm,G_svm,lambdainfo,lambdaG,lambdaAf,lambdaAt,lambdaBf,lambdaBt,matrix=MethodsTSPen.Cross_valGD(Tensor_train,y_train,C_values,G_values,Lambda_info,LambdaG,Lambda_Af,Lambda_At,Lambda_Bf,Lambda_Bt,T,nbclasses,kf,kt,maximum_iterations,tol,InitStrat,pool)


    C_svm,G_svm,lambdainfo,lambdaG,lambdaAf,lambdaAt,lambdaBf,lambdaBt,matrix=MethodsTSPen.Cross_valTSPen(Tensor_train,y_train,C_values,G_values,Lambda_info,LambdaG,Lambda_Af,Lambda_At,Lambda_Bf,Lambda_Bt,T,nbclasses,kf,kt,maximum_iterations,tol,InitStrat,parallel_decision,pool)
    
    print(C_svm,G_svm,lambdainfo)
    #We perform the decomposition of the training tensor
    #The decomposition yields:
      #The error related to each updating error_list;
      #The temporal and spectral dictionary components A_f and A_t;
      #The number of iterations and the activation coefficients G
    #We dimension purpose, we can reduce the size of the tensor to be decomposed.This can be done in the following way:
        #Tensor_train=dtensor(Preprocessing.rescaling(Tensor_train,If,It)) where If and It correspond to the desired sizes.    
    
    G,A_f,A_t,error_list,nb_iter=MethodsTSPen.PenalizedTuckerTSPen(Tensor_train,y_train,nbclasses,kf,kt,maximum_iterations,tol,InitStrat,lambdainfo,lambdaG,lambdaAf,lambdaAt,lambdaBf,lambdaBt,T,parallel_decision,pool)
    
    #We define the training features. They are obtained by vectorizing the matrices G[k,:,:]
    #Training_features=Methods.Training_features_extraction(G)
    Training_features=MethodsTSPen.Test_features_extraction(Tensor_train,A_f,A_t)

    #Training_features=Methods.Test_features_extraction(Tensor_train,A_f,A_t)
    scaler=StandardScaler()
    Training_features=scaler.fit_transform(Training_features)

    #C_values=np.power(10,np.linspace(-2,5,4),dtype=float)
    #G_values=np.power(10,np.linspace(-2,2,5),dtype=float)


    #We define the test features
     #If we resize the training tensor, it is obligatory to resize the test tensor for dimensionality coherence
     #This is done via the rescaling function defined in Preprocessing
    Test_features=MethodsTSPen.Test_features_extraction(Tensor_test,A_f,A_t)
    Test_features=scaler.transform(Test_features)
    #We define the classifier and perform the classification task
    clf=svm.SVC(C=C_svm,gamma=G_svm,kernel='rbf')
    clf.fit(Training_features,y_train)

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
    k=k+1
    Perf[k]=np.mean(performances)
    print(Perf[k])
#    fig = plt.figure()
#    ax = fig.add_subplot(1,1,1)
#    ax.set_aspect('equal')
#    factor_basis=matrixscore[:,:,0]
#    plt.imshow(factor_basis, origin="lower", aspect="auto",cmap='jet', interpolation="none")
#    plt.colorbar()
#    plt.show()    
    #pdb.set_trace()
    del Tensor_train
    del Tensor_test
    del G
    del Training_features
    del Test_features
    del confusionmatrix
    del matrix_train
    del matrix_test
    gc.collect()  
print(Perf)
pdb.set_trace()    
#   import matplotlib.pylab as plt
#   fig = plt.figure()
#   ax = fig.add_subplot(1,1,1)
#   ax.set_aspect('equal')
#   ax.set_xticklabels([])
#   ax.set_yticklabels([])
 
#   #factor_basis=G[20:30,:,:].mean(axis=0)
#   factor_basis=matrixscore[:,:,0]
#   plt.imshow(factor_basis, origin="lower", aspect="auto",cmap='jet', interpolation="none")
#   plt.colorbar()
#   plt.show()