#!/usr/bin/env python3
"""
Created on Fri Sep 15 13:37:38 2017
# -*- coding: utf-8 -*-

@author: Traoreabraham
"""

import numpy as np

from multiprocessing import Pool

import h5py

import MethodsTSPenTestAB

import pdb

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold

from sklearn import svm


import sys

sys.path.append("/home/scr/etu/sin811/traorabr/sktensor/")

from sktensor import dtensor


BASE_PATH="/home/scr/etu/sin811/traorabr/RealToy/Realtoy.h5"
#BASE_PATH='/Users/Traoreabraham/Desktop/ProjectGit/Data/Realtoy.h5'
with h5py.File(BASE_PATH,'r') as hf:
    Tensor = np.array(hf.get('Tensordata')).astype('float32')
    labels = np.array(hf.get('labels')).astype('uint32')  


#import matplotlib.pyplot as plt
#plt.imshow(Tensor[0,:,:])
#plt.show()
#pdb.set_trace() 
eta=5

width=25

length=25

Tensor=dtensor(MethodsTSPenTestAB.rescaling(Tensor,width,length))





#K=2  #Perf=0.8179,TS=152
#K=3 #Perf=0.8578,TS=208
#K=4  #Perf=0.8375,TS=232
#K=5   #Perf=0.87,TS=240
#K=6    #Perf=0.87,TS=256
#K=7     #Perf=0.8464,TS=264
#K=8      #Perf=0.8624,TS=264
#K=9       #Perf=0.875,TS=272
#K=10       #Perf=0.875,TS=272
#K=11        #Perf=0.843,TS=280
#K=13         #Perf=0.8916,TS=288






seed=2




#First strategy: we keep only the last
#MethodsCD
#K=2     #Perf=0.916,TS=152

#K=3     #Perf=0.928125,TS=208

#K=7     #Perf=0.9428,TS=264

#K=9     #Perf=0.95,TS=272

#K=13    #Perf=0.975,TS=288


#MethodsPG
#K=2     #Perf=0.887,TS=152

#K=3     #Perf=0.9140,TS=208

#K=7     #Perf=0.9178,TS=264

#K=9     #Perf=0.955,TS=272

#K=13    #Perf=0.975,TS=288







#Second strategy: we take all the perf
K=10
#MethodsPG
#K=2     #Perf=0.896,TS=

#K=3     #Perf=0.9246,TS=

#K=4     #Perf=0.9265,TS=

#K=5     #Perf=0.9215,TS=

#K=6     #Perf=0.9325,TS=

#K=7     #Perf=0.9296,TS=

#K=8     #Perf=0.9276,TS=

#K=9     #Perf=0.9315,TS=

#K=10     #Perf=0.933,TS

#K=11    #Perf=,TS=

#K=12    #Perf=,TS=

#K=13    #Perf=,TS=

skf=StratifiedKFold(n_splits=K,random_state=seed,shuffle=True)#skf=StratifiedKFold(n_splits=5,random_state=17,shuffle=True)  

k=-1

firstsize=np.array(Tensor.shape,dtype=int)[1]

secondsize=np.array(Tensor.shape,dtype=int)[2]
    
seeds=np.array([1,2,3,4,5],dtype=int)

Perf=np.zeros(5) 

for i in range(5):
    
   np.random.seed(seeds[i])
    
   k=-1
   
   GPerf=np.zeros(K)
    
   TensorTFR=Tensor+eta*np.random.rand(400,width,length)     
    
   vector_all_features=MethodsTSPenTestAB.Transform_tensor_into_featuresmatrix(TensorTFR)
   
   Sizes_list=[]
#   for trainvalid_index, test_index in skf.split(vector_all_features,labels):
#       
#       k=k+1
#       
#       matrix_trainvalid,matrix_test=vector_all_features[trainvalid_index],vector_all_features[test_index]
#    
#       y_trainvalid,y_test=labels[trainvalid_index],labels[test_index]
#       
#       skf1=StratifiedKFold(n_splits=10,random_state=seed,shuffle=True)
#    
#       for train_index, test_index in skf1.split(matrix_trainvalid,y_trainvalid):
#    
#          matrix_train,matrix_valid=matrix_trainvalid[train_index],matrix_trainvalid[test_index]
#         
#          y_train,y_valid=y_trainvalid[train_index],y_trainvalid[test_index]
#    
#       Tensor_train=MethodsTSPenTestAB.Transform_featuresmatrix_into_tensor(matrix_train,firstsize,secondsize)
#       Tensor_test=MethodsTSPenTestAB.Transform_featuresmatrix_into_tensor(matrix_test,firstsize,secondsize)
#       Tensor_valid=MethodsTSPenTestAB.Transform_featuresmatrix_into_tensor(matrix_valid,firstsize,secondsize)
#       print(Tensor_train.shape)



   for trainvalid_index, test_index in skf.split(vector_all_features,labels):
       
       k=k+1
       
       matrix_trainvalid,matrix_test=vector_all_features[trainvalid_index],vector_all_features[test_index]
       
       y_trainvalid,y_train=labels[trainvalid_index],labels[test_index]
              
       skf1=StratifiedKFold(n_splits=20,random_state=seed,shuffle=True)
    
       for train_index, test_index in skf1.split(matrix_trainvalid,y_trainvalid):
    
          matrix_train,matrix_valid=matrix_trainvalid[train_index],matrix_trainvalid[test_index]
         
          y_test,y_valid=y_trainvalid[train_index],y_trainvalid[test_index]
          
       Tensor_train=MethodsTSPenTestAB.Transform_featuresmatrix_into_tensor(matrix_test,firstsize,secondsize)
       Tensor_test=MethodsTSPenTestAB.Transform_featuresmatrix_into_tensor(matrix_train,firstsize,secondsize)
       Tensor_valid=MethodsTSPenTestAB.Transform_featuresmatrix_into_tensor(matrix_valid,firstsize,secondsize)
       print(Tensor_train.shape)
       print(y_train.shape)
      
       
       
       #print("Point II")
       #We perform the Cross_validation to determine the best values for the hyperparameters
       #The definition of kf,kt amounts to the definition Jf and Jt because in our framework, Jf=nbclasses*kf. The same definition holds for Jt
    
       C_values=np.power(10,np.linspace(-3,4,4),dtype=float)
       G_values=np.power(10,np.linspace(-3,4,4),dtype=float)
       Lambdainfo=np.power(10,np.linspace(-4,1,1),dtype=float)
       Alpha=np.power(10,np.linspace(-4,1,1),dtype=float)
       Lratio=np.power(10,np.linspace(-3,1,1),dtype=float)
       
       #C_values=np.power(10,np.arange(-3,4,2),dtype=float)
       #G_values=np.power(10,np.arange(-5,3,1),dtype=float)
       #Lambda_info=np.power(10,np.arange(-3,4,2),dtype=float)
       #Alpha= [0.01,0.1,1,10,20]
       #Lratio=[0.001,0.01,0.1,0.25,0.5,0.75]

       Kd=32   #16
       nbclasses=8
       pool=Pool(3)
       pool_decision=False
       #C_svm,G_svm,hyperparam,matrix=Methods.Cross_valNMFPenalized(Tensor_train,y_train,C_values,G_values,Hyper_values,lamb,Kd,nbclasses,pool)
       #C_svm,G_svm,hyperparam,l1_s,l2_s,matrix=MethodsTSPen.Cross_valNMFPenalized(Tensor_valid,y_valid,C_values,G_values,Hyper_values,L1_reg_active,L2_reg_dict,Kd,nbclasses,pool)
    
       C_svm,G_svm,lambdainfo,lratio,alpha,matrix=MethodsTSPenTestAB.Cross_valNMF_PG(Tensor_valid,y_valid,C_values,G_values,Lambdainfo,Alpha,Lratio,Kd,nbclasses,pool,pool_decision)
       print("Point III")
       #We perform the decomposition of the training tensor
       #The decomposition yields:
         #The error related to each updating error_list;
         #The temporal and spectral dictionary components A_f and A_t;
         #The number of iterations and the activation coefficients G;
       #We dimension purpose, we can reduce the size of the tensor to be decomposed.This can be done in the following way:
         #Tensor_train=dtensor(Preprocessing.rescaling(Tensor_train,If,It)) where If and It correspond to the desired sizes.
       array_of_examples_numbers=MethodsTSPenTestAB.Determine_the_number_of_examples_per_class(y_train)
    
       #D=MethodsTSPen.NMF_penalized_CD(Tensor_train,lambdainfo,alpha,lratio,Kd,nbclasses,array_of_examples_numbers)
       D=MethodsTSPenTestAB.NMF_penalized_PG(Tensor_train,lambdainfo,Alpha,Lratio,Kd,nbclasses,array_of_examples_numbers)
       
       #We define the training features. They are obtained by vectorizing the matrices G[k,:,:]
       #Training_features=Methods.Training_features_extraction(G)
       #Training_features=Methods.Training_features_extraction(G)
       #Training_features=MethodsTSPen.Features_extractionPenalizedNMF(Tensor_train,D,pool)
    
       Training_features=MethodsTSPenTestAB.Feature_extraction_process(Tensor_train,D,Alpha,lratio,pool,pool_decision)

       scaler=StandardScaler()
       Training_features=scaler.fit_transform(Training_features)

       #C_values=np.power(10,np.linspace(-2,5,4),dtype=float)
       #G_values=np.power(10,np.linspace(-2,2,5),dtype=float)


       #We define the test features
       #If we resize the training tensor, it is obligatory to resize the test tensor for dimensionality coherence
         #This is done via the rescaling function defined in Preprocessing
       #Test_features=MethodsTSPen.Features_extractionPenalizedNMF(Tensor_test,D,pool)
    
       Test_features=MethodsTSPenTestAB.Feature_extraction_process(Tensor_test,D,Alpha,lratio,pool,pool_decision)
       Test_features=scaler.transform(Test_features)
       #We define the classifier and perform the classification task
    
       clf=svm.SVC(C=C_svm,gamma=G_svm,kernel='rbf')
       clf.fit(Training_features,y_train)

       #Recover_classes_all_labels is designed to recover the classes present in an array with the help of the labels. 
       #There are three outputs:
         #result corresponds to the examples followed by their classes label
         #indexes contains the classes present in the array
         #number_of_samples_per_class yields the number of examples per class.
       result,indexes,number_of_samples_per_class=MethodsTSPenTestAB.Recover_classes_all_labels(Test_features,y_test)
       #The function classificaton_rate_classes is designed in order to determine the performances per classe in order to compute the MAP
       #There are two outputs:
        #performances represent the proportion of well classified examples per class.
        #confusion_matrix represents the prediction of the classifier for each example in each class. It is usefull to understand which example has not been well classified
       performances,confusionmatrix=MethodsTSPenTestAB.classificaton_rate_classes(result,clf)
       GPerf[k]=np.mean(performances)  
       pool.terminate()
   Perf[i]=np.mean(GPerf)
       #print(Perf[i])      
#       if(k==4):
#         with h5py.File('/Users/Traoreabraham/Desktop/ResultsComparisonNMFPG/CaseUnderCompleteWithNoPool'+str(Kd)+'eta20/Parameters'+str(i)+'.'+'h5' ,'w') as hf:
#           hf.create_dataset('Dictionaryatoms', data=D)
#           hf.create_dataset('Performances', data=np.mean(Perf))
#           hf.create_dataset('SVMPenaltyparam', data=C_svm)
#           hf.create_dataset('SVMVariance', data=G_svm)
#print(Perf)
print(Perf)
print(np.mean(Perf))
pdb.set_trace()
#New spectrograms 
#Max-pooling
#TS=24,          Perf=24.09%
#TS=64,          Perf=26.75%     
#TS=104   (2,2), Perf=27.95%
#TS=152   (2,4), Perf=28.35%
#TS=202   (3,4), Perf=29.84%
#TS=267   (9,4), Perf=33.35%
#TS=328   (10,10), Perf=30.5%

#Mean-pooling
#TS=24,          Perf=30.54%
#TS=64,          Perf=36.36%     
#TS=104   (2,2), Perf=43.6%
#TS=152   (2,4), Perf=70.7%
#TS=202   (3,4), Perf=60.87%
#TS=267   (9,4), Perf=80.35%
#TS=328   (10,10), Perf=56.65%

#Vectorization
#TS=24,          Perf=19.90%
#TS=64,          Perf=25.5%     
#TS=104   (2,2), Perf=23.35%
#TS=152   (2,4), Perf=13.05%
#TS=202   (3,4), Perf=16.98%
#TS=267   (9,4), Perf=19.76%
#TS=328   (10,10), Perf=23.4%


#ONTDL
#TS=24,          Perf=59.74% 
#TS=64,          Perf=88.35%   
#TS=104   (2,2), Perf=96.65%  
#TS=152   (2,4), Perf=96.85% 
#TS=202   (3,4), Perf=94.20% 
#TS=267   (9,4), Perf=98.89% pending in nnTensorbis











#Former spedctrograms
#With max_pooling
#TS=24, Perf=72.52%
#TS=64, Perf=79.42%
#TS=104, Perf=84.55%
#TS=152, Perf=79.45%
#TS=202, Perf=86.55%
#TS=267, Perf=86.51%
#TS=328, Perf=84.7%

#With mean
#TS=24, (20,20), Perf=78.49%
#TS=64, (7,20) , Perf=87.55%
#TS=104  (2,2), Perf=89.05%
#TS=152, (2,4) Perf=90.35%
#TS=202, (3,4) Perf=91.89%
#TS=267, (9,4) Perf=92.62%
#TS=328, (10,10) Perf=91.45%
