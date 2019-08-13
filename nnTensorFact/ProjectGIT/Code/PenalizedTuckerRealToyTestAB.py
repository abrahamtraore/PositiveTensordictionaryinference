#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 18:08:34 2017

@author: Traoreabraham
"""



import numpy as np

from multiprocessing import Pool

import MethodsTSPenTestAB

import pdb

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold

from sklearn import svm

import h5py

import sys

sys.path.append("/home/scr/etu/sin811/traorabr/sktensor/")

from sktensor import dtensor

#BASE_PATH="/home/scr/etu/sin811/traorabr/RealToy/Realtoy.h5"
BASE_PATH='/Users/Traoreabraham/Desktop/ProjectGit/Data/Realtoy.h5'
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

Tensor=dtensor(MethodsTSPenTestAB.rescaling(Tensor,width,length))+eta*np.random.rand(400,width,length)

Tensor=Tensor
labels=labels
seed=2

#K1=20,K2=4 for Tranining samples=24
#K1=7,K2=4 for Training samples=64

K=7
skf=StratifiedKFold(n_splits=K,random_state=seed,shuffle=True)#skf=StratifiedKFold(n_splits=5,random_state=17,shuffle=True)   

firstsize=np.array(Tensor.shape,dtype=int)[1]

secondsize=np.array(Tensor.shape,dtype=int)[2]


seeds=np.array([1,2,3,4,5],dtype=int)

Perf=np.zeros(5)

for i in range(5):
    
   np.random.seed(seeds[i])
  
   TensorTFR=dtensor(MethodsTSPenTestAB.rescaling(Tensor,width,length))+eta*np.random.rand(400,width,length)
   
   vector_all_features=MethodsTSPenTestAB.Transform_tensor_into_featuresmatrix(TensorTFR)
  
   k=-1
  
   GPerf=np.zeros(K)
      
   countdown=0
   
   for trainvalid_index, test_index in skf.split(vector_all_features,labels):
    
      k=k+1
      
      
      matrix_trainvalid,matrix_test=vector_all_features[trainvalid_index],vector_all_features[test_index]
    
      y_trainvalid,y_test=labels[trainvalid_index],labels[test_index]
    
      skf1=StratifiedKFold(n_splits=4,random_state=seed,shuffle=True)
    
      for train_index, valid_index in skf1.split(matrix_trainvalid,y_trainvalid):
    
          matrix_train,matrix_valid=matrix_trainvalid[train_index],matrix_trainvalid[valid_index]
         
          y_train,y_valid=y_trainvalid[train_index],y_trainvalid[valid_index]
         
      print("Point I")

    
      Tensor_train=MethodsTSPenTestAB.Transform_featuresmatrix_into_tensor(matrix_train,firstsize,secondsize)
      Tensor_test=MethodsTSPenTestAB.Transform_featuresmatrix_into_tensor(matrix_test,firstsize,secondsize)
      Tensor_valid=MethodsTSPenTestAB.Transform_featuresmatrix_into_tensor(matrix_valid,firstsize,secondsize)
      print(Tensor_train.shape)
      print(Tensor_test.shape)
      pdb.set_trace()
   
#   for trainvalid_index, test_index in skf.split(vector_all_features,labels):
#       
#      k=k+1
#       
#      matrix_trainvalid,matrix_test=vector_all_features[trainvalid_index],vector_all_features[test_index]
#       
#      y_trainvalid,y_train=labels[trainvalid_index],labels[test_index]
#             
#      skf1=StratifiedKFold(n_splits=20,random_state=seed,shuffle=True)
#    
#      for train_index, test_index in skf1.split(matrix_trainvalid,y_trainvalid):
#    
#          matrix_train,matrix_valid=matrix_trainvalid[train_index],matrix_trainvalid[test_index]
#         
#          y_test,y_valid=y_trainvalid[train_index],y_trainvalid[test_index]
#          
#      Tensor_train=MethodsTSPenTestAB.Transform_featuresmatrix_into_tensor(matrix_test,firstsize,secondsize)
#      Tensor_test=MethodsTSPenTestAB.Transform_featuresmatrix_into_tensor(matrix_train,firstsize,secondsize)
#      Tensor_valid=MethodsTSPenTestAB.Transform_featuresmatrix_into_tensor(matrix_valid,firstsize,secondsize)
#      print(Tensor_train.shape)
#      print(y_train.shape)

      #We perform the Cross_validation to determine the best values for the hyperparameters
      #The definition of kf,kt amounts to the definition Jf and Jt because in our framework, Jf=nbclasses*kf. The same definition holds for Jt   
      C_values=np.power(10,np.linspace(-3,4,4),dtype=float)
      G_values=np.power(10,np.linspace(-3,4,4),dtype=float)
      Lambda_info=np.power(10,np.linspace(-4,1,1),dtype=float)
      LambdaG=np.power(10,-4,dtype=float)*np.power(10,np.linspace(-3,3,1),dtype=float)
      Lambda_Af=(1-np.power(10,-3,dtype=float))*np.power(10,np.linspace(-4,3,1),dtype=float)
      Lambda_At=(1-np.power(10,-3,dtype=float))*np.power(10,np.linspace(-4,3,1),dtype=float)
      Lambda_Bf=(1-np.power(10,-3,dtype=float))*np.power(10,np.linspace(-4,3,1),dtype=float)
      Lambda_Bt=(1-np.power(10,-3,dtype=float))*np.power(10,np.linspace(-4,3,1),dtype=float)
       
      T=2
      nbclasses=8
      kt=4 #1
      kf=4 #1
    
      tol=np.power(10,-8,dtype=float)
      maximum_iterations=10
      pool=Pool(3)
      InitStrat="Multiplicative"#"Tucker2HOSVD"
      parallel_decision=True
      #C_svm,G_svm,lambdainfo,lambdaG,lambdaAf,lambdaAt,lambdaBf,lambdaBt,matrix=MethodsTSPenTestAB.Cross_valTSPen(Tensor_valid,y_valid,C_values,G_values,Lambda_info,LambdaG,Lambda_Af,Lambda_At,Lambda_Bf,Lambda_Bt,T,nbclasses,kf,kt,maximum_iterations,tol,InitStrat,parallel_decision,pool)
     
      C_svm,G_svm,lambdainfo,lambdaG,lambdaAf,lambdaAt,lambdaBf,lambdaBt,matrix=MethodsTSPenTestAB.Cross_valTSPen(Tensor_valid,y_valid,C_values,G_values,Lambda_info,LambdaG,Lambda_Af,Lambda_At,Lambda_Bf,Lambda_Bt,T,nbclasses,kf,kt,maximum_iterations,tol,InitStrat,parallel_decision,pool)
      G,A_f,A_t,error_list,nb_iter=MethodsTSPenTestAB.PenalizedTuckerTSPen(Tensor_train,y_train,nbclasses,kf,kt,maximum_iterations,tol,InitStrat,lambdainfo,lambdaG,lambdaAf,lambdaAt,lambdaBf,lambdaBt,T,parallel_decision,pool)
      with h5py.File("/Users/Traoreabraham/Desktop/SImages267.h5",'w') as hf:        
           hf.create_dataset('Spectraldictionaries'+str(seeds[i]), data = A_f)      
      pdb.set_trace()
      #We define the training features. They are obtained by vectorizing the matrices G[k,:,:] and normalized.
    
      #Training_features=MethodsTSPen.Test_features_extraction(Tensor_train,A_f,A_t)
      penalty=np.power(10,-4,dtype=float)*np.power(10,-3,dtype=float)
      Training_features=MethodsTSPenTestAB.decomposition_all_examples_parallelized(Tensor_train,np.kron(A_f,A_t),penalty,pool)
      
      scaler=StandardScaler()

      Training_features=scaler.fit_transform(Training_features)

      #We define the test features and normalize them
      #Test_features=MethodsTSPen.Test_features_extraction(Tensor_test,A_f,A_t)
      #Test_features=MethodsTSPenTestAB.decomposition_all_examples_parallelized(Tensor_test,np.kron(A_f,A_t),pool)
      Test_features=MethodsTSPenTestAB.decomposition_all_examples_parallelized(Tensor_test,np.kron(A_f,A_t),penalty,pool)
      
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
      #print(np.mean(performances))
      GPerf[k]=np.mean(performances)
      pool.terminate()
   #with h5py.File("/home/scr/etu/sin811/traorabr/ResultsTuckerReal/Classifparameters"+str(i)+".h5",'w') as hf:
        #hf.create_dataset('Spectraldictionaries'+str(i), data = A_f)
        #hf.create_dataset('Temporaldictionaries'+str(i), data = A_t)
        #hf.create_dataset('Activationcoefficients'+str(i), data=G)
        #hf.create_dataset('Performances'+str(i), data=np.mean(Perf))
        #hf.create_dataset('SVMPenaltyparam'+str(i), data=C_svm)
        #hf.create_dataset('SVMVariance'+str(i), data=G_svm)
   #print(np.mean(GPerf))
   Perf[i]=np.mean(GPerf)
#      if(k==4):
#         Importantlist.append(np.mean(Perf))
#         with h5py.File('/home/scr/etu/sin811/traorabr/ResultsTuckerReal/ResultsKf'+str(nbclasses*kf)+'eta20/Parameters'+str(i)+'.'+'h5' ,'w') as hf:
#         #with h5py.File("/home/scr/etu/sin811/traorabr/ClassificationArchive/Classifparameters.h5",'w') as hf:
#           hf.create_dataset('Spectraldictionaries'+str(seeds[i]), data = A_f)
#           hf.create_dataset('Temporaldictionaries'+str(seeds[i]), data = A_t)
#           hf.create_dataset('Activationcoefficients'+str(seeds[i]), data=G)
#           hf.create_dataset('Performances'+str(seeds[i]), data=np.mean(Perf))
#           hf.create_dataset('SVMPenaltyparam'+str(seeds[i]), data=C_svm)
#           hf.create_dataset('SVMVariance'+str(seeds[i]), data=G_svm)

print(Perf)
print(np.mean(Perf))
print(Tensor_train.shape)     
pdb.set_trace()
    
    
    