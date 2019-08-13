#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:54:23 2017

@author: Traoreabraham
"""

# -*- coding: utf-8 -*-
# This is my first idea to tackle my problem
#from scipy.fftpack import fft, ifft

from multiprocessing import Pool 
 
from sklearn.decomposition import NMF

import sys

#sys.path.append('/Users/Traoreabraham/Desktop/ProjectGit/') this works very well

sys.path.append("/home/scr/etu/sin811/traorabr/")

#/home/scr/etu/sin811/traorabr/Code/Methods.py

#"/home/scr/etu/sin811/traorabr/Code/tempEA.py"

from sktensor import dtensor

import random

random.seed(2)

import numpy as np

import scipy

def Periodicstep(X,A_old,Bm,derivative,lamb,a,b):
    r=(np.sqrt(5)-1)/2
    A=a
    B=b
    L=B-A
    lambda1=A+r*r*L
    lambda2=A+r*L
    nb_iter=0
    F1=1
    F2=0
    Lambda=0
    while(nb_iter<5):
      nb_iter=nb_iter+1
      F1=np.linalg.norm(X-np.dot(np.maximum(A_old-lambda1*derivative,0),Bm))**2+(lamb/2)*np.linalg.norm(np.maximum(A_old-lambda1*derivative,0))**2
      F2=np.linalg.norm(X-np.dot(np.maximum(A_old-lambda2*derivative,0),Bm))**2+(lamb/2)*np.linalg.norm(np.maximum(A_old-lambda2*derivative,0))**2       
      if (F1>F2):
          A=lambda1
          lambda1=lambda2
          L=B-A
          lambda2=A+r*L
      else:
          B=lambda2
          lambda2=lambda1
          L=B-A
          lambda1=A+r*r*L          
      if((L<0.001) or nb_iter>=5):
          Lambda=(B+A)/2
          break
    return Lambda



def Factors_updating(X,B,epsilon,max_iter,lamb,T): #the function is used to mimimize ||X-AB||^2+(lambda/2)*||A||^2 w.r.t A
    K=np.array(B.shape,dtype=int)[0]
    step=np.power(10,-3,dtype=float)
    model= NMF(n_components=K, init='random', random_state=0)
    A=model.fit_transform(X)
    A_old=np.zeros(A.shape)
    result=np.zeros(A.shape)
    previous_error=0 
    error=np.linalg.norm(X-np.dot(A,B))**2+(lamb/2)*np.linalg.norm(A)**2
    nb_iter=0
    a=np.power(10,-5,dtype=float)
    b=np.power(10,-2,dtype=float)
    while(nb_iter<=max_iter):
        nb_iter=nb_iter+1
        previous_error=error
        A_old=A
        derivative=-2*np.dot((X-np.dot(A_old,B)),np.transpose(B))+lamb*A
        if(nb_iter%T==0):
            step=Periodicstep(X,A_old,B,derivative,lamb,a,b)
        A=np.maximum(A_old-step*derivative,0)
        error=np.linalg.norm(X-np.dot(A,B))**2+(lamb/2)*np.linalg.norm(A)**2
        result=A
        if (previous_error-error<epsilon):
            result=A_old
            break
    return result 

def Replace_zero_elements_3way_array(A,epsilon):
    zero_indices=np.argwhere(A==0)
    test=A
    for zero_position in zero_indices:
       row=zero_position[0]
       column=zero_position[1]
       sample=zero_position[2]
       test[row,column,sample]=epsilon
    return test 

def Replace_zero_elements(A,epsilon):
    result=np.maximum(A,epsilon)
    return result


def Core_multiplicative_initialization(X,G,list_of_factor_matrices):
    mode=-1
    numerator=X
    denumerator=G
    epsilon=np.power(10,-3,dtype=float)
    for factor_matrix in list_of_factor_matrices:
        mode=mode+1
        numerator=numerator._ttm_compute(np.transpose(factor_matrix),mode,False)
        denumerator=numerator._ttm_compute(np.dot(np.transpose(factor_matrix),factor_matrix),mode,False)
    denumerator=Replace_zero_elements_3way_array(denumerator,epsilon)
    G=G*(numerator/denumerator)
    return G

def Multiplicative_initialization_NTD2(X,Coretensorsize):
    
    [K,If,It]=np.array(np.shape(X),dtype=int)
    
    epsilon=0.001
    
    A1=np.eye(K,Coretensorsize[0])
    #Init=NMF_initialization(X.unfold(1),Coretensorsize[1])
    
    model = NMF(n_components=Coretensorsize[1], init='random', random_state=0)    
    W = model.fit_transform(X.unfold(1))    
    A2=W
    #Init=NMF_initialization(X.unfold(2),Coretensorsize[2])
    model = NMF(n_components=Coretensorsize[2], init='random', random_state=0)
    W = model.fit_transform(X.unfold(2))   
    A3=W    
    G=dtensor(np.random.rand(Coretensorsize[0],Coretensorsize[1],Coretensorsize[2]))
    for i in range(3):
       S=np.dot(G.unfold(1),np.transpose(np.kron(A3,A1)))
       numerator=np.dot(X.unfold(1),np.transpose(S))
       denumerator=np.dot(A2,np.dot(S,np.transpose(S)))
       denumerator=Replace_zero_elements(denumerator,epsilon)
       A2=A2*(numerator/denumerator)       
       S=np.dot(G.unfold(2),np.transpose(np.kron(A2,A1)))
       numerator=np.dot(X.unfold(2),np.transpose(S))
       denumerator=np.dot(A3,np.dot(S,np.transpose(S)))
       denumerator=Replace_zero_elements(denumerator,epsilon)
       A3=A3*(numerator/denumerator)
       
       list_of_factors_matrices=[A1,A2,A3]
       G=Core_multiplicative_initialization(X,G,list_of_factors_matrices)
    return G,list_of_factors_matrices[1:3]


#This function is used to vectorize a tensor. The vectorization process will be explained in a pdf
def vectorization(X):
    vecX=np.reshape(np.array(X),np.size(np.array(X)))
    size=np.array(X.shape,dtype=int)
    K=size[0]
    nrows=size[1]
    ncolumns=size[2]
    for i in range(K):
      matrix_of_interest=X[i,:,:]
      vecX[i*(nrows*ncolumns):(i+1)*(nrows*ncolumns)]=np.reshape(matrix_of_interest,np.size(matrix_of_interest))
    return vecX

def matrix_completion(A):
    [nbrows,nbcols]=np.array(np.shape(A),dtype=int)
    result=np.copy(A)
    for j in range(nbcols):
        if(np.linalg.norm(result[:,j])==0):
            result[:,j]=np.random.rand(nbrows)
    return result

#This function is used to normalize the columns of a matrix if they are not zero-norm vectors
def normalize_columns(A):
    ncols=np.array(np.shape(A),dtype=int)[1]
    B=np.zeros(np.shape(A))
    for j in range(ncols):
        if(np.linalg.norm(A[:,j])!=0):
           B[:,j]=A[:,j]/np.linalg.norm(A[:,j])
    return B


#def define_class_information(nbexamples,nbclasses,Jf,Jt,classnumbers):
#    C=dtensor(np.zeros((nbexamples,Jf,Jt)))
#    M=nbclasses#In the rationale, M should obligatory be a divisor of Jf and Jt
#    for k in range(nbexamples):
#      C[k,classnumbers[k]*(Jf/M):(classnumbers[k]+1)*(Jf/M),classnumbers[k]*(Jt/M):(classnumbers[k]+1)*(Jt/M)]=np.ones((Jf/M,Jt/M))
#    return C

#This function is used to split the problem min|AX-B| with respect to X into independant problems
def NMF_decoupling(A,B):
    size_B=np.array(B.shape,dtype=int)
    size_A=np.array(A.shape,dtype=int)
    nrows_sol=size_A[1]
    ncol_sol=size_B[1]
    result=np.zeros((nrows_sol,ncol_sol))
    for j in range(ncol_sol):
        result[:,j],residual=scipy.optimize.nnls(A,B[:,j])   
    return result

#This function is used to solve one positive least squares problem and paves the way to the paralellization of the problem
def positive_least_squares_single(args) : 
    solution=scipy.optimize.nnls(args[0],args[1][:,args[2]])
    #print(solution)
    #result[n]=solution
    return solution

#This function is used to parallelize the problem min|AX-B| with respect to X
def NMF_decoupling_parallelized(A,B,pool):
    #start_time=time.time()
    size_A=np.array(A.shape,dtype=int)
    size_B=np.array(B.shape,dtype=int)
    nrows_sol=size_A[1]
    ncol_sol=size_B[1]
    solution=np.zeros((nrows_sol,ncol_sol)) 
    sol=pool.map(positive_least_squares_single,[[A,B,n] for  n in range(ncol_sol)])
    for i in range(len(sol)):
        solution[:,i]=sol[i][0]
    return  solution  
    
#This function is used to used to compute the Frobenius norm of the differences of two matrices 
def Error_estimation_factor(B,X,n,solution): 
   result=np.linalg.norm(np.dot(B,np.transpose(solution))-np.transpose(X.unfold(n)))
   return result
 
#This function is used to solve the problem which yields the core tensor
def decomposition_for_core_retrieving(X,U):
    [K,I2,I3]=np.array(X.shape,dtype=int)
    size_factor2=np.array(U[0].shape,dtype=int)
    J2=size_factor2[1]
    size_factor3=np.array(U[1].shape,dtype=int)
    J3=size_factor3[1]
    vecX=np.resize(X,np.size(X))
    vecG=np.zeros(K*J2*J3)
    beta=U[0]   
    number_of_factor_matrices=len(U)
    for length in range(number_of_factor_matrices-1):
        beta=np.kron(beta,U[length+1])
    for k in range(K):
         g_of_interest,residual=scipy.optimize.nnls(beta,vecX[k*(I2*I3):(k+1)*(I2*I3)])     
         vecG[k*(J2*J3):(k+1)*(J2*J3)]=g_of_interest
    G_result=np.reshape(vecG,(K,J2,J3))
    return G_result
  
def retrieving_a_small_part_of_the_core(args):
    [K,I2,I3]=np.array(args[0].shape,dtype=int)
    vecX=np.resize(args[0],np.size(args[0]))
    beta=[np.kron(args[1][0],args[1][1]),np.sqrt(args[1][5])*np.kron(args[1][2],args[1][3])][0]
    Vector=[vecX[args[2]*(I2*I3):(args[2]+1)*(I2*I3)],np.sqrt(args[1][5])*args[1][4]][0]
    g_of_interest,residual=scipy.optimize.nnls(beta,Vector)
    return g_of_interest

#This function is used to parallelize the code computation problem
def decomposition_for_core_retrieving_parallelized(X,U,pool):
    [K,I2,I3]=np.array(X.shape,dtype=int)
    size_factor2=np.array(U[0].shape,dtype=int)
    J2=size_factor2[1]
    size_factor3=np.array(U[1].shape,dtype=int)
    J3=size_factor3[1]
    vecG=np.zeros(K*J2*J3)
    vecG=pool.map(retrieving_a_small_part_of_the_core,[ [X,U,k] for k in range(K)])
    G_result=dtensor(np.reshape(vecG,(K,J2,J3)))
    return G_result

def Core_retrieving(X,U,pool,hyperparam,vecInfotensor,parallel_decision):
    if(parallel_decision==True):
       #args=[A2_old,A3_old,B2_old,B3_old,vecInfotensor,hyperparam]       
       #G_new=decomposition_for_core_retrieving_parallelized(X,args,pool)
       args=[U[1],U[2],U[4],U[5],vecInfotensor,hyperparam]
       G=decomposition_for_core_retrieving_parallelized(X,args,pool)
       return G
    if(parallel_decision==False):
       G=decomposition_for_core_retrieving(X,U)
       return G
#This function is used to multiply the a 3-order tensor with 3 matrices    
def Product_with_factors(coretensor,factor_list):    
    approxim=np.copy(coretensor)    
    approxim=dtensor(approxim)    
    mode=-1    
    for factor_matrix in factor_list:
        mode=mode+1
        approxim=approxim._ttm_compute(factor_matrix,mode,False)        
    return approxim

#This function is used to compute the TUcker-2 algorithm with the first latent factor fixed to identity
def Tucker2HOSVD(X,Coretensorsize,N,m,decision_to_compute_the_core):   
    factor_matrices_list=[] 
    for i in range(N):
        if (i!=m):
           modematrix=X.unfold(i)
           Umodematrix,Emodematrix,Vmodematrix=np.linalg.svd(modematrix,full_matrices=False)
           factor_matrix=Umodematrix[:,0:Coretensorsize[i]]
           factor_matrices_list.append(factor_matrix)
    if (decision_to_compute_the_core==False):
       return factor_matrices_list 
    if(decision_to_compute_the_core==True):
        coretensor=X
        mode=0
        for factormatrice in factor_matrices_list:
            mode=mode+1
            coretensor=coretensor._ttm_compute(np.transpose(factormatrice),mode, False)
        return coretensor,factor_matrices_list

def Initialization(X,Coretensorsize,Infotensor,Initname):
    [K,If,It]=np.array(np.shape(X),dtype=int)
    [Jf,Jt]=Coretensorsize[1:3]
    A2_old=np.zeros((If,Jf))
    A3_old=np.zeros((It,Jt))
    B2_old=np.zeros((Jf,Jf))
    B3_old=np.zeros((Jt,Jt))    
    if (Initname=="Multiplicative"):
       G_old,list_of_factors=Multiplicative_initialization_NTD2(X,Coretensorsize)
       A2_old=list_of_factors[0]
       A2_old=normalize_columns(A2_old)
       A3_old=list_of_factors[1]
       A3_old=normalize_columns(A3_old)
       list_of_factors=Multiplicative_initialization_NTD2(Infotensor,Coretensorsize)[1]
       B2_old=list_of_factors[0]
       B3_old=list_of_factors[1] 
       B2_old=normalize_columns(B2_old)
       B3_old=normalize_columns(B3_old)
    if(Initname=="Tucker2HOSVD"):
       if((Jf>If) or (Jt>It)):          
           raise AssertionError("The number of temporal or spectral dictionaries must not exceed the correspond dimension of the tensor to decompose for Tucker2 initialization")
       if((Jf<=If) and (Jt<=It)):           
           G_old,list_of_factors=Tucker2HOSVD(X,Coretensorsize,3,0,True)
           G_old=np.maximum(G_old,0)
           A2_old=np.maximum(list_of_factors[0],0)    
           A2_old=normalize_columns(A2_old) 
           A3_old=np.maximum(list_of_factors[1],0)   
           A3_old=normalize_columns(A3_old) 
           [B2_old,B3_old]=Tucker2HOSVD(Infotensor,Coretensorsize,3,0,False)
           B2_old=np.maximum(B2_old,0)
           B3_old=np.maximum(B3_old,0)
           B2_old=normalize_columns(B2_old)  
           B3_old=normalize_columns(B3_old)
    return G_old,A2_old,A3_old,B2_old,B3_old   
        
       
#This function is used to perform the supervised decomposition
def NTD_ALS_decoupledversion(X,Coretensorsize,max_iter,N,m,epsilon,hyperparam,Infotensor,InitStrat,parallel_decision,pool):    
    #G_old,list_of_factors=Tucker2HOSVD(X,Coretensorsize,N,m,True) 
    
#    G_old,list_of_factors=Multiplicative_initialization_NTD2(X,Coretensorsize)
#    
#    G_old=np.maximum(G_old,0)
#    A2_old=list_of_factors[0]
#    #A2_old=np.maximum(list_of_factors[0],0)    
#    A2_old=normalize_columns(A2_old)    
#    A3_old=list_of_factors[1]
#    #A3_old=np.maximum(list_of_factors[1],0)   
#    A3_old=normalize_columns(A3_old)    
#    A2_new=np.zeros(A2_old.shape)    
#    A3_new=np.zeros(A3_old.shape)
#    G_result=G_old    
#    I=np.identity(Coretensorsize[0])    
#    #[B2_old,B3_old]=Tucker2HOSVD(Infotensor,Coretensorsize,N,m,False) 
#    list_of_factors=Multiplicative_initialization_NTD2(Infotensor,Coretensorsize)[1]
#    B2_old=list_of_factors[0]
#    B3_old=list_of_factors[1]    
#    #B2_old=np.maximum(B2_old,0)    
#    B2_old=normalize_columns(B2_old)    
#    #B3_old=np.maximum(B3_old,0)    
#    B3_old=normalize_columns(B3_old)    
#    B2_new=np.zeros(B2_old.shape)
#    B3_new=np.zeros(B3_old.shape)
    I=np.identity(Coretensorsize[0])
    G_old,A2_old,A3_old,B2_old,B3_old=Initialization(X,Coretensorsize,Infotensor,InitStrat)   
    U=[I,A2_old,A3_old,I,B2_old,B3_old] 
    nb_iter=0   
    error_list=[]
    error=((X-Product_with_factors(G_old,U[0:3])).norm())**2+hyperparam*((Infotensor-Product_with_factors(G_old,U[3:6])).norm())**2
    previous_error=0
    error_list.append(error)
    vecInfotensor=np.resize(Infotensor,np.size(Infotensor))
    while(nb_iter<max_iter):        
       print('iter {:d}/{:d}'.format(nb_iter,max_iter))
       previous_error=error       
       nb_iter=nb_iter+1         
       #args=[A2_old,A3_old,B2_old,B3_old,vecInfotensor,hyperparam]       
       #G_new=decomposition_for_core_retrieving_parallelized(X,args,pool) 
       #G_new=Core_retrieving(X,U,pool,parallel_decision)
       G_new=Core_retrieving(X,U,pool,hyperparam,vecInfotensor,parallel_decision)
       error=((X-Product_with_factors(G_new,U[0:3])).norm())**2+hyperparam*((Infotensor-Product_with_factors(G_new,U[3:6])).norm())**2
       error_list.append(error)
       n=1
       temp=G_new.unfold(n)        
       B2=np.dot(np.kron(A3_old,I),np.transpose(temp))  
       #A2_new=np.transpose(NMF_decoupling(B2,np.transpose(X.unfold(n)),False)) #We update A1
       A2_new=np.transpose(NMF_decoupling_parallelized(B2,np.transpose(X.unfold(n)),pool)) #We update A2 
       A2_new=matrix_completion(A2_new)
       A2_new=normalize_columns(A2_new)       
       U[1]=A2_new
       error=((X-Product_with_factors(G_new,U[0:3])).norm())**2+hyperparam*((Infotensor-Product_with_factors(G_new,U[3:6])).norm())**2 
       error_list.append(error)
       n=2       
       temp=G_new.unfold(n)       
       B3=np.dot(np.kron(A2_new,I),np.transpose(temp))       
       A3_new=np.transpose(NMF_decoupling_parallelized(B3,np.transpose(X.unfold(n)),pool))#We update A2 
       A3_new=matrix_completion(A3_new)
       A3_new=normalize_columns(A3_new)       
       U[2]=A3_new       
       error=((X-Product_with_factors(G_new,U[0:3])).norm())**2+hyperparam*((Infotensor-Product_with_factors(G_new,U[3:6])).norm())**2 
       error_list.append(error)
       n=1       
       temp=G_new.unfold(n)       
       B2=np.dot(np.kron(B3_old,I),np.transpose(temp))       
       B2_new=np.transpose(NMF_decoupling_parallelized(B2,np.transpose(Infotensor.unfold(n)),pool))
       B2_new=matrix_completion(B2_new)
       B2_new=normalize_columns(B2_new)      
       U[4]=B2_new       
       error=((X-Product_with_factors(G_new,U[0:3])).norm())**2+hyperparam*((Infotensor-Product_with_factors(G_new,U[3:6])).norm())**2        
       error_list.append(error)       
       n=2       
       temp=G_new.unfold(n)       
       B3=np.dot(np.kron(B2_new,I),np.transpose(temp))       
       B3_new=np.transpose(NMF_decoupling_parallelized(B3,np.transpose(Infotensor.unfold(n)),pool))
       B3_new=matrix_completion(B3_new)       
       B3_new=normalize_columns(B3_new)      
       U[5]=B3_new       
       error=((X-Product_with_factors(G_new,U[0:3])).norm())**2+hyperparam*((Infotensor-Product_with_factors(G_new,U[3:6])).norm())**2       
       error_list.append(error)            
       U=[I,A2_new,A3_new,I,B2_new,B3_new]
       G_result=G_new       
       error=((X-Product_with_factors(G_new,U[0:3])).norm())**2+hyperparam*((Infotensor-Product_with_factors(G_new,U[3:6])).norm())**2
       error_list.append(error)
       if(previous_error-error<epsilon):
         print("The information about the error")
         print("The error")
         print(error)
         print("The previous error")
         print(previous_error)
         print("The difference between the error and the previous error")
         print(previous_error-error)
         print("The tolerance is")
         print(epsilon)
         U=[I,A2_old,A3_old,I,B2_old,B3_old]
         G_result=G_old       
         break
       A2_old=A2_new
       A3_old=A3_new
       B2_old=B2_new
       B3_old=B3_new
       G_old=G_new
    if((np.min(G_result)<0) or (np.min(U[1])<0) or (np.min(U[2])<0) or (np.min(U[4])<0) or (np.min(U[5])<0) ):
       raise AssertionError("There is a problem with the decomposition")
    if(nb_iter==1):
        raise AssertionError("The algorithm did not work because of insufficient number of iterations")
    return U,G_result,error_list,nb_iter
 






#This function is used to perform the supervised decomposition with penalty term on the spectral and temporal latent factors
def NTD_ALS_decoupledversionGD(X,Coretensorsize,max_iter,N,m,epsilon,lambdainfo,lambdaAf,lambdaAt,lambdaBf,lambdaBt,T,Infotensor,InitStrat):    
    #G_old,list_of_factors=Tucker2HOSVD(X,Coretensorsize,N,m,True) 
    
#    G_old,list_of_factors=Multiplicative_initialization_NTD2(X,Coretensorsize)
#    
#    G_old=np.maximum(G_old,0)
#    A2_old=list_of_factors[0]
#    #A2_old=np.maximum(list_of_factors[0],0)    
#    A2_old=normalize_columns(A2_old)    
#    A3_old=list_of_factors[1]
#    #A3_old=np.maximum(list_of_factors[1],0)   
#    A3_old=normalize_columns(A3_old)    
#    A2_new=np.zeros(A2_old.shape)    
#    A3_new=np.zeros(A3_old.shape)
#    G_result=G_old    
#    I=np.identity(Coretensorsize[0])    
#    #[B2_old,B3_old]=Tucker2HOSVD(Infotensor,Coretensorsize,N,m,False) 
#    list_of_factors=Multiplicative_initialization_NTD2(Infotensor,Coretensorsize)[1]
#    B2_old=list_of_factors[0]
#    B3_old=list_of_factors[1]    
#    #B2_old=np.maximum(B2_old,0)    
#    B2_old=normalize_columns(B2_old)    
#    #B3_old=np.maximum(B3_old,0)    
#    B3_old=normalize_columns(B3_old)    
#    B2_new=np.zeros(B2_old.shape)
#    B3_new=np.zeros(B3_old.shape)
    I=np.identity(Coretensorsize[0])
    G_old,A2_old,A3_old,B2_old,B3_old=Initialization(X,Coretensorsize,Infotensor,InitStrat)   
    U=[I,A2_old,A3_old,I,B2_old,B3_old]
    nb_iter=0   
    error_list=[]
    error=((X-Product_with_factors(G_old,U[0:3])).norm())**2+lambdainfo*((Infotensor-Product_with_factors(G_old,U[3:6])).norm())**2+(lambdaAf/2)*np.linalg.norm(A2_old)**2+(lambdaAt/2)*np.linalg.norm(A3_old)**2+(lambdaBf/2)*np.linalg.norm(B2_old)**2+(lambdaBt/2)*np.linalg.norm(B3_old)**2   
    previous_error=0
    error_list.append(error)
    vecInfotensor=np.resize(Infotensor,np.size(Infotensor))
    while(nb_iter<max_iter):        
       print('iter {:d}/{:d}'.format(nb_iter,max_iter))
       previous_error=error       
       nb_iter=nb_iter+1         
       args=[A2_old,A3_old,B2_old,B3_old,vecInfotensor,lambdainfo]       
       G_new=decomposition_for_core_retrieving_parallelized(X,args)       
       error=((X-Product_with_factors(G_new,U[0:3])).norm())**2+lambdainfo*((Infotensor-Product_with_factors(G_new,U[3:6])).norm())**2+(lambdaAf/2)*np.linalg.norm(A2_old)**2+(lambdaAt/2)*np.linalg.norm(A3_old)**2+(lambdaBf/2)*np.linalg.norm(B2_old)**2+(lambdaBt/2)*np.linalg.norm(B3_old)**2     
       error_list.append(error)
       n=1
       temp=G_new.unfold(n)        
       B=np.dot(temp,np.transpose(np.kron(A3_old,I)))
       A2_new=Factors_updating(X.unfold(n),B,epsilon,int(max_iter/2),lambdaAf,T)
       #A2_new=matrix_completion(A2_new)
       #A2_new=normalize_columns(A2_new)       
       U[1]=A2_new
       error=((X-Product_with_factors(G_new,U[0:3])).norm())**2+lambdainfo*((Infotensor-Product_with_factors(G_new,U[3:6])).norm())**2+(lambdaAf/2)*np.linalg.norm(A2_new)**2+(lambdaAt/2)*np.linalg.norm(A3_old)**2+(lambdaBf/2)*np.linalg.norm(B2_old)**2+(lambdaBt/2)*np.linalg.norm(B3_old)**2
       error_list.append(error)
       n=2       
       temp=G_new.unfold(n)       
       #B3=np.dot(np.kron(A2_new,I),np.transpose(temp))
       B=np.dot(temp,np.transpose(np.kron(A2_new,I)))       
       A3_new=Factors_updating(X.unfold(n),B,epsilon,int(max_iter/2),lambdaAt,T)       
       #A3_new=matrix_completion(A3_new)
       #A3_new=normalize_columns(A3_new)       
       U[2]=A3_new       
       error=((X-Product_with_factors(G_new,U[0:3])).norm())**2+lambdainfo*((Infotensor-Product_with_factors(G_new,U[3:6])).norm())**2+(lambdaAf/2)*np.linalg.norm(A2_new)**2+(lambdaAt/2)*np.linalg.norm(A3_new)**2+(lambdaBf/2)*np.linalg.norm(B2_old)**2+(lambdaBt/2)*np.linalg.norm(B3_old)**2
       error_list.append(error)
       n=1       
       temp=G_new.unfold(n)        
       B=np.dot(temp,np.transpose(np.kron(B3_old,I)))
       B2_new=Factors_updating(Infotensor.unfold(n),B,epsilon,int(max_iter/2),lambdaBf,T)
       #B2_new=matrix_completion(B2_new)
       #B2_new=normalize_columns(B2_new)      
       U[4]=B2_new       
       error=((X-Product_with_factors(G_new,U[0:3])).norm())**2+lambdainfo*((Infotensor-Product_with_factors(G_new,U[3:6])).norm())**2+(lambdaAf/2)*np.linalg.norm(A2_new)**2+(lambdaAt/2)*np.linalg.norm(A3_new)**2+(lambdaBf/2)*np.linalg.norm(B2_new)**2+(lambdaBt/2)*np.linalg.norm(B3_old)**2     
       error_list.append(error)       
       n=2       
       temp=G_new.unfold(n)       
       B=np.dot(temp,np.transpose(np.kron(B2_new,I)))      
       B3_new=Factors_updating(Infotensor.unfold(n),B,epsilon,int(max_iter/2),lambdaBt,T)
       
       #B3_new=matrix_completion(B3_new)       
       #B3_new=normalize_columns(B3_new)      
       U[5]=B3_new       
       error=((X-Product_with_factors(G_new,U[0:3])).norm())**2+lambdainfo*((Infotensor-Product_with_factors(G_new,U[3:6])).norm())**2+(lambdaAf/2)*np.linalg.norm(A2_new)**2+(lambdaAt/2)*np.linalg.norm(A3_new)**2+(lambdaBf/2)*np.linalg.norm(B2_new)**2+(lambdaBt/2)*np.linalg.norm(B3_new)**2      
       error_list.append(error)            
       U=[I,A2_new,A3_new,I,B2_new,B3_new]
       G_result=G_new       
       error=((X-Product_with_factors(G_new,U[0:3])).norm())**2+lambdainfo*((Infotensor-Product_with_factors(G_new,U[3:6])).norm())**2+(lambdaAf/2)*np.linalg.norm(A2_new)**2+(lambdaAt/2)*np.linalg.norm(A3_new)**2+(lambdaBf/2)*np.linalg.norm(B2_new)**2+(lambdaBt/2)*np.linalg.norm(B3_new)**2
       error_list.append(error)
       if(np.abs(previous_error-error)<epsilon):
         U=[I,A2_old,A3_old,I,B2_old,B3_old]
         G_result=G_old       
         break
       A2_old=A2_new
       A3_old=A3_new
       B2_old=B2_new
       B3_old=B3_new
       G_old=G_new
    if((np.min(G_result)<0) or (np.min(U[1])<0) or (np.min(U[2])<0) or (np.min(U[4])<0) or (np.min(U[5])<0) ):
       raise AssertionError("There is a problem with the decomposition")
    if(nb_iter==1):
        raise AssertionError("The algorithm did not work because of insufficient number of iterations. You should change the parameters")
    return U,G_result,error_list,nb_iter