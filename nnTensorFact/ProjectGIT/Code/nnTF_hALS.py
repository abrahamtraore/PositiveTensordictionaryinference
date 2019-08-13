#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 11:56:16 2017

@author: maxime
"""
import numpy as np
import tempEA
import pdb

from sktensor import dtensor


def nnTF2_hALS(Y,Coretensorsize,epsilon,max_iter):
    # Y tensor of size I1,...,In
    # gshape : shape of the core-tensor 
    # epsilon : stop stepsize between successive reconstruction error
    #----------------------------------
    # ouputs : G core tensor of gshape
    # A a collection of factors of size Ii times Ji
    # error:list of reconstruction error
    
    # Example use :
    #Y = np.random.rand(60,80,50)
    #G,A,error = nnTF_hALS.nnTF2_hALS(Y,(60,20,20),1e-6)
    
    # Rmk : there is a possible bug when w=0 
    
    
    error= []
    N = Y.ndim
    # initialisation method is the same
    Y = dtensor(Y)
    G,Atemp = tempEA.Tucker2HOSVD(Y,Coretensorsize,N,0,True)
    
    # this is the core Tensor
    G = np.maximum(G,0)
    G = dtensor(G)
    
    A = []
    for n in np.arange(0,N):
        if n == 0 :
            # first factor mode is identity and is insered
             A.append(np.eye(Y.shape[0]))
        else : 
            # other factors
            A.append(np.copy(Atemp[n-1]))
            A[n] = np.maximum(A[n],0)
            #check vacant factors
            check = np.argwhere(A[n].sum(axis=0)==0).flatten()
            A[n][:,check] = np.random.rand(Y.shape[n],len(check))
            # normalize
            fact = np.dot(np.ones((Y.shape[n],1)),A[n].sum(axis=0)[np.newaxis])
            fact = np.power(fact,-1)
            A[n] = np.multiply(fact,A[n])
   
    
    
    # needs initial computation of residuals to work
    Yhat = dtensor(np.copy(G))
    Yhat = Yhat.ttm(A,np.arange(0,N))
    E = dtensor(Y - Yhat)
    Eold = np.linalg.norm(E)**2+2*epsilon
    #
    nbiter=0
    while ( (abs(Eold -np.linalg.norm(E)**2) > epsilon) and (nbiter<max_iter) ):
        nbiter=nbiter+1
        Eold = np.linalg.norm(E)**2
        print(Eold)
        error.append(Eold)
        # first update A each mode factor matrix, except the first
        for n in np.arange(1,N):
            Gi = np.copy(G)
            Gi = dtensor(Gi)
            for m in np.setdiff1d(np.arange(0,N),n):
                Gi = Gi.ttm(A[m],m)
            #each columns of factor matrix 
            for i in np.arange(0,A[n].shape[1]):
                cond = np.zeros(A[n].shape[1]) 
                cond[i] = 1
                Xi = np.copy(Gi.compress(cond,axis=n))
                # select the sub-part of the tensor where the A[n,:,i] is 
                # the only non-null component on mode n
                Xi = dtensor(Xi)
                Xi = Xi.unfold(n)
                w = np.linalg.norm(Xi)**2
                w=np.maximum(w,epsilon)
                if w > 0:
                    # Maj a
                    ai = np.copy(A[n][:,i][:,np.newaxis])
                    ai = ai  + 1/w*np.dot(E.unfold(n),Xi.T)
                    ai = np.maximum(ai,0)
                    if ai.sum()== 0:
                        ai = np.random.rand(ai.shape[0],1)
                else:
                    pdb.set_trace()
                #Maj E
                Xi = np.copy(Gi.compress(cond,axis=n))
                Xi = dtensor(Xi)
                E = E + Xi.ttm(A[n][:,i][:,np.newaxis]-ai,n) 
                # normalisation de a
                A[n][:,i] = ai.flatten()/np.linalg.norm(ai) 
         
        
        #each elements of G
        for index in np.ndindex(G.shape):
            #build 
            Eter = dtensor(np.copy(E))
            Ebis = np.array([1])
            for n in np.arange(0,N-1):
                Ebis=Ebis[:,np.newaxis]
            Ebis = dtensor(Ebis)   

            for n in np.arange(0,N):
                Al = A[n][:,index[n]][:,np.newaxis]
                Ebis = Ebis.ttm(Al,n)
                Eter = Eter.ttm(Al.T,n)                                 
                              
            g = np.copy(G[index]) + Eter.item()
            g =  np.maximum(g,0)
            E = E + (G[index]-g)*Ebis
            G[index] = g
            # delete elements
            del Ebis
            del Eter
            
         
                     
    return G,A,error