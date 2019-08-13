from sktensor import dtensor

import numpy as np

import scipy

import copy

def vectorization(X):
    vecX=np.reshape(np.array(X),np.size(np.array(X)))
    [K,nrows,ncolumns]=np.array(X.shape,dtype=int)
    for i in range(K):
      matrix_of_interest=X[i,:,:]
      vecX[i*(nrows*ncolumns):(i+1)*(nrows*ncolumns)]=np.reshape(matrix_of_interest,np.size(matrix_of_interest))
    return vecX

def NMF_decoupling(A,B,decision_residual):
    size_B=np.array(B.shape,dtype=int)
    size_A=np.array(A.shape,dtype=int)
    nrows_sol=size_A[1]
    ncol_sol=size_B[1]
    result=np.zeros((nrows_sol,ncol_sol))
    for j in range(ncol_sol):
        result[:,j],residual=scipy.optimize.nnls(A,B[:,j])
    if (decision_residual==True):
      return result
    if (decision_residual==False):
      return result
def positive_least_squares_single(args) : 
    solution=scipy.optimize.nnls(args[0],args[1][:,args[2]])
    return solution
  
def NMF_decoupling_parallelized(A,B):
    size_A=np.array(A.shape,dtype=int)
    size_B=np.array(B.shape,dtype=int)
    nrows_sol=size_A[1]
    ncol_sol=size_B[1]
    solution=np.zeros((nrows_sol,ncol_sol)) 
    pool=Pool(2)
    sol=pool.map(positive_least_squares_single,[[A,B,n] for  n in range(ncol_sol)])
    for i in range(len(sol)):
        solution[:,i]=sol[i][0]
    return  solution  
   
def Error_estimation_factor(B,X,n,solution): 
   result=np.linalg.norm(np.dot(B,np.transpose(solution))-np.transpose(X.unfold(n)))
   return result
   
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
    beta=args[1][0]
    number_of_factor_matrices=len(args[1])
    for length in range(number_of_factor_matrices-1):
        beta=np.kron(beta,args[1][length+1])
    g_of_interest,residual=scipy.optimize.nnls(beta,vecX[args[2]*(I2*I3):(args[2]+1)*(I2*I3)])
    return g_of_interest
from multiprocessing import Pool  

def decomposition_for_core_retrieving_parallelized(X,U):
    [K,I2,I3]=np.array(X.shape,dtype=int)
    size_factor2=np.array(U[0].shape,dtype=int)
    J2=size_factor2[1]
    size_factor3=np.array(U[1].shape,dtype=int)
    J3=size_factor3[1]
    vecG=np.zeros(K*J2*J3)
    beta=U[0]
    number_of_factor_matrices=len(U)
    for length in range(number_of_factor_matrices-1):
        beta=np.kron(beta,U[length+1])
    pool=Pool(2)
    vecG=pool.map(retrieving_a_small_part_of_the_core,[ [X,U,k] for k in range(K)])
    G_result=np.reshape(vecG,(K,J2,J3))
    return G_result
    
def Product_with_factors(coretensor,factor_list):
    approxim=coretensor
    mode=-1
    for factor_matrix in factor_list:
        mode=mode+1
        approxim=approxim._ttm_compute(factor_matrix,mode,False)        
    return approxim

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



def NTD_ALS_decoupledversion(X,Coretensorsize,max_iter,N,m,epsilon):
    G_old,list_of_factors=Tucker2HOSVD(X,Coretensorsize,N,m,True)
    A2_old=np.maximum(list_of_factors[0],0)
    A3_old=np.maximum(list_of_factors[1],0)
    A2_new=np.zeros(A2_old.shape)
    A3_new=np.zeros(A3_old.shape)
    G_result=G_old
    I=np.identity(Coretensorsize[0]) 
    U=[I,A2_old,A3_old] 
    nb_iter=0   
    error_list=[]
    approxim=Product_with_factors(G_old,U)
    error=np.linalg.norm(X-approxim)
    previous_error=0
    error_list.append(error)
    while(nb_iter<max_iter):     
       previous_error=copy.copy(error)
       nb_iter=nb_iter+1       
       G_new=decomposition_for_core_retrieving_parallelized(X,U[1:3])
       approxim=Product_with_factors(dtensor(G_new),U)
       error=(X-dtensor(approxim)).norm() 
       error_list.append(error)
       for n in range(3):         
          if(n==1):
            temp=dtensor(G_new)
            temp=temp.unfold(n)      
            B2=np.dot(np.kron(A3_old,I),np.transpose(temp))    
            A2_new=np.transpose(NMF_decoupling(B2,np.transpose(X.unfold(n)),False)) #We update A1
            #A2_new=np.transpose(NMF_decoupling_parallelized(B2,np.transpose(X.unfold(n)))) #We update A2
            error=Error_estimation_factor(B2,X,n,A2_new)
            error_list.append(error) 
          if(n==2):
            temp=dtensor(G_new)
            temp=temp.unfold(n)
            B3=np.dot(np.kron(A2_new,I),np.transpose(temp)) 
            #A3_new=np.transpose(NMF_decoupling_parallelized(B3,np.transpose(X.unfold(n))))#We update A2
            A3_new=np.transpose(NMF_decoupling(B3,np.transpose(X.unfold(n)),False))#We update A2
            error=Error_estimation_factor(B3,X,n,A3_new)
            error_list.append(error)  
       U=[I,A2_new,A3_new]
       G_result=G_new
       if(previous_error-error<epsilon):        
         U=[I,A2_old,A3_old]
         G_result=G_old
       
         break
       A2_old=A2_new
       A3_old=A3_new
       G_old=G_new
      
    return U,G_result,error_list,nb_iter
 


