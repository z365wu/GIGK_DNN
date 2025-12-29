#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 20:38:32 2024
@author: z365wu, Haokun, and q7he
    collaborate work: Zhenggao Wu and Haokun Zhao: Discrete PH/PH/K queue
    --- Compute the stationary distribution of the queue length
"""
import numpy as np
from math import factorial
from numpy.linalg import inv
from numpy import random
# import sympy as sp
import copy
import math
np.seterr(all='ignore')

# Generate discrete time PH-representation (alpha, T)
# m: state number
def  DTPH_Rep_generator(m):        
    v_alpha = np.random.rand(m) 
    v_alpha = v_alpha/v_alpha.sum()   # normalize alpha so that the sum of each element equals 1
    # print(v_alpha.shape)
    
    m_T = np.random.rand(m, m + 1)    # The (m+1)-st column is for absorbtion probability
    # Normalize each row so that the sum of each row equals 1
    row_sums = m_T.sum(axis=1)
    m_T_normalized = m_T / row_sums[:, np.newaxis]
    # only keep the m * m generative matrix
    m_T = m_T_normalized[:,:-1]

    return  v_alpha, m_T     # Return (alpha, T)

#####  discrete time PHD moments generator   ##### 
#    PH-representation (alpha, T)
#    n_max: The highest order of moments
def  DTPHD_Moments(v_alpha, m_T, n_max):
    # factorial moments for a discrete random variable 
    # reference: http://www2.imm.dtu.dk/courses/02407/lectnotes/ftf.pdf
    # reference 2 book: Alfa_Applied_Discrete_Time_Queues.pdf, page 39
    
    moment_f_list = [] # a list of factorial moments
    for n in range(1,n_max+1):
        momnet_part = np.linalg.matrix_power(np.linalg.inv(np.identity(m_T.shape[0])-m_T), n)
        # moment_factorial = np.math.factorial(n) * np.sum(np.matmul(v_alpha,np.matmul(np.linalg.matrix_power(m_T, n-1), momnet_part)))
        moment_factorial = math.factorial(n) * np.sum(np.matmul(v_alpha,np.matmul(np.linalg.matrix_power(m_T, n-1), momnet_part)))
        moment_f_list.append(moment_factorial)
    
    # Calculate the moments using the factorial moments and the polynomial coefficients
    v_moments = np.zeros(n_max)  # an array to store the calculated moments
    coffe_listNEW = [1]
    for n in range(0, n_max):
        if n == 0:
            v_moments[0] = moment_f_list[0]    # moment 1
        else:
            if n == 1:
                coffe_listNEW.append(-1)
            else:
                tempv = 1
                for i in range(1, n):
                    tempv2 = coffe_listNEW[i]
                    coffe_listNEW[i] = coffe_listNEW[i] - n*tempv
                    tempv = tempv2
                coffe_listNEW.append(-n*tempv)              
            # print(n, 'coeff_new = ', coffe_listNEW)
            tempvector = copy.copy(coffe_listNEW)
            tempvector.pop(0)
            tempvector.reverse()
            # print('tempvector', tempvector)
            # Get the coefficients for the current polynomial
            coffe_list = tempvector
            # Calculate the current moment by subtracting the sum of products of previous moments and their coefficients
            moment_n = moment_f_list[n] - sum([a * b for a, b in zip(v_moments.tolist(), coffe_list)])
            v_moments[n] = moment_n # Store the calculated moment
    
    return v_moments # Return the first n_max momen


########## This function is for the construction of L+(k,m) ##############
def Discrete_CSFP_Matrices_PPlusu(k, m, beta, PPluskm):    
    if k==1:                     # PPlusbetakm[1][k]
        tempM = beta[0:m].reshape(1,-1) # change row vector into a matrix
        # Note: change '[beta[0:m]]' into 'beta[0:m].reshape(1,-1)'
    elif m==1:                   # PPlusbetakm[k][1]
        tempM = beta[0:1].reshape(1,-1) # change row vector into a matrix
        # Note: change '[beta[0:m]]' into 'beta[0:m].reshape(1,-1)'
    else:
        Mij = np.zeros((k, k+1, 1, 2))     # To find the sizes of blocks (note the k equals k_input + 1, see main_QBD_PHPHK_CSFP() )
        for j in range(0, k):
            m1 = np.size(PPluskm[k-1-j][m-2], 0)   
            for i in range(0, k):
                # print(i, j, k, m) # for testing
                m2 = np.size(PPluskm[k-1-i][m-2], 1)           
                Mij[j][i] = [m1, m2]
            Mij[j][k] = [m1, 1]        
        tempM = 0     # To find the matrix with k by k+1 blocks: Only diaonal blocks and upper off-diagonal (k,k+1) are non zero
        for j in range(0, k):   # For each row, since all blocks in each row have the same number of rows, we use hstack() to put them together
            if j==0:
                tempM_RowJ = PPluskm[k-1][m-2]    # The first block in the each row of blocks: block [1,1]
            else: 
                tempM_RowJ = np.zeros((int(Mij[j][0][0,0]), int(Mij[j][0][0,1]))) # Block[j,1] = 0 if j=>1; np.zeros(row size, col size)
            for i in range(1, k+1):  # The rest of the blocks in the j-th row of blocks
                if j == i:
                    tempM_RowJ = np.hstack((tempM_RowJ, PPluskm[k-1-j][m-2]))      # Block[j,j]: stack it to the right       
                elif j+1 == i:                                                     # Block[j,j+1]: stack it to the right
                    tempM_RowJ = np.hstack((tempM_RowJ, beta[m-1]*np.identity(np.size(PPluskm[k-1-j][m-2], 0))))  
                else:  # All other blocks are zero matrices
                    tempM_RowJ = np.hstack((tempM_RowJ, np.zeros((int(Mij[j][i][0,0]), int(Mij[j][i][0,1])))))
            if j==0:  # The first row of blocks
               tempM = tempM_RowJ
            else:     # The 2, 3, ..., k rows: Stack them below.
               tempM = np.vstack((tempM, tempM_RowJ))
    # print('New', tempM)        
    return tempM

####This function is for the construction of L-(k,m) ##########################
def Discrete_CSFP_Matrices_Pminus(k, m, s0, PMinuskm):
    if k==1:               # PMinuss0km[1][k]
        tempM = s0[0:m]
    elif m==1:             # PMinuss0km[k][1]
        tempM = k*s0[0:1]
    else:
        Mij = np.zeros((k+1, k, 1, 2))     # To find the size of blocks: (k+1) by k blocks 
        for j in range(0, k):
            m1 = np.size(PMinuskm[k-1-j][m-2], 0)   
            for i in range(0, k):
                m2 = np.size(PMinuskm[k-1-i][m-2], 1)           
                Mij[j][i] = [m1, m2]    
            m2_j = np.size(PMinuskm[k-1-j][m-2], 1)  #  The last row   
            Mij[k][j] = [1, m2_j]    #  The last row
        # print(Mij)
        tempM = 0   # The matrix is a k+1 by k blocks of matrix. Only blocks[k][k] and blocks[k+1][k] are nonzero.
        for j in range(0, k):  # Rows 1 to k 
            if j==0:
                tempM_RowJ = PMinuskm[k-1][m-2]   # Block[1][1] 
            elif j==1:
                tempM_RowJ = j*s0[m-1]*np.identity(np.size(PMinuskm[k-1-j][m-2],0))  # Block[2][1]
            else:
                tempM_RowJ = np.zeros((int(Mij[j][0][0,0]), int(Mij[j][0][0,1]))) # Blocks[k][1] = 0 for k>2
            for i in range(1, k):  # Blocks 2, 3, ..., k (i.e., [j][2], [j][3], ..., [j][k])
                if j == i:
                    tempM_RowJ = np.hstack((tempM_RowJ, PMinuskm[k-1-j][m-2]))   # Block[j][j]          
                elif j == i+1:      # Note: deleted and k-1-j > -1  # Block[j+1][j]  
                    tempM_RowJ = np.hstack((tempM_RowJ, j*s0[m-1]*np.identity(np.size(PMinuskm[k-1-j][m-2],0))))
                else:   # The rest of blocks: zeros
                    tempM_RowJ = np.hstack((tempM_RowJ, np.zeros((int(Mij[j][i][0,0]), int(Mij[j][i][0,1])))))
            if j==0:
                tempM = tempM_RowJ   # The first row
            else:
                tempM = np.vstack((tempM, tempM_RowJ))  # The 2, 3, ..., k rows: Stock it below.
        tempM_RowJ = np.zeros((int(Mij[k][0][0,0]), int(Mij[k][0][0,1])))  # The (k+1)-st row
        for i in range(1, k):            
            if k == i+1:
                tempM_RowJ = np.hstack((tempM_RowJ, k*s0[m-1]*np.identity(1))) # why need to multiple identity matrix
                # note: changed 'np.identity(np.size(PMinuskm[0][m-2],1))' into np.identity(1)
            else:
                tempM_RowJ = np.hstack((tempM_RowJ, np.zeros((int(Mij[k][i][0,0]), int(Mij[k][i][0,1])))))
        if j==0:
            tempM = tempM_RowJ
        else:
            tempM = np.vstack((tempM, tempM_RowJ))     
    #print('New PMinuskm', tempM)    
    return tempM       

##### Descripton: This function is for the construction of S(k,m).
def Discrete_Matrices_Pkm(k, m, S, PtPluskm, PtMinuskm, Pkm):
    if k==1:
        tempM = np.array([[1.0]])                # Pkm[1][k]
    elif m==1:
        tempM = np.array([[S[0][0] ** (k-1)]])       # Pkm[k][1]
    else:
        for j in range(1, k+1): # row
            for i in range(1, k+1): # col
                temp_sum = 0 # store the summation over P_{S[m,1:m-1],S[1:m-1,m]}
                for t in range(max(0, (k-j)-(k-i)), min(k-(k-i)-1,k-j)+1):
                    tempP = 1                    
                    # calculate P_{u, v}{q,j,m|k} as described in Proposition 4.2 (He and Alfa, 2015)
                    for u in range(k-j, max(1, k-j-t+1)-1, -1):
                        # Perform multiplication over \( L_{v}^{-} \)
                        # Utilize np.dot for matrix operations as 'tempP' is a scalar in the initial iteration
                        # np.dot(a, b): For 2-D arrays, it performs matrix multiplication
                        tempP = np.dot(tempP, PtMinuskm[u-1][m-2]) 
                    tempP = np.dot(tempP, Pkm[k-j-t][m-2])
                    for u in range(max(0, k-j-t)+1, k-i+1):
                        # Use np.matmul as tempP is a matrix in these iterations
                        tempP = np.matmul(tempP, PtPluskm[u-1][m-2]) 
                    consta = (1/factorial(t)) * (factorial(j-1) / (factorial(i-1-t) * factorial(j+t-i))) # coefficients: 1/k! * (k-q-l out of k-j)  (equation 19 in He and Alfa, 2015) 
                    temp_sum += consta * tempP * (S[m-1, m-1] ** (max(0, i-1-t))) # times (s_{m,m}^{k-q-l}) (equation 19 in He and Alfa, 201)
                if i == 1:
                    tempM_RowJ = temp_sum
                else:
                    tempM_RowJ = np.hstack((tempM_RowJ, temp_sum)) # combine matrix over column
            if j==1:
                tempM = tempM_RowJ
            else:
                tempM = np.vstack((tempM, tempM_RowJ)) # combine matrix over row 
    #print('New', tempM)    
    return tempM 

################# Construct QBD for MAP/PH/K (including PH/PH/K) by Count server for phase #######################
def  main_QBD_DTPHPHK_CSFP(ma, ms, D0, D1, alpha_s, T_s, K, Lmax):
    # Construction of transition blocks: PPlus(k, m), Qminus(k, m), Q(k,m), PPlus(k,m)
    # PPlus(k,m); K=2
    # print('start to build QBD')
    PPlusbetakm = [[[] for m in range(0, ms)] for k in range(0, K)]
    for k in range(0, K):
        for m in range(0, ms):
            PPlusbetakm[k][m] = Discrete_CSFP_Matrices_PPlusu(k+1, m+1, alpha_s, PPlusbetakm)
            # print('PPluskm = ', k, m, PPluskm[k][m])            
    # QMinus(k
    PMinuss0km = [[[] for m in range(ms)] for k in range(K)]   # cell(K,ms);       # S-(k,m)
    T_s0 = np.ones([ms,1]) - np.matmul(T_s, np.ones([ms,1])) 
    for k in range(0, K):         #k=1:K
        for m in range (0, ms):   # m=1:ms
            PMinuss0km[k][m] = Discrete_CSFP_Matrices_Pminus(k+1, m+1, T_s0, PMinuss0km)
            # print('QMinuskm = ', k, m, QMinuskm[k][m])       
    # For Q(k,m), we need PtPluskm and PtMinuskm
    Pkm = [[[] for k in range(0, ms)] for m in range(0, K+1)]  # cell(K+1,ms);          % Q(k,m)
    for m in range(0, ms):  # m=1:ms
        PtPluskm = [[[] for j in range(ms-1)] for k in range(K)]   #cell(K,ms-1);
        for k in range(0, K):   #k=1:K
            for j in range(0, m):   # j=1:m-1
                PtPluskm[k][j] = Discrete_CSFP_Matrices_PPlusu(k+1, j+1, T_s[m][:m+1], PtPluskm)
                
        #print('PtPluskm', PtPluskm)
        PtMinuskm = [[[] for j in range(ms-1)] for k in range(K)]   #(K,ms-1); ?
        for k in range(0, K):   #k=0:K
            for j in range(0, m):  #j=1:m-1
                PtMinuskm[k][j] =Discrete_CSFP_Matrices_Pminus(k+1, j+1, T_s[:m+1,m].reshape(-1,1), PtMinuskm)
        #print('PtMinuskm', PtMinuskm)
        for k in range (0, K+1):   #k=1:K+1
            Pkm[k][m] = Discrete_Matrices_Pkm(k+1, m+1, T_s, PtPluskm, PtMinuskm, Pkm)
        #print(m, Pkm)
    #print(Pkm)
                
    ### Calculate transition probability blocks based on Proposition 4.1 (He and Alfa, 2015)
    ### Construction of Ak,j and {A0, A1, ..., AK}%%%%%
    # {A0, A1, ..., AK}
    # print('start to construct Ak')
    Ak = [[] for k in range(K+2)]  # A0, A1, ..., AK
    Ak[0] = np.kron(D1, Pkm[K][ms-1])
    for k in range(1,K+1):
        tempM1 = 1
        for j in range(K, K+3-k-1-1,-1):
            tempM1 = np.dot(tempM1, PMinuss0km[j-1][ms-1]) 
        tempM2 = 1
        for j in range(K+3-k-1, K+1):
            tempM2 = np.dot(tempM2, PPlusbetakm[j-1][ms-1])
        consta = 1 / factorial(k-1)
        Ak[k] = np.kron(D0, consta * np.dot(np.dot(tempM1, Pkm[K+3-k-1-1][ms-1]), tempM2))
        tempM1 = 1
        for j in range(K, K+2-k-1-1,-1):
            tempM1 = np.dot(tempM1, PMinuss0km[j-1][ms-1])
        tempM2 = 1
        for j in range(K+1-k, K+1):
            tempM2 = np.dot(tempM2, PPlusbetakm[j-1][ms-1])
        consta = 1/factorial(k)
        Ak[k] = Ak[k] + np.kron(D1, consta*np.dot(np.dot(tempM1, Pkm[K+2-k-1-1][ms-1]), tempM2))        
    tempM1 = 1
    for k in range(K,0,-1):
        tempM1 = np.dot(tempM1, PMinuss0km[k-1][ms-1]) 
    tempM2 = 1
    for k in range(1, K+1):
        tempM2 = np.dot(tempM2, PPlusbetakm[k-1][ms-1])
    consta = 1/factorial(K)
    Ak[K+1] = np.kron(D0, consta * np.dot(np.dot(tempM1, Pkm[0][ms-1]), tempM2))  
    # Construction of all the blocks:Ak,j
    Akk = [[[] for m in range(0, 2*K+2)] for k in range(0, 2*K+2)]
    # Ak, k+1: k = 0, 1, ..., K-1
    for k in range(0, K):
        Akk[k][k+1] = np.kron(D1, np.matmul(Pkm[k][ms-1], PPlusbetakm[k][ms-1]))        
    # Ak,k+1: k = K, K+1, ..., 2K+1
    for k in range(K, 2*K+1):
        Akk[k][k+1] = np.kron(D1, Pkm[K][ms-1])
    # Ak,0: k = 1, 2, ..., K
    for k in range(0, K+1):
        tempM = 1
        for j in range(k, 2-2,-1):
            tempM = np.dot(tempM, PMinuss0km[j-1][ms-1])
        Akk[k][0] = np.kron(D0, tempM/factorial(k))
    # Ak,k-K: k = K+1:2K; serving K customers
    tempM = 1
    for j in range(K, 0, -1):
        tempM = np.dot(tempM, PMinuss0km[j-1][ms-1])
    tempM = tempM / factorial(K)
    for k in range(K+2-1, 2*K+1):
        tempM2 = 1
        for j in range(0, k-K):
            tempM2 = np.dot(tempM2, PPlusbetakm[j][ms-1])    
        Akk[k][k-K] = np.kron(D0, np.dot(tempM, tempM2))
    # Ak,j: k<= K and j<=k
    for k in range(1,K+1):
        for j in range(1, k+1):
            tempM1 = 1
            for t in range(k-1,j-1,-1):
                tempM1 = np.dot(tempM1, PMinuss0km[t][ms-1])
            Akk[k][j] = np.kron(D0, np.dot(tempM1, Pkm[j][ms-1])/factorial(max(0,k-1-j+1)))
            tempM1 = np.dot(tempM1, PMinuss0km[j-1][ms-1])/factorial(max(0,k-1-j+2))
            Akk[k][j] = Akk[k][j] + np.kron(D1, np.matmul(np.dot(tempM1, Pkm[j-1][ms-1]), PPlusbetakm[j-1][ms-1]))    
    # Ak,j: k = K+1, ..., 2*K-1; j=k-K+1, ..., k
    for k in range(K+1, 2 * K):
        for j in range(k-K+1, k+1):
            tempM1 = 1
            for t in range(K-1, K-(k-j)+1-2, -1):
                tempM1 = np.dot(tempM1, PMinuss0km[t][ms-1])
            tempM2 = 1
            for t in range(K-(k-j)+1-1, min(j-1, K-1)+1):
                tempM2 = np.dot(tempM2, PPlusbetakm[t][ms-1])                
            Akk[k][j] = np.kron(D0, np.dot(np.dot(tempM1, Pkm[K-(k-j)+1-1][ms-1]), tempM2)/factorial(k-j))
            tempM1 = np.dot(tempM1, PMinuss0km[K-(k-j)-1][ms-1])
            tempM2 = np.dot(PPlusbetakm[K-(k-j)-1][ms-1], tempM2)
            Akk[k][j] = Akk[k][j] + np.kron(D1, np.dot(np.dot(tempM1, Pkm[K-(k-j)-1][ms-1]), tempM2)/factorial(k-j+1))    
    # Ak,j: k =2K+1, 2K+2
    for k in range(2*K, 2*K+2):
        for j in range(k-K+1, k+1):
            tempM1 = 1
            for t in range(K-1, K-(k-j)+1-2, -1):
                tempM1 = np.dot(tempM1, PMinuss0km[t][ms-1])
            tempM2 = 1
            for t in range(K-(k-j)+1-1, K):
                tempM2 = np.dot(tempM2, PPlusbetakm[t][ms-1])                
            Akk[k][j] = np.kron(D0, np.dot(np.dot(tempM1, Pkm[K-(k-j)+1-1][ms-1]), tempM2)/factorial(k-j))
            tempM1 = np.dot(tempM1, PMinuss0km[K-(k-j)-1][ms-1])
            tempM2 = np.dot(PPlusbetakm[K-(k-j)-1][ms-1], tempM2)
            Akk[k][j] = Akk[k][j] + np.kron(D1, np.dot(np.dot(tempM1, Pkm[K-(k-j)-1][ms-1]), tempM2)/factorial(k-j+1))           
    # Check correctioness
    # Row sums to be one
    for k in range(0, K+1):
        tempv = np.sum(Akk[k][0], axis=1) # Sum across each row
        for j in range(1, k+1+1):
            tempv = tempv + np.sum(Akk[k][j], axis =1) # Sum across each row    
    for k in range(K+1, 2*K+1):
        if not np.any(Akk[k][k-K]):
            tempv = 0
        else:
            tempv = np.sum(Akk[k][k-K], axis = 1)
        for j in range(k-K+1, k+1+1):
            tempv = tempv + np.sum(Akk[k][j], axis = 1)
    # Row sums to be one
    tempM = Ak[0]
    for k in range(1,K+2):
        tempM = tempM + Ak[k]
    np.sum(tempM, axis=1)
    # Compute the mean queue length:
    # Compute matrix R: before blocking
    # Matrix R: R = A0 + R*A1 + R^2*A2 + ... + R^{K+1}A_{K+1}
    ### Iteration for matrix R
    # print('start to compute R')
    m_R = np.zeros([np.size(Ak[0],0), np.size(Ak[0],0)])   
    Iter_max = 5000
    epslon_err = 1.0e-15
    Iter_num = 0
    error_sum = 1.0e10
    while Iter_num < Iter_max and error_sum > epslon_err:
        R_new = Ak[0]
        Rk = m_R
        for j in range(1, K+2):
            R_new = R_new + np.matmul(Rk, Ak[j])
            Rk = np.matmul(Rk, m_R)
        error_sum = sum(sum(abs(m_R-R_new)))
        Iter_num = Iter_num + 1 
        m_R = R_new        
    print('Iteration number for R (with R size = ', np.size(Ak[0], 0), ') = ', Iter_num)
    # print(max(abs(np.linalg.eig(m_R)[0]))) # the maximal eigenvalue of R

    # Compute R_{K}, R_{K-1}, ..., R_1
    Rtemp = [[] for k in range(0, 2*K+1)]
    for k in range(K+1, 2 * K + 1):
        Rtemp[k] = m_R        
    for k in range(K, -1, -1):
        tempM = np.matmul(Rtemp[k+K], Akk[k+K][k])
        for j in range(K-1, 0, -1):
            tempM = np.matmul(Rtemp[k+j], Akk[k+j][k] + tempM)
        tempM = tempM + Akk[k][k]
        tempM = np.eye(Akk[k][k].shape[0], Akk[k][k].shape[0]) - tempM
        if k > 0:
            Rtemp[k] = np.matmul(Akk[k-1][k], inv(tempM))    
    tempM[:,0] = np.ones(Akk[0][0].shape[0])
    tempM = inv(tempM)
    pi0 = tempM[0,:]
    tempM = np.eye(Akk[0][0].shape[0])
    sumv = np.ones(Akk[0][0].shape[0])
    for k in range(0, K-1):
        tempM = np.matmul(tempM, Rtemp[k+1]) 
        sumv = sumv + np.sum(tempM, axis = 1)    
    tempM = np.matmul(np.matmul(tempM, Rtemp[K]), inv(np.eye(m_R.shape[0]) - m_R))
    sumv = sumv + np.sum(tempM, axis = 1)
    pi0 = pi0 / np.dot(pi0, sumv)    
    pi = []
    pi.append(pi0.sum())
    pi_temp = pi0.reshape(1,-1)
    for k in range(1, Lmax):
        if k < K+1:
            pi_k = np.matmul(pi_temp, Rtemp[k]) # pi(k) = pi(k-1)*R_{k}; note that R_k is Rtemp[k] (R_1 = Rtemp[1]; and Rtemp[0] is useless)
        else: # using R if k > K
            pi_k = np.matmul(pi_temp, Rtemp[K+1])
        pi.append(pi_k.sum())
        pi_temp = pi_k    
    return pi, Iter_num


def Discrete_PH_PH_K(j, Sample_size, K, m_max, n_max, Lmax, rho_given=False):    
    indicator = 0
    while indicator == 0:
        ma = random.randint(1, m_max) # Phase numbers of arrival process
        ms = random.randint(1, m_max) # Phase number of service process
        if rho_given == False:
            # Randomly generate traffic intensity \in (0, 1)
            rho = random.uniform(1e-10, 1) # use 1e-10 to exclude 0    
        else:
            rho = rho_given
        # rRamdonly generate PH representaion for arrival and serving processes
        alpha_a, T_0 = DTPH_Rep_generator(ma)  # PH-representation for the interarrival time; alpha_a.sum() = 1
        alpha_s, S_0 = DTPH_Rep_generator(ms)  # PH-representation for the service time; alpha_s.sum() = 1    
        
        # Interarrival/service rate of the ramdonly generated PH/PH/K
        lbd_0 = 1 / np.sum(np.matmul(alpha_a, inv(np.identity(T_0.shape[0])-T_0)))
        mu_0 = 1 / np.sum(np.matmul(alpha_s, inv(np.identity(S_0.shape[0])-S_0)))
        # Adjust ramdonly generated T_0 and T_0 to make the PH/PH/1 have Traffic intensity rho
        delta = lbd_0 / (rho * K* mu_0)
        T = (1 - min(1, 1/delta)) * np.identity(T_0.shape[0]) + min(1, 1/delta) * T_0
        S = (1 - min(1, delta)) * np.identity(S_0.shape[0]) + min(1, delta) * S_0
        mu = 1 / np.sum(np.matmul(alpha_s, inv(np.identity(S.shape[0])-S))) # service rate
        lbd = 1 / np.sum(np.matmul(alpha_a, inv(np.identity(T.shape[0])-T))) # interarrival rate    
        # moment generation function
        moments_Arrival = DTPHD_Moments(alpha_a, T, n_max)
        moments_Service = DTPHD_Moments(alpha_s, S, n_max)
        # Normalize the moments so that E[X_s]=1 by letting X_s =: X_s/E[X_s] and X_a =: X_a/E[X_a] 
        mean_s = moments_Service[0]
        for i in range(0, n_max):
            moments_Arrival[i] = moments_Arrival[i]/np.power(mean_s, i+1)
            moments_Service[i] = moments_Service[i]/np.power(mean_s, i+1)
        
        max_moments = max(max(moments_Arrival), max(moments_Service)) 
        D0 = T
        tempM = np.ones([ma, 1]) - np.matmul(D0, np.ones([ma, 1]))
        D1 = tempM*alpha_a
        T_s = S
        print('D: Sample j =', j+1, 'with (K =', K, 'ma =', ma, 'and ms =', ms, ') of a total of', Sample_size, 'samples.')        
        # print(max_moments)
        if max_moments < 1.0e30:      
            indicator = 1
            # Log transform of the moments to make those numbers smaller for DNN training
            SCV_A = moments_Arrival[1]/(moments_Arrival[0]** 2) - 1  # SCV for the arrival time            
            SCV_S = moments_Service[1]/(moments_Service[0]** 2) - 1  # SCV for the service time            
            moments_Arrival_log = np.log(1 + moments_Arrival)
            moments_Service_log = np.log(1 + moments_Service)    
            v_stationary, Iter_num = main_QBD_DTPHPHK_CSFP(ma, ms, D0, D1, alpha_s, T_s, K, Lmax)           # print(DTPHPH1StatDist)
            if rho_given == False:
                return moments_Arrival_log, moments_Service_log, v_stationary, SCV_A, SCV_S, rho, Iter_num
            else:
                return ma, alpha_a, T, ms, alpha_s, S, moments_Arrival, moments_Service, v_stationary
    
#    m_max: The maximum order of Discrete PH-representation (alpha, T)
#    n_max: The highest order of moments
#    K:     The number of servers
def  Input_Output_Moments_Generator_Discrete(Sample_size, K, m_max, n_max, Lmax, max_moment_bound, rho_lower, rho_upper):
     Moments = np.zeros((Sample_size, 2*n_max))               # Moments of interarrival and service times (return)
     Stationary_distribution = np.zeros((Sample_size, Lmax))  # Statioanry distributions of queue length (return)
     SCVs = np.zeros((Sample_size, 2))  # SCVs of our distributions (return): To demonstrate the versatility of samples
     Rhos = np.zeros((Sample_size, 1))  # Traffice intensity of queues (return): To demonstrate the versatility of samples
     R_Iter_num = np.zeros((Sample_size, 1))  #  R iteration number for stationary distribution of queue length
     queue_time_type = ['discrete'] * Sample_size # To save samples into a csv file (Excel file)
     #M_T_type_a = [] # type of m_T (e.g., Coxian, PH, Erlang) for arrival processes
     #M_T_type_s = [] # type of m_T (e.g., Coxian, PH, Erlang) for service processes
     
     for j in range(0, Sample_size):
        print('- A discrete time PH/PH/K sample')
        Moments[j,0:n_max], Moments[j,n_max:2*n_max], Stationary_distribution[j,:], SCVs[j,0], SCVs[j,1], Rhos[j], R_Iter_num[j] = Discrete_PH_PH_K(j, Sample_size, K, m_max, n_max, Lmax)  # Call for the discrete module

     return Moments, Stationary_distribution, SCVs, Rhos, queue_time_type, R_Iter_num
    

######### Test ###############################
# K = 3
# m_max = 10
# n_max = 10
#Lmax = 100
#j = 0
#Sample_size = 1
#moments_Arrival_log, moments_Service_log, v_stationary, SCV_A, SCV_S, rho = Discrete_PH_PH_K(j, Sample_size, K, m_max, n_max, Lmax)
#print(f'stationary queue length distribution: {v_stationary}')
