# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:53:10 2024: Construction of QBD for GI/G/K using CSFP

@author: z365wu and q7he
"""
import numpy as np
#import random
#from numpy.linalg import inv

########## This function is for the construction of S+(k,m) ##############
def  Matrices_QPlus(k, m, beta, SPluskm):
    if k==1:                     # Spluskm[1][k]
        tempM = [beta[0:m]] 
    elif m==1:                   # Spluskm[k][1]
        tempM = [beta[0:1]]
    else:                        # Other cases
        Mij = np.zeros((k, k+1, 1, 2))     # To find the sizes of blocks
        for j in range(0, k):
            m1 = np.size(SPluskm[k-1-j][m-2], 0)   
            for i in range(0, k):
                m2 = np.size(SPluskm[k-1-i][m-2], 1)           
                Mij[j][i] = [m1, m2]
            Mij[j][k] = [m1, 1]
        tempM = 0     # To find the matrix with k+1 by k blocks: Only diaonal blocks and upper off-diagonal (k,k+1) are non zero
        for j in range(0, k):   # For each row, since all blocks in each row have the same number of rows, we use hstack() to put them together
            if j==0:
                tempM_RowJ = SPluskm[k-1][m-2]    # The first block in the each row of blocks: block [1,1]
            else: 
                tempM_RowJ = np.zeros((int(Mij[j][0][0,0]), int(Mij[j][0][0,1]))) # Block[j,1] = 0 if j>1
            for i in range(1, k+1):  # The rest of the blocks in the j-th row of blocks
                if j == i:
                    tempM_RowJ = np.hstack((tempM_RowJ, SPluskm[k-1-j][m-2]))      # Block[j,j]: stack it to the right       
                elif j+1 == i:                                                     # Block[j,j+1]: stack it to the right
                    tempM_RowJ = np.hstack((tempM_RowJ, beta[m-1]*np.identity(np.size(SPluskm[k-1-j][m-2], 0))))  
                else:  # All other blocks are zero matrices
                    tempM_RowJ = np.hstack((tempM_RowJ, np.zeros((int(Mij[j][i][0,0]), int(Mij[j][i][0,1])))))
            if j==0:  # The first row of blocks
               tempM = tempM_RowJ
            else:     # The 2, 3, ..., k rows: Stack them below.
               tempM = np.vstack((tempM, tempM_RowJ))
    # print('New', tempM)    
    return tempM


##### Descripton: This function is for the construction of S(k,m).
def   Matrices_Qkm(k, m, S, StPluskm, StMinuskm, Skm):
    if k==1:
        tempM = [[0]]                # Skm[1][k]
    elif m==1:
        tempM = [[(k-1)*S[0][0]]]      # Skm[k][1]
    else:                            # Other cases
        Mij = np.zeros((k, k, 1, 2))     # To find the sizes of blocks
        for j in range(0, k):
            m1 = np.size(Skm[k-1-j][m-2], 0)   
            for i in range(0, k):
                m2 = np.size(Skm[k-1-i][m-2], 1)           
                Mij[j][i] = [m1, m2]
        # print('Mij', Mij)
        tempM = 0   # To initialize the block matrices k by k: Tri-diagonal with only blocks[j][j], [j+1][j], [j][j+1] blocks nonzero.
        for j in range(0, k):
            if j==0:
                tempM_RowJ = Skm[k-1][m-2]    # Block[1][1] (nonzero)
            elif j==1:
                # print('ddd', 5*np.array(StPluskm[k-1-j][m-2]))
                tempM_RowJ = j*np.array(StPluskm[k-1-j][m-2]) # Block[2][1] (nonzero)
            else:
                tempM_RowJ = np.zeros((int(Mij[j][0][0][0]), int(Mij[j][0][0][1])))  # Zero blocks in the first columns
            for i in range(1, k):  # Columns 2, 3, ..., k, blocks
                if j == i:     # Block[j][j]
                    #print(k, j, i, m, Skm[k-1-j][m-2])
                    #print('aaaaa', tempM_RowJ)
                    tempM_RowJ = np.hstack((tempM_RowJ, j*(S[m-1][m-1])*np.identity(np.size(Skm[k-1-j][m-2], 0)) + Skm[k-1-j][m-2]))             
                elif j+1 == i:  # Block[j][j+1]
                    #print('tttt', k, j, i, k-2-j, m, m-2, tempM_RowJ)
                    #print(StMinuskm[k-2-j][m-2])
                    tempM_RowJ = np.hstack((tempM_RowJ, StMinuskm[k-2-j][m-2]))  
                elif j == i+1:  # Block[j+1][j]
                    #print('www', k-1-j, tempM_RowJ, StPluskm[k-1-j][m-2])
                    tempM_RowJ = np.hstack((tempM_RowJ, j*np.array(StPluskm[k-1-j][m-2])))
                else:  # Other zero blocks
                    tempM_RowJ = np.hstack((tempM_RowJ, np.zeros((int(Mij[j][i][0,0]), int(Mij[j][i][0,1])))))
                #print('b=', k, j, i, m, tempM_RowJ)
            if j==0:
                tempM = tempM_RowJ
            else:
                tempM = np.vstack((tempM, tempM_RowJ))
    #print('New', tempM)    
    return tempM      

 
####This function is for the construction of S-(k,m) ##########################
def     Matrices_QMinus(k, m, s0, SMinuskm):
    if k==1:               # SMinuskm[1][k]
        tempM = s0[0:m]
    elif m==1:             # SMinuskm[k][1]
        tempM = k*s0[0:1]
    else:
        Mij = np.zeros((k+1, k, 1, 2))     # To find the size of blocks: (k+1) by k blocks 
        for j in range(0, k):
            m1 = np.size(SMinuskm[k-1-j][m-2], 0)   
            for i in range(0, k):
                m2 = np.size(SMinuskm[k-1-i][m-2], 1)           
                Mij[j][i] = [m1, m2]    
        for j in range(0,k):
            m2 = np.size(SMinuskm[k-1-j][m-2], 1)   
            Mij[k][j] = [1, m2]    #  The last row
        # print(Mij)
        tempM = 0   # The matrix is a k+1 by k blocks of matrix. Only blocks[k][k] and blocks[k+1][k] are nonzero.
        for j in range(0, k):  # Rows 1 to k 
            if j==0:
                tempM_RowJ = SMinuskm[k-1][m-2]   # Block[1][1] 
            elif j==1:
                tempM_RowJ = j*s0[m-1]*np.identity(np.size(SMinuskm[k-1-j][m-2],0))  # Block[2][1]
            else:
                tempM_RowJ = np.zeros((int(Mij[j][0][0,0]), int(Mij[j][0][0,1]))) # Blocks[k][1] = 0 for k>2
            for i in range(1, k):  # Blocks 2, 3, ..., k (i.e., [j][2], [j][3], ..., [j][k])
                if j == i:
                    tempM_RowJ = np.hstack((tempM_RowJ, SMinuskm[k-1-j][m-2]))   # Block[j][j]          
                elif j == i+1 and k-1-j > -1:                                    # Block[j+1][j]
                    tempM_RowJ = np.hstack((tempM_RowJ, j*s0[m-1]*np.identity(np.size(SMinuskm[k-1-j][m-2],0))))
                else:   # The rest of blocks: zeros
                    tempM_RowJ = np.hstack((tempM_RowJ, np.zeros((int(Mij[j][i][0,0]), int(Mij[j][i][0,1])))))
            if j==0:
                tempM = tempM_RowJ   # The first row
            else:
                tempM = np.vstack((tempM, tempM_RowJ))  # The 2, 3, ..., k rows: Stock it below.
        tempM_RowJ = np.zeros((int(Mij[k][0][0,0]), int(Mij[k][0][0,1])))  # The (k+1)-st row
        for i in range(1, k):            
            if k == i+1:
                tempM_RowJ = np.hstack((tempM_RowJ, k*s0[m-1]*np.identity(np.size(SMinuskm[0][m-2],1))))
            else:
                tempM_RowJ = np.hstack((tempM_RowJ, np.zeros((int(Mij[k][i][0,0]), int(Mij[k][i][0,1])))))
        if j==0:
            tempM = tempM_RowJ
        else:
            tempM = np.vstack((tempM, tempM_RowJ))     
    #print('New SMinuskm', tempM)    
    return tempM       


################# Construct QBD for MAP/PH/K (including PH/PH/K) by Count server for phase #######################
def  main_QBD_PHPHK_CSFP(ma, alpha_a, T_a, ms, alpha_s, T_s, K):
    # Define the arrival process
    D0 = T_a   # [[-5, 1], [2, -3]]  # D0 = T_a    
    D1 = -(np.matmul(T_a, np.ones([ma, 1])))*alpha_a   # D1 = np.matmul(np.matmul(T_a, np.ones([ma, 1])), alpha_a)
    # The service completion rates T^0 = -T*e
    T_s0 = -np.matmul(T_s, np.ones([ms, 1])) 
    
    # Construction of transition blocks: PPlus(k, m), Qminus(k, m), Q(k,m), PPlus(k,m)
    # PPlus(k,m)
    PPluskm = [[[] for m in range(0, ms)] for k in range(0, K)]
    for k in range(0, K):
        for m in range(0, ms):
            PPluskm[k][m] = Matrices_QPlus(k+1, m+1, alpha_s, PPluskm)
            # print('PPluskm = ', k, m, PPluskm[k][m])
    # QMinus(k,m)
    QMinuskm = [[[] for m in range(ms)] for k in range(K)]   # cell(K,ms);       # S-(k,m)
    for k in range(0, K):         #k=1:K
        for m in range (0, ms):   # m=1:ms
            QMinuskm[k][m] = Matrices_QMinus(k+1, m+1, T_s0, QMinuskm)
            # print('QMinuskm = ', k, m, QMinuskm[k][m])
    # For Q(k,m), we need QtPluskm and QtMinuskm
    Qkm = [[[] for k in range(ms)] for m in range(K+1)]  # cell(K+1,ms);          % Q(k,m)
    for m in range(0, ms):  # m=1:ms
        QtPluskm = [[[] for j in range(ms-1)] for k in range(K)]   #cell(K,ms-1);
        for k in range(K):   #k=1:K
            for j in range(0, m):   # j=1:m-1
                QtPluskm[k][j] = Matrices_QPlus(k+1, j+1, T_s[m][:], QtPluskm)
        #print('QtPluskm', QtPluskm)
        QtMinuskm = [[[] for j in range(ms-1)] for k in range(K)]   #(K,ms-1); ?
        for k in range(K):   #k=1:K
            for j in range(0, m):  #j=1:m-1
                tempM = np.transpose(T_s)
                tempv = [tempM[m][:]]
                tempv = np.transpose(tempv)
                #print('tempv', tempv)
                QtMinuskm[k][j] = Matrices_QMinus(k+1, j+1, tempv, QtMinuskm)
        #print('QtMinuskm', QtMinuskm)
        for k in range (K+1):   #k=1:K+1
            Qkm[k][m] = Matrices_Qkm(k+1, m+1, T_s, QtPluskm, QtMinuskm, Qkm)
        #print(m, Qkm)
    #print(Qkm)

    # %% %%%%%%%%%%  Construction of Ak,j, A0, A1, and A2 %%%%%%%%%%%%%%
    Akdown = [[] for k in range(K+1)]  # cell(1,K);        % A(k,k-1), k=1, 2, ..., K
    Akplus = [[] for k in range(K)]    # cell(1,K);        % A(k,k+1), k=0, 1, ..., K-1
    Akk = [[] for k in range (K+1)]    # cell(1,K+1);           % A(k,k), k=0, 1, ..., K
    for k in range(0,K):  # =1:K
        Akdown[k] = np.kron(np.identity(ma), QMinuskm[k][ms-1])
        #print('Akdown =', k, Akdown[k])
        Akplus[k] = np.kron(D1, PPluskm[k][ms-1])
        #print('Akplus = ', Akplus[k])
        Akk[k+1] = np.kron(D0, np.identity(np.size(Qkm[k+1][ms-1],0))) + np.kron(np.identity(ma), Qkm[k+1][ms-1])
        #print('Akk = ', Akk[k+1])
    Akk[0] = D0
    #print(Akk[0])
    #print('QMinuskm', QMinuskm[K-1][ms-1])
    #print('PPluskm', PPluskm[K-1][ms-1])
    A2 = np.kron(np.identity(ma), np.matmul(QMinuskm[K-1][ms-1], PPluskm[K-1][ms-1]))  # From level n to n-1
    #print('A2 = ', A2)
    A0 = np.kron(D1, np.identity(int(np.size(A2, 0)/ma)))   # From level n to n+1
    Akdown[K] = A2
    #print('A0 = ', A0)
    A1 = np.kron(D0, np.identity(int(np.size(A2, 0)/ma))) + np.kron(np.identity(ma), Qkm[K][ms-1])  # From level n to n
    #print('Size of A1 = ', A1)
    return  Akdown, Akplus, Akk, A0, A1, A2


     
     
     
