# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:53:10 2024
   This is for the DNN model for the GI/G/K queue using CSFP for both discrete and continuous PH/PH/K queues
@author: z365wu and q7he
"""
import numpy as np
import random
from numpy.linalg import inv
# from numpy.linalg import eig
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm # Compute the matrix exponential
import pickle

# Generate a PH-representation (Coxian) (alpha, T) of given order m
#    m: order of PH-representation (alpha, T)
#    rho: = 1/mean;
#    We use different ways to generate some specific PH-distributions: Erlang, continuous/discrete PHD. 
def  CTPH_Rep_generator(m, rho):    
    v_alpha = np.zeros(m)  
    m_T = np.zeros([m, m])  
    tempp = random.random()  # For separating different types of PH-representations
    if tempp < 0.25:    # Not structured PH-Rep: 25% of such distributions
        # Vector alpha of size m
        for i in range(0, m):
            v_alpha[i] = random.random()
        v_alpha = v_alpha/np.sum(v_alpha)    # Normalization to sum 1
        # Matrix T
        for i in range(0,m):
            for j in range(0,m):
                m_T[i,j] = random.random()/random.random() # Element (i,j)
            m_T[i,i] = -np.sum(m_T[i])
        m_T_type = 'PH-Rep'
    elif tempp < 0.5:   #  Erlang distribution: 25% of such distributions
        # Vector alpha
        v_alpha[0] = 1
        # Matrix T
        tempx = random.random()/random.random()    
        for i in range(0,m):
            m_T[i,i] = -tempx # Diagonal elements (negative)
            if i<m-1:
                m_T[i,i+1] = tempx 
        m_T_type = 'Erlang'
    else:   # General Coxian distribution: 50% Coxian distributions
        # Vector alpha
        v_alpha[0] = random.random()
        # Matrix T
        on_diag_value = random.random()/random.random()
        for i in range(0,m):
            m_T[i,i] = -on_diag_value # Diagonal elements (negative)
            if i<m-1:
                if random.random() > 0.2:  # Off-diagonal element (i, i+1); Make T from generalized Erlang to Coxian
                    m_T[i,i+1] = -m_T[i,i] 
                else:
                    v_alpha[i+1] = random.random()
                    on_diag_value = random.random()/random.random()
        if np.sum(v_alpha) > 1:
            v_alpha = v_alpha/np.sum(v_alpha)    # Normalization
        m_T_type = 'coxian'
    # Normalize (v_alpha, m_T) to get E[X] = 1/rho
    v_alpha = v_alpha/np.sum(v_alpha)
    temp_mean = np.sum(np.matmul(v_alpha, inv(-m_T))) # alpha*(-T)^-1*e    
    m_T = rho*m_T*temp_mean  # Normalize the mean to 1/rho
    return  v_alpha, m_T #, m_T_type     # Return (alpha, T)


#####  PHD moments generator  (continuous time case) ##### 
#    PH-representation (alpha, T)
#    n_max: The highest order of moments
def  CTPHD_Moments(v_alpha, m_T, n_max):
     # Moments up to n_max
     v_alpha = [v_alpha]
     temp_invT = -inv(m_T)
     v_moments = np.zeros(n_max)
     temp_v = np.matmul(v_alpha, temp_invT)
     for i in range(0, n_max):
         v_moments[i] = np.sum(temp_v)   # The (i+1)st moment: (i+1)! * alpha*((-T)^{-1})^{i+1}*e
         temp_v = (i+2)*np.matmul(temp_v, temp_invT)
     return v_moments   # Return the first n_max moments


########## This function is for the construction of S+(k,m) ##############
def  CT_Matrices_QPlus(k, m, beta, SPluskm):
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
def   CT_Matrices_Qkm(k, m, S, StPluskm, StMinuskm, Skm):
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
def     CT_Matrices_QMinus(k, m, s0, SMinuskm):
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
def  main_QBD_CTPHPHK_CSFP(ma, alpha_a, T_a, ms, alpha_s, T_s, K):
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
            PPluskm[k][m] = CT_Matrices_QPlus(k+1, m+1, alpha_s, PPluskm)
            # print('PPluskm = ', k, m, PPluskm[k][m])
    # QMinus(k,m)
    QMinuskm = [[[] for m in range(ms)] for k in range(K)]   # cell(K,ms);       # S-(k,m)
    for k in range(0, K):         #k=1:K
        for m in range (0, ms):   # m=1:ms
            QMinuskm[k][m] = CT_Matrices_QMinus(k+1, m+1, T_s0, QMinuskm)
            # print('QMinuskm = ', k, m, QMinuskm[k][m])
    # For Q(k,m), we need QtPluskm and QtMinuskm
    Qkm = [[[] for k in range(ms)] for m in range(K+1)]  # cell(K+1,ms);          % Q(k,m)
    for m in range(0, ms):  # m=1:ms
        QtPluskm = [[[] for j in range(ms-1)] for k in range(K)]   #cell(K,ms-1);
        for k in range(K):   #k=1:K
            for j in range(0, m):   # j=1:m-1
                QtPluskm[k][j] = CT_Matrices_QPlus(k+1, j+1, T_s[m][:], QtPluskm)
        #print('QtPluskm', QtPluskm)
        QtMinuskm = [[[] for j in range(ms-1)] for k in range(K)]   #(K,ms-1); ?
        for k in range(K):   #k=1:K
            for j in range(0, m):  #j=1:m-1
                tempM = np.transpose(T_s)
                tempv = [tempM[m][:]]
                tempv = np.transpose(tempv)
                #print('tempv', tempv)
                QtMinuskm[k][j] = CT_Matrices_QMinus(k+1, j+1, tempv, QtMinuskm)
        #print('QtMinuskm', QtMinuskm)
        for k in range (K+1):   #k=1:K+1
            Qkm[k][m] = CT_Matrices_Qkm(k+1, m+1, T_s, QtPluskm, QtMinuskm, Qkm)
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

# main_QBD_CTPHPHK_CSFP()


#    m: Order of PH-representations
#    (alpha_a, T_a): PH-representation for the interarrival time
#    (alpha_s, T_s): PH-representation for the service time
#    Lmax: Maximum queue length (truncation point)
def  CTPHPHK_Stationary_Queue_Length(ma, alpha_a, T_a, ms, alpha_s, T_s, K, Lmax):
    # Construct QBD
    ###### The QBD model with K boundary levels ######  A0 is down, A2 is up
    [Akdown, Akplus, Akk, A0, A1, A2] = main_QBD_CTPHPHK_CSFP(ma, alpha_a, T_a, ms, alpha_s, T_s, K)
    # [Akdown, Akplus, Akk, A0, A1, A2] = main_QBD_PHPHK_TPFS(ma, alpha_a, T_a, ms, alpha_s, T_s, K)    
    ### Iteration for matrix R: R = A0*(-A1)^{-1} + R^2*A2*(-A1)^{-1}
    m_R = np.zeros([np.size(A1,0), np.size(A1,0)])
    C0 = np.matmul(A0, inv(-A1))
    C2 = np.matmul(A2, inv(-A1))
    Iter_max = 5000      # maximum number of iterations
    epslon_err = 1.0e-15 # error bound
    Iter_num = 0         # Actual number of iterations
    error_sum = 1.0e10   # Actual error
    print('R iteration start with R size = ', np.size(A1,0), ': ')
    while Iter_num < Iter_max and error_sum > epslon_err:
        m_R_new = C0 + np.matmul(m_R, np.matmul(m_R, C2))    
        error_sum = sum(sum(abs(m_R-m_R_new)))
        Iter_num = Iter_num + 1   
        m_R = m_R_new
    print('R iteration number: ', Iter_num)
    # print('R=', m_R)
    #print('R = ', eig(m_R)) ## np.linalg.eig(m) 
    #### Boundary R0, R1, ..., RK for levels 0, 1, ..., K. #########
    Rk = [[] for k in range (0, K+1)]
    Rk[K] = m_R    
    for k in range (K-1, -1, -1):
        Rk[k] = np.matmul(Akplus[k], inv(-Akk[k+1] - np.matmul(Rk[k+1], Akdown[k+1])))   
    ##### (pi0, pi1, ..., piK): boundary probabilities for levels 0, 1, ..., K
    pik = [[] for k in range (0, K+1)]
    pik[0] = np.zeros([1,ma])
    Q00 = Akk[0] + np.matmul(Rk[0], Akdown[0])    # Q-matrix for level 0. pi(0)*Q00 = 0
    # Q00 is for the censored Markov chain for level 0 only. Thus, we can find pik(0)*c, where c can be obtained by normalization
    for i in range(0, ma):  # pi(0)*Q1(00) = (1, 0, ..., 0), where Q1(00) is obtained by replacing the first column of Q00 by one    
        Q00[i,0] = 1.0
    tempM = inv(Q00)   # tempM = inv(Q1(00)) 
    pik[0] = tempM[0,:]    # pi(0) is the first row of matrix tempM
    for k in range (1, K+1): 
        pik[k] = np.matmul(pik[k-1], Rk[k-1])    # pi(k) = pi(k-1)*R(k-1)
    ###### Stationary queue length distribution: matrix geometrix solution  ### 
    v_stationary = np.zeros(Lmax)
    for k in range (0, K):
        v_stationary[k] = np.sum(pik[k])   # pi(k)*e = P{q(t) = k}
    temp_v = pik[K]
    for n in range(K, Lmax):
        v_stationary[n] = np.sum(temp_v)   # pi(n) = pi(K)*R^{n-K}
        temp_v = np.matmul(temp_v, m_R)
    # Normalization of the stationary vector to make it suming to one 
    v_stationary = v_stationary/np.sum(v_stationary)
    # print(v_stationary)
    return  v_stationary, Iter_num
        

#    m_max: The maximum order of continous PH-representation (alpha, T)
#    n_max: The highest order of moments
#    K:     The number of servers
def  Input_Output_Moments_Generator_Continuous(Sample_size, K, m_max, n_max, Lmax, max_moment_bound, rho_lower, rho_upper):
     Moments = np.zeros((Sample_size, 2*n_max))               # Moments of interarrival and service times (return)
     Stationary_distribution = np.zeros((Sample_size, Lmax))  # Statioanry distributions of queue length (return)
     SCVs = np.zeros((Sample_size, 2))  # SCVs of our distributions (return): To demonstrate the versatility of samples
     Rhos = np.zeros((Sample_size, 1))  # Traffice intensity of queues (return): To demonstrate the versatility of samples
     R_Iter_num = np.zeros((Sample_size, 1))  #  R iteration number for stationary distribution of queue length
     queue_time_type = [] # To save samples into a csv file (Excel file)
     #M_T_type_a = [] # type of m_T (e.g., Coxian, PH, Erlang) for arrival processes
     #M_T_type_s = [] # type of m_T (e.g., Coxian, PH, Erlang) for service processes
     
     for j in range(0, Sample_size):
        print('- A continuous time PH/PH/K sample')
        # add queue time type: continuous time
        queue_time_type.append('continuous')
        Indicator = 0    # If moments are too big, we igonore this sample and go to the next one.
        while Indicator == 0:
            # Interarrival time: for i in range(0, m_max) for distribution of arrival time
            rho = random.random() * (rho_upper - rho_lower) + rho_lower               # Generate mean interarrival time and traffic intensity within [rho_lower, rho_upper]  
            i_a = random.randint(0, m_max-1)
            alpha_a, T_a = CTPH_Rep_generator(i_a+1, rho*K)      # rho<K and mean for interarrival arrival time
            moments_Arrival = CTPHD_Moments(alpha_a, T_a, n_max)
            Moments[j, 0:n_max] = moments_Arrival
            SCVs[j, 0] = moments_Arrival[1]/(moments_Arrival[0]*moments_Arrival[0]) - 1  # SCV for the arrival time            
            Rhos[j] = rho
            # Service time: for i in the range(1, m_max) for the distribution of service time
            i_s = random.randint(0, m_max-1)
            alpha_s, T_s = CTPH_Rep_generator(i_s+1, 1)    # rho=1 and E[S] = 1 for service times 
            # Note: the system rho is guaranteed by rho*K/(K*1) = rho 
            moments_Service = CTPHD_Moments(alpha_s, T_s, n_max)
            Moments[j, n_max:2*n_max] = moments_Service
            SCVs[j, 1] = moments_Service[1]/(moments_Service[0]*moments_Service[0]) - 1  # SCV for the service time   
              
            # Queueing quantities: Stationary distribution of queue length   
            print('C: Sample j =', j+1, 'with (K =', K, 'ma =', i_a+1, 'and ms =', i_s+1, ') of a total of', Sample_size, 'samples.')            
            if max(moments_Arrival[n_max-1], moments_Service[n_max-1]) < max_moment_bound:                
                Indicator = 1
                PHPH1StatDist, R_Iter = CTPHPHK_Stationary_Queue_Length(i_a+1, alpha_a, T_a, i_s+1, alpha_s, T_s, K, Lmax)
                Stationary_distribution[j, :] = PHPH1StatDist 
                # Log transform of the moments to make those numbers smaller for DNN training
                for i in range(0, 2*n_max):
                    Moments[j,i] = np.log(1+Moments[j,i])
                
                # R iteration number for stationary distribution of queue length
                R_Iter_num[j] = R_Iter

     return Moments, Stationary_distribution, SCVs, Rhos, queue_time_type, R_Iter_num


# Function to compute PDF using the PH representation
# x: value of the distribution
# alpha, T: PH-representation
# t0: np.dot(-T, one_vector)
### caluclate e_Tx = I + Tx + T^2x^2/2! + T^3x^3/3! + ..... 
def CTPH_pdf(x, alpha, T, t0):
    e_Tx = expm(T * x)  # Matrix exponential
    return alpha @ e_Tx @ t0     

# ####  Test functions in this file #########     
# #### Generate pdf for a random variable X with a Continuous-Time Phase-Type (PH) Representation###
# if __name__ == "__main__":
    
#     #------------------------------------------
#     # generate multiple pdfs of PH distribution samples
#     m_max = 15   # The maximum order of PH-representation
#     df_pdf = pd.DataFrame(columns=['x'])
#     x_values = np.arange(0, 3.1, 0.01)
#     df_pdf['x'] = x_values
    
#     df_save = [] #initialize a empty set to save the samples
#     df_sub = []
    
#     for k in range(0, 100):
#         m = random.randint(1, m_max - 1)
#         alpha, T = CTPH_Rep_generator(m, 1)
#         one_vector = np.ones([m, 1])
#         t_0 = np.dot(-T, one_vector)
#         # Generate x values and compute the PDF
#         pdf_values = [CTPH_pdf(x, alpha, T, t_0)[0] for x in x_values]
#         df_pdf['cdf_'+ str(k)] = pdf_values
#         df_sub = [k, alpha, T, pdf_values]
#         df_save.append(df_sub)
    
#     # save random samples
#     df_pdf.to_csv('samples/PH-distribution_samples.csv', index=False)
#     with open("samples/PH-distribution_samples_list.txt", "w") as file:
#         for item in df_save:
#             file.write(f"{item}\n")
#     with open("samples/PH-distribution_samples_list.pkl", "wb") as file:
#         pickle.dump(df_save, file)
    
#     # Load the list from the pickle file
#     '''
#     df_pdf = pd.read_csv('samples/PH-distribution_samples.csv')
#     with open("samples/PH-distribution_samples_list.pkl", "rb") as file:
#         df_save = pickle.load(file)
#     '''
    
#     ### Plot multiple PDF samples np.r_[3, 6, 15, 33, 35, 79, 80:88,99]
#     plt.figure(figsize=(10, 6))
#     for idx, col in enumerate(df_pdf.columns[[3, 6, 15, 33, 35, 79, 83,99, 10, 20, 85, 89, 91, 93, 94]]):  # Exclude the 'x' column
#         plt.plot(df_pdf['x'], df_pdf[col], label=col)
    
#     # Add title, labels, and legend
#     plt.title('Examples of sampled PH distributions', fontsize=14)
#     plt.xlabel('x', fontsize=12)
#     plt.ylabel('PDF', fontsize=12)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.ylim(0, 4)  # Limit y-axis from 0 to 4
#     #plt.legend(fontsize=10, title='PDF Samples')
#     plt.show()
    
