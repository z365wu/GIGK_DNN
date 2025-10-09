#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:33:21 2024
    This is used to approximate queue length using the paper of Whitt1993
@author: Zhenggao Wu (z365wu) and Haokun Zhao
"""

from math import factorial
import numpy as np
from scipy.stats import norm # compute the cumulative distribution function (CDF)
# from Sampling.QBD_for_CT_PHPHK_CSFP import  PHD_Moments, PHPHK_Stationary_Queue_Length

def Whitt1993(Lmax, m, rho, c_a_2, c_s_2):
    
    tau = 1 # This is the expected serving time and equals 1 withoutloss of generality in the setting of Whitt1993 (on page 116)
    lbd = rho * m / tau # lbd is the arriaval rate
    
    ## obtain equation (2.4) in Whitt1993
    sum_part = 0 # initial value
    for k in range(0, m): #  summation part in equation (2.4) in Whitt1993
        sum_part += (m * rho)**k / factorial(k) 
    xi =  1 / ((m * rho)**m / (factorial(m) * (1 - rho)) +  sum_part) # equation (2.4) in Whitt1993
    
    ## calculate P{W(M/M/m) > 0}: using equation (2.3) in Whitt1993 (P{W(M/M/m) > 0}: P(W>0) = P(N>=m) (see the line below (2.4))
    P_W_bigger_0 = (m * rho)**m/(factorial(m) * (1-rho)) * xi
    
    ## calculate E[W(M/M/m)] using equation (2.8) in Whitt1993 
    E_W_MMm =  P_W_bigger_0 * tau / (m * (1 - rho))
    
    ## Calculate E[W]
    # define equation (2.25) in Whitt1993
    def phi(rho, c_a_2, c_s_2, m):
        
        # prepare used paramaters/functions
        gamma = min(0.24, (1 - rho) * (m-1) * ((4+ 5 * m)**(1/2) - 2) / (16 * m * rho))# equation (2.17) in Whitt1993
        phi_1 = 1 + gamma # one line below equation (2.16) in Whitt1993
        
        phi_2 = 1 - 4 * gamma # one line below equation (2.18) in Whitt1993
        
        phi_3 = phi_2 * np.exp(-2 * (1- rho) /(3 * rho)) # one line below equation (2.20) in Whitt1993
        
        phi_4 = min(1, (phi_1 + phi_3) / 2) # equation (2.21) in Whitt1993
        
        ## calculate Psi using equtation (2.22) in Whitt1993
        c_2 = (c_a_2 + c_s_2) / 2 # from the input of Psi in equation (2.25)
        if c_2 >= 1:
            Psi = 1
        else:
            Psi = phi_4 ** (2 * (1 - c_2))
        
        # calculate phi using equation (2.25) in Whitt1993
        if c_a_2 >= c_s_2:
            phi = 4 * (c_a_2 - c_s_2) / (4 * c_a_2 - 3 * c_s_2) * phi_1 + c_s_2 / (4 * c_a_2 - 3 * c_s_2) * Psi
        else:
            phi = (c_s_2 - c_a_2) / (2 * c_a_2 + 2 * c_s_2) * phi_3 + (c_s_2 + 3 * c_a_2) / (2 * c_a_2 + 2 * c_s_2) * Psi
        
        return phi
    
    ## calculate E[W] using equtions (2.24) and (2.25) in Whitt1993
    E_W = phi(rho, c_a_2, c_s_2, m) * (c_a_2 + c_s_2) / 2 * E_W_MMm
    
    ## calculate E[Q] using equation (2.2) in Whitt1993
    E_Q = lbd * E_W
    
    ## calculate P(W > 0) using equation (3.9)
    # P(W>0) = min(pi, 1)
    def P_W(rho, c_a_2, c_s_2, m):
        z = (c_a_2 + c_s_2) / (1 + c_s_2) # equation (3.8)
        gamma = (m - m * rho - 0.5) / (m * rho * z) ** (1/2) # equation (3.5)
        
        # notation: Phi (or norm.cdf() in the Python code) is the standard normal cdf, defined below equation (2.11)
        pi_4 = min(1, (1 - norm.cdf((1 + c_s_2) * (1- rho) * m **(1/2) / (c_a_2 + c_s_2))) / (1 - norm.cdf((1 - rho) * m ** (1/2))) * P_W_bigger_0)
        pi_5 = min(1, (1 - norm.cdf(2 * (1 - rho) * m **(1/2) / (1 + c_a_2))) / (1 - norm.cdf((1 - rho) * m ** (1/2))) * P_W_bigger_0)
        pi_6 = 1 - norm.cdf((m - m * rho - 0.5) / (m * rho * z) ** (1/2))
        pi_1 = rho ** 2 * pi_4 + (1 - rho ** 2) * pi_5
        pi_2 = c_a_2 * pi_1 + (1 - c_a_2) * pi_6
        pi_3 = 2 * (1 - c_a_2) * (gamma - 0.5) * pi_2 + (1 - (2 * (1 - c_a_2) * (gamma - 0.5))) * pi_1
    
        # equation (3.10) in Whitt1993
        if m <= 6 or gamma <= 0.5 or c_a_2 >=1:
            pi = pi_1
        elif m >= 7 and gamma >= 1 and c_a_2 < 1:
            pi = pi_2
        else:
            pi = pi_3
        
        P_W = min(pi, 1) # approximate P(W(rho, c_a_2, c_s_2, m) > 0) using equation (3.9)
        
        return P_W
    
    ## calculate c_Q_2 
    # calculate c_W_2: SCV of the queue, which can be obtained from the stationary???? 
    # approximate d_s^3 using equation (4.3) 
    if c_s_2 >=1:
        d_s_3 = 3 * c_s_2 * (1 + c_s_2)
    else:
        d_s_3 = (2 * c_s_2 + 1) * (c_s_2 + 1)
    
    # calculate c_D^2 using equation (4.2)
    c_D_2 = 2 * rho - 1 + 4 * (1 - rho) * d_s_3 / (3 * (c_s_2 + 1) ** 2) 
    
    # calculatec_W_2 using the middle equation in equations (4.4): [c_D_1 + 1 - P(W>0)] / P(W>0)
    P_W_bigger_0 = P_W(rho, c_a_2, c_s_2, m) # based on the line below (4.4): using equation (3.9)
    c_W_2 = (c_D_2 + 1 - P_W_bigger_0) / P_W_bigger_0
    c_Q_2 = 1 / E_Q + c_W_2 # equation (5.6)
    
    ## calculate c_C_2 using equation (5.4)
    # approximate P(Q > 0)
    P_Q_bigger_0 = rho * P_W_bigger_0 # approximation using equation (5.2)
    
    # calculate c_C^2
    c_C_2 = P_Q_bigger_0 * c_Q_2 - 1 + P_Q_bigger_0 # using equation (5.4)
    
    ## calculate P{C = k}
    # calculate E(C) using equation (5.8)
    E_C = max(1, E_Q / P_Q_bigger_0) 
    
    P_C_k_list = [] # create the list to store P_C_k
    # C is the conditional queue length given that the queue is nonempty, i.e, C = (Q|Q>0) (in section 5.2.1)
    
    for k in range(1, Lmax): # Lmax is the maximum queue length (from 0 to 99 in our samples)
        if c_C_2 > 1 - 1/E_C + 0.02: # case 1: c_C^2 > 1 - 1/E(C) + 0.02 on page 154
            c_2 = c_C_2 # line below eqution (5.16); typo in Whitt1993; c_2 should be c_C^2 !!!
    
            # equation (5.18)
            gamma_5 = (1 + (1 - 2 / (c_2 + 1 + 1/(E_C))) ** (1/2) ) / 2
            m_1 = E_C / 2 / gamma_5
            m_2 = E_C / 2 / (1 - gamma_5)
            p_1 = 1 / m_1
            if p_1 <=1:
                p_2 = 1/ m_2
                P_C_k = gamma_5 * p_1 * (1 - p_1) ** (k-1) + (1- gamma_5) * p_2 * (1 - p_2) ** (k-1)
            else: # skip to Case 2 and use a simple geometric distribution (last sentence in Case 1 on page 154)
                p = 1 / E_C # first line on page 155
                P_C_k = p * (1 - p) ** (k - 1) # geometric distrbution
            
        elif abs(c_C_2 - 1 + 1/E_C) <= 0.02: # case 2 on page 154 
            p = 1 / E_C # first line on page 155
            P_C_k = p * (1 - p) ** (k - 1) # geometric distrbution
        
        elif (E_C ** 2 - 1) / (2 * E_C ** 2) < c_C_2 and c_C_2 <= 1 - 1 / E_C - 0.02: # case 3 on page 155
            x = E_C # based on lines above euqation (5.20)
            c_2 = c_C_2 # based on lines above euqation (5.20)
            m_1 = ((x - 1) - ((x - 1) ** 2 - 2 * x ** 2 * (1 - c_2 - 1/x)) ** (1/2)) / 2
            m_2 = ((x + 1) + ((x - 1) ** 2 - 2 * x ** 2 * (1 - c_2 - 1/x)) ** (1/2)) / 2
            p_1 = 1 / (m_1 + 1)
            p_2 = 1 / m_2
            
            P_C_k = 0 # inital value for the following summation
            for j in range(0, k): # equation (5.19)
                P_C_k += p_1 * (1 - p_1) ** j * p_2 * (1 - p_2) ** (k-j-1)
            
        elif c_C_2 <= (E_C ** 2 - 1) / (2 * E_C ** 2): # case 4 on page 155
            m_1 = (E_C - 1) / 2
            m_2 = (E_C + 1) / 2
            p_1 = 1 / (1 + m_1)
            p_2 = 1 / m_2
            
            P_C_k = 0 # inital value for the following summation (same as the case 3)
            for j in range(0, k): # equation (5.19)
                P_C_k += p_1 * (1 - p_1) ** j * p_2 * (1 - p_2) ** (k-j-1)
            
        P_C_k_list.append(P_C_k) # Note: right!!!
        
    ## calculate P{Q = k}
    P_Q_k_list = [] # create a list to store probability P{Q = k}, k = 1, 2, ..., Lmax-1
    for k in range(1, Lmax):
        P_Q_k = P_Q_bigger_0 * P_C_k_list[k-1] # based on the first paragraph of section 5.3
        P_Q_k_list.append(P_Q_k) # Store the calculated probability in the list
        
    # calculate \alpha using numerical interpolation
    for i in np.arange(0, 10000, 0.001): # we assume the range of Poisson intensity /alpha is (0, 10000) (page of equation (5.22))
        k_p_k_sum = 0 # Initialize the left hand side of equation (5.23)
        q_j_sum = 0
        for j in range(0, m +1):
            q_j_sum += i ** j * np.exp(-i) / (factorial(j))
        for k in range(0, m + 1):
            q_k = i ** k * np.exp(-i) / (factorial(k))
            k_p_k_sum += k * q_k / q_j_sum
            
        rhs = m * (rho - P_Q_bigger_0)   # Compute the right-hand side of equation (5.23)
        diff_error = k_p_k_sum - rhs # Compute the difference between LHS and RHS
        
        if i == 0:
            diff_error_0 = diff_error # Store the initial difference error for comparison
        else:
            if diff_error_0 * diff_error < 0:  # Check if the difference changes sign (i.e., crossing zero)
                alpha = i
                print(f'alpha: {alpha}')
                break # Exit the loop since alpha is found
                
        if i == 10000 - 0.001:
            raise ValueError('Error: alpha is not in the np.arange; consider increasing the upper bound of the range')
    
    p_k_list = [] # create the list to store p_k
    q_j_sum = 0
    for j in range(0, m + 1):
        q_j_sum += alpha ** j * np.exp(-alpha) / factorial(j)
    for k in range(0, m + 1):
        q_k = alpha ** k * np.exp(-alpha) / factorial(k)
        p_k_list.append(q_k / q_j_sum)
        
    ## calculate P{N=k}, k=0,1,2,...,Lmax (equation 5.21)
    P_N_k_list = []
    
    for k in range(0, Lmax):
        if k <= m:
            P_N_k_list.append(p_k_list[k])
        else:
            P_N_k_list.append(P_Q_k_list[k-m-1]) # -1 as the initial index is 0 in python list
    
    ## check the sum in the P_N_k_list
    # print(f'P_N_k: {P_N_k_list[:10]}')
    # print(f'sum: {sum(P_N_k_list[:10])}')
    
    # normalization
    P_N_k_sum = sum(P_N_k_list)
    P_N_k_list_norm = [i / P_N_k_sum for i in P_N_k_list]
    
    return P_N_k_list_norm, E_W


# ## for testing the function Whitt1993 (works for large m, see the last second paragraph on page 155)
# if __name__ == "__main__":
    
#     # Test using M/M/1
#     m = 1 # number of server
#     rho = 1/1.42857143 # traffic intensity
#     tau = 1 # mean of service time and is 1 WLOG in the seeting of Whitt1933
#     lbd = rho * m / tau # lbd is the arrival rate
#     c_a_2 = 1 # SCV of arrival time
#     c_s_2 = 1 # SCV of service time
#     Statdist, E_W = Whitt1993(10, m, rho, c_a_2, c_s_2) # Stationary queue length distribution
#     print('Whitt: ', Statdist)
    
#     # using QBD
#     # i) calculate the stationary distribution using QBD method
#     m_a = 1
#     m_s = 1
#     alpha_a = np.array([1])
#     T_a = np.array([[-0.7]])
#     alpha_s = np.array([1])
#     T_s = np.array([[-1]])
#     K = 1
#     Lmax = 100
    
#     n_max = 15     # The maximum order of moments
#     moments_Arrival = PHD_Moments(alpha_a, T_a, n_max)
#     moments_Service = PHD_Moments(alpha_s, T_s, n_max)
#     rho = 1/moments_Arrival[0] / (1 / moments_Service[0])
    
#     QBD_StatDist = PHPHK_Stationary_Queue_Length(m_a, alpha_a, T_a, m_s, alpha_s, T_s, K, Lmax)
#     print('QBD: ', QBD_StatDist[:10])
    
#     '''
#     # parameters
#     rho = 1/1.42857143 # traffic intensity
#     m = 1 # number of server
#     tau = 1 # mean of service time and is 1 WLOG in the seeting of Whitt1933
#     lbd = rho * m / tau # lbd is the arrival rate
    
#     c_a_2 = moments_Arrival[1]/moments_Arrival[0]**2 - 1 # SCV of arrival time
#     c_s_2 = moments_Service[1]/moments_Service[0]**2 - 1 # SCV of service time
    
#     Statdist, E_W = Whitt1993(10, m, rho, c_a_2, c_s_2)
#     print('Stationary distribution queue length', Statdist[0:10])
    
#     # calculate queue length
#     queue = [x for x in range(0, 100)]
#     q_length = np.dot(np.array(Statdist), np.array(queue))
#     print('Queue length: ',  q_length)
    
#     # print predicted E[W] by little's law
#     print("predicted E[W] by little's law: ", q_length / lbd)
    
#     # print predicted E[W] by Whitt1993
#     print('predicted E[W] by Whitt1993: ', E_W)
    
#     ### caluclate mean and variance of exmaple in Table 29
#     n_length = [0, 1, 2, 3, 4, 5]
#     p_length = [0.928, 0.38, 0.018, 0.008, 0.04, 0.002]
#     E_length = np.dot(np.array(n_length), np.array(p_length))
#     E_X_square =  np.dot(np.array(n_length) ** 2, np.array(p_length))   
    
#     ###
#     # i) calculate the stationary distribution using QBD method
#     m_a = 1
#     m_s = 1
#     alpha_a = np.array([1])
#     T_a = np.array([[-0.5]])
#     alpha_s = np.array([1])
#     T_s = np.array([[-1]])
#     K = 1
#     Lmax = 100
    
#     n_max = 15     # The maximum order of moments
#     moments_Arrival = PHD_Moments(alpha_a, T_a, n_max)
#     moments_Service = PHD_Moments(alpha_s, T_s, n_max)
#     rho = 1/moments_Arrival[0] / (1 / moments_Service[0])
    
#     QBD_StatDist = PHPHK_Stationary_Queue_Length(m_a, alpha_a, T_a, m_s, alpha_s, T_s, K, Lmax)
#     '''
