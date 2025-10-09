#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:53:59 2025

generate PH distribution with multi-modes

@author: z365wu and q7he
"""

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from GIGK_Input_output_CSFP import PH_Rep_generator, PHD_Moments, PHPHK_Stationary_Queue_Length
from Simulation_CT_PHPHK_Queue import Simulation_StatDist # continuous case
from Whitt_approximation_queue_length import Whitt1993
from numpy.linalg import inv
from scipy.linalg import expm # Compute the matrix exponential

####### Generate parameters of a continuous time PH/PH/K queue #########
####### K, (m_a, alpha_a, T_a), (m_s, alpha_s, T_s) 
#    m_max: The maximum order of PH-representation (alpha, T)
#    n_max: The highest order of moments
#    K:     The number of servers
def  PH_Represent_Continuous(K, m_max, n_max, Lmax):
    Indicator = 0    # If moments are too big, we ignore this sample and go to the next one.
    while Indicator == 0:
        # Interarrival time: for i in range(0, m_max) for distribution of arrival time
        rho = random.random()                 # Mean interarrival time and traffic intensity       
        i_a = random.randint(0, m_max-1)
        [alpha_a, T_a] = PH_Rep_generator(i_a+1, rho*K)      # rho<K and mean for interarrival arrival time
        moments_Arrival = PHD_Moments(alpha_a, T_a, n_max)
        # Service time: for i in the range(1, m_max) for the distribution of service time
        i_s = random.randint(0, m_max-1)
        [alpha_s, T_s] = PH_Rep_generator(i_s+1, 1)    # rho=1 and E[S] = 1 for service times 
        # Note: the system rho is guaranteed by rho*K/(K*1) = rho 
        moments_Service = PHD_Moments(alpha_s, T_s, n_max) 
        # Queueing quantities: Stationary distribution of queue length   
        if max(moments_Arrival[n_max-1], moments_Service[n_max-1]) < 1.0e+30:                
            Indicator = 1

    return rho, i_a+1, alpha_a, T_a, i_s+1, alpha_s, T_s, moments_Arrival, moments_Service # return paramters for continuous PD distribution

# Generate a multiModes PH-representation (Coxian) (alpha, T) of given order m
#    m: order of PH-representation (alpha, T)
#    n_max: The highest order of moments
#    rho: = 1/mean;
#    We use different ways to generate some specific PH-distributions: Erlang, continuous/discrete PHD. 
def  PH_TwoModes_Erlang_generator(m, rho):    
    v_alpha = np.zeros(m) 
    v_alpha2 = np.zeros(m) 
    m_T = np.zeros([m, m])  
    m_T2 = np.zeros([m, m]) 
    
    ## generate PH representation 1
    v_alpha[0] = random.random()    # Normalization to sum 0.5
    # Matrix T
    tempx = random.random()/random.random()    
    for i in range(0,m):
        m_T[i,i] = -tempx # Diagonal elements (negative)
        if i<m-1:
            m_T[i,i+1] = tempx   
    
    ## generate PH representation 2
    v_alpha2[0] = random.random()     # Normalization to sum 1
    # Matrix T
    tempx = random.random()/random.random()    
    for i in range(0,m):
        m_T2[i,i] = -tempx # Diagonal elements (negative)
        if i<m-1:
            m_T2[i,i+1] = tempx  
    
    ## PH for the sum of PH 1 and generate PH 2
    t_0 = - np.sum(m_T, axis = 1, keepdims=True) # calculate t_0 for PH 1
    t_0_alpha2 = np.matmul(t_0, v_alpha2.reshape(1, -1)) # calculate the upper right block matrix
    m_T_all = np.zeros([2*m, 2*m])# PH for the sum of the two PHs  
    m_T_all[:m,:m] = m_T # the upper left corner
    m_T_all[:m,m:] = t_0_alpha2 # the upper right corner
    m_T_all[m:,m:] = m_T2 # the lower left corner
    v_alpha_all = np.zeros(2*m)
    v_alpha_all[0] = v_alpha[0]
    v_alpha_all[m] = (1-v_alpha[0]) * v_alpha2[0]

    # Normalize (v_alpha, m_T) to get E[X] = 1/rho
    temp_mean = np.sum(np.matmul(v_alpha_all, inv(-m_T_all))) # alpha*(-T)^-1*e    
    m_T_all = rho*m_T_all*temp_mean  # Normalize the mean to 1/rho
    return  v_alpha_all, m_T_all     # Return (alpha, T)

def  PH_ThreeModes_Erlang_generator(m, rho):    
    v_alpha = np.zeros(m) 
    v_alpha2 = np.zeros(m) 
    v_alpha3 = np.zeros(m) 
    m_T = np.zeros([m, m])  
    m_T2 = np.zeros([m, m]) 
    m_T3 = np.zeros([m, m]) 
    
    ## generate PH representation 1
    v_alpha[0] = random.random()    # Normalization to sum 0.5
    # Matrix T
    tempx = random.random()/random.random()    
    for i in range(0,m):
        m_T[i,i] = -tempx # Diagonal elements (negative)
        if i<m-1:
            m_T[i,i+1] = tempx   
    
    ## generate PH representation 2
    v_alpha2[0] = random.random()     # Normalization to sum 1
    # Matrix T
    tempx = random.random()/random.random()    
    for i in range(0,m):
        m_T2[i,i] = -tempx # Diagonal elements (negative)
        if i<m-1:
            m_T2[i,i+1] = tempx  

    ## generate PH representation 2
    v_alpha3[0] = random.random()     # Normalization to sum 1
    # Matrix T
    tempx = random.random()/random.random()    
    for i in range(0,m):
        m_T3[i,i] = -tempx # Diagonal elements (negative)
        if i<m-1:
            m_T3[i,i+1] = tempx 
    
    ## PH for the sum of PH 1 and generate PH 2
    t_0 = - np.sum(m_T, axis = 1, keepdims=True) # calculate t_0 for PH 1
    t_0_2 = - np.sum(m_T2, axis = 1, keepdims=True) # calculate t_0 for PH 2
    t_0_alpha2 = np.matmul(t_0, v_alpha2.reshape(1, -1)) # calculate the upper right block matrix
    t_0_alpha3 = np.matmul(t_0_2, v_alpha3.reshape(1, -1)) # calculate the upper right block matrix
    m_T_all = np.zeros([3*m, 3*m])# PH for the sum of the two PHs  
    m_T_all[:m,:m] = m_T # the upper left corner
    m_T_all[:m,m:2*m] = t_0_alpha2 # the upper right corner
    m_T_all[m:2*m,m:2*m] = m_T2 # the lower left corner
    m_T_all[m:2*m,2*m:] = t_0_alpha3 # the upper right corner
    m_T_all[2*m:, 2*m:] = m_T2 # the lower left corner
    v_alpha_all = np.zeros(3*m)
    v_alpha_all[0] = v_alpha[0]
    v_alpha_all[m] = (1-v_alpha[0]) * v_alpha2[0]
    v_alpha_all[2*m] = (1-v_alpha[0] - (1-v_alpha[0]) * v_alpha2[0]) * v_alpha3[0]
    
    # Normalize (v_alpha, m_T) to get E[X] = 1/rho
    temp_mean = np.sum(np.matmul(v_alpha_all, inv(-m_T_all))) # alpha*(-T)^-1*e    
    m_T_all = rho*m_T_all*temp_mean  # Normalize the mean to 1/rho
    return  v_alpha_all, m_T_all     # Return (alpha, T)


def  PH_MultiModes_Coxian_generator(m, rho):    
    v_alpha = np.zeros(m) 
    m_T = np.zeros([m, m])  

    # Vector alpha
    v_alpha[0] = random.random()

    # Matrix T
    on_diag_value = random.random()/random.random()
    for i in range(0,m):
        m_T[i,i] = -on_diag_value # Diagonal elements (negative)
        #if i == 0:
        #    m_T[i,i+1] = -m_T[i,i]
        if i<m-1:
            if random.random() > 0.3:  # Off-diagonal element (i, i+1); Make T from generalized Erlang to Coxian
                m_T[i,i+1] = -m_T[i,i] 
            else:
                v_alpha[i+1] = random.random()
                on_diag_value = random.random()/random.random()
    if np.sum(v_alpha) > 1:
        v_alpha = v_alpha/np.sum(v_alpha)    # Normalization
    # Normalize (v_alpha, m_T) to get E[X] = 1/rho
    temp_mean = np.sum(np.matmul(v_alpha, inv(-m_T))) # alpha*(-T)^-1*e    
    m_T = rho*m_T*temp_mean  # Normalize the mean to 1/rho
    return  v_alpha, m_T     # Return (alpha, T)


def  PH_MultiModes_PH_generator(m, rho):    
    v_alpha = np.zeros(m) 
    v_alpha2 = np.zeros(m) 
    m_T = np.zeros([m, m])  
    m_T2 = np.zeros([m, m]) 
    
    ## generate PH representation 1
    for i in range(0, m):
        v_alpha[i] = random.random()
    v_alpha = v_alpha/np.sum(v_alpha) * 0.5    # Normalization to sum 1
    # Matrix T
    # Matrix T
    for i in range(0,m):
        for j in range(0,m):
            m_T[i,j] = random.random()/random.random() # Element (i,j)
        m_T[i,i] = -np.sum(m_T[i]) 
    
    ## generate PH representation 2
    for i in range(0, m):
        v_alpha2[i] = random.random()
    v_alpha2 = v_alpha2/np.sum(v_alpha2)    # Normalization to sum 1
    # Matrix T2
    # Matrix T2
    for i in range(0,m):
        for j in range(0,m):
            m_T2[i,j] = random.random()/random.random() # Element (i,j)
        m_T2[i,i] = -np.sum(m_T2[i])
    
    ## PH for the sum of PH 1 and generate PH 2
    t_0 = - np.sum(m_T, axis = 1, keepdims=True) # calculate t_0 for PH 1
    t_0_alpha2 = np.matmul(t_0, v_alpha2.reshape(1, -1)) # calculate the upper right block matrix
    m_T_all = np.zeros([2*m, 2*m])# PH for the sum of the two PHs  
    m_T_all[:m,:m] = m_T # the upper left corner
    m_T_all[:m,m:] = t_0_alpha2 # the upper right corner
    m_T_all[m:,m:] = m_T2 # the lower left corner
    
    v_alpha_all = np.zeros(2*m)
    v_alpha_all[:m] = v_alpha
    v_alpha_all[m:] = (1-v_alpha.sum()) * v_alpha2

    # Normalize (v_alpha, m_T) to get E[X] = 1/rho
    temp_mean = np.sum(np.matmul(v_alpha_all, inv(-m_T_all))) # alpha*(-T)^-1*e    
    m_T_all = rho*m_T_all*temp_mean  # Normalize the mean to 1/rho
    return  v_alpha_all, m_T_all     # Return (alpha, T)

def ph_pdf(x, alpha, T, t0):
    e_Tx = expm(T * x)  # Matrix exponential
    return alpha @ e_Tx @ t0


#--------------------
n_max = 10   # The maximum order of moments
df_pdf = pd.DataFrame(columns=['x'])
x_values = np.arange(0, 3.1, 0.01)
df_pdf['x'] = x_values

for k in range(0, 9):
    m = random.randint(5, n_max)
    alpha, T = PH_MultiModes_Coxian_generator(m, 1)
    one_vector = np.ones([m, 1])
    t_0 = np.dot(-T, one_vector)
    # Generate x values and compute the PDF
    pdf_values = [ph_pdf(x, alpha, T, t_0)[0] for x in x_values]
    df_pdf['cdf_'+ str(k)] = pdf_values


### Plot multiple PDF samples
plt.figure(figsize=(10, 6))
for col in df_pdf.columns[1:]:  # Exclude the 'x' column
    plt.plot(df_pdf['x'], df_pdf[col], label=col)

# Add title, labels, and legend
plt.title('Examples of sampled PH distributions', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('PDF', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 4)  # Limit y-axis from 0 to 4
#plt.legend(fontsize=10, title='PDF Samples')
plt.show()

    
#### loop to generate multi Modes
###### loop to generate a pdf sample with multiple modes ##############
#--------------------
n_max = 15   # The maximum order of moments
df_pdf = pd.DataFrame(columns=['x'])
x_values = np.arange(0, 3.1, 0.05)
df_pdf['x'] = x_values
### check peaks of pdf
def has_multiple_peaks(lst):
    """
    Check if an ordered list has multiple peaks.
    :param lst: List of numbers
    :return: True if there are multiple peaks, False otherwise
    """
    peaks = []
    for i in range(1, len(lst) - 1):
        if lst[i] > lst[i - 1] and lst[i] > lst[i + 1]:
            peaks.append(i)  # Record the index of the peak

    # Return whether there are multiple peaks
    return len(peaks) >= 2, peaks

# loop until generate multiple mode sample
has_multi = False
count_it = 0
while has_multi == False:
    count_it += 1
    m = random.randint(2, n_max - 1)
    alpha, T = PH_MultiModes_Coxian_generator(m, 1)
    # exit rate t_0 
    one_vector = np.ones([m, 1])
    t_0 = np.dot(T, one_vector)
    
    # Example usage
    has_multi, peak_indices = has_multiple_peaks(pdf_values)
# add the pdf with multiple modes
df_pdf['cdf_multi_modes'] = pdf_values    


####################
# Example: The set of Coxian distributions contains the set of generalized Erlang distributions

#--------------------
n_max = 15   # The maximum order of moments
df_pdf = pd.DataFrame(columns=['x'])
x_values = np.arange(0, 3.1, 0.05)
df_pdf['x'] = x_values

m = 15
alpha = np.array([0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0, 0, 0, 0])
T = np.array([[-5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -5, 5, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -5, 5, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -5, 5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -10, 10, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -10, 10, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -10, 10, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -10, 10],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -10]])
    
one_vector = np.ones([m, 1])
t_0 = np.dot(-T, one_vector)
# Generate x values and compute the PDF
pdf_values = [ph_pdf(x, alpha, T, t_0)[0] for x in x_values]
df_pdf['cdf_1'] = pdf_values

### Plot multiple PDF samples
plt.figure(figsize=(10, 6))
plt.plot(df_pdf['x'], df_pdf['cdf_1'])

# Add title, labels, and legend
plt.title('Examples of sampled PH distributions', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('PDF', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 4)  # Limit y-axis from 0 to 4
#plt.legend(fontsize=10, title='PDF Samples')
plt.show()

