# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 16:21:58 2025
      GI/M/K queue; Comparison with 
      i) DNN; ii) Exact (Hsu's book); iii) Simulation; iv) Sherzer (2025); v) Baron (2024) for K=1 only.
      GI = uniform on [a=0, b=2]
@author: he201
"""
import numpy as np
import random
import math
import pandas as pd ## revised
from scipy.special import binom

############ Exact stationary distribution of queue length of GI/M/K queue ########
# The GI/M/K queue:  i) Compute the stationary distribution of the queue length;
#                   ii) Mean queue length; and p_0
# Using the method in Hsu's book, Chapter 4 for GI/M/K queues
def GIMK_Queues(K, a, b, mu, Lmax):

    ### Compute epsilon_l, l = 1, 2, ..., K
    vec_epsilonl = np.zeros(K+1)
    vec_epsilonl[0] = 1
    for i in range(1, K+1):
        vec_epsilonl[i] = (np.exp(-a*i*mu)-np.exp(-b*i*mu))/((b-a)*i*mu)
        # print(i, vec_epsilonl[i])
    ### Compute c_k, k = 0, 1, 2, ..., K
    vec_ck = np.zeros(K+1)
    vec_ck[0] = 1
    for i in range (1, K+1):
        vec_ck[i] = vec_ck[i-1]*vec_epsilonl[i]/(1-vec_epsilonl[i])
        # print(i, vec_ck[i])
    ### Compute theta 
    theta = 0  # Initialization
    diff = 100 
    while diff > 10e-15:
        new_theta = (np.exp(-a*K*mu*(1-theta)) - np.exp(-b*K*mu*(1-theta)))/((b-a)*K*mu*(1-theta))
        diff = np.abs(new_theta - theta)
        theta = new_theta        
    # print('theta = ', theta)
    ### Compute Delta 
    Delta = 1/(1-theta)
    for k in range (1, K+1):
        Delta += ((K*(1-vec_epsilonl[k])-k)*binom(K, k))/(vec_ck[k]*(1-vec_epsilonl[k])*(K*(1-theta)-k))
    Delta = 1/Delta
    # print('Delta = ', Delta)
    ### Compute Ur 
    vec_Ur = np.zeros(K)
    vec_Ur[K-1] = Delta*vec_epsilonl[K]/(vec_ck[K]*(1-vec_epsilonl[K])*theta)
    for r in range (2,K+1):
        vec_Ur[K-r] = vec_Ur[K-r+1] + Delta*binom(K, K-r+1)*(K*(1-vec_epsilonl[K-r+1]) - (K-r+1))/(vec_ck[K-r+1]*(1-vec_epsilonl[K-r+1])*(K*(1-theta)-(K-r+1)))    
        # print('Ur{K-r} =', vec_Ur[K-r])
    for r in range (0, K):
        vec_Ur[r] *= vec_ck[r]
    
    #### Compute pi_j: Queue length distribution at arrival epochs
    vec_pi = np.zeros(Lmax)
    for i in range (K):
        for j in range (i, K):
            vec_pi[i] += np.power(-1, j-i)*binom(j, i)*vec_Ur[j]
    vec_pi[K] = Delta
    for i in range (K+1, Lmax):
        vec_pi[i] = vec_pi[i-1]*theta
    
    #### Compute p_j: Stationary queue length distribution
    vec_p = np.zeros(Lmax)
    rho = 2/(K*mu*(a+b))
    for j in range (K, Lmax):
        vec_p[j] = rho*vec_pi[j-1]
    for j in range (1, K):
        vec_p[j] = K*rho*vec_pi[j-1]/j
    vec_p[0] = 1- sum(vec_p)
    
    Eq = 0 
    for i in range (Lmax):
        Eq = Eq + i*vec_p[i]
    p0 = vec_p[0]
    ##### Test p0: Check accuracy
    p01 = 1 - rho
    for i in range (1, K):
        p01 -= K*rho*vec_pi[i-1]*(1/i - 1/K)
    # print('p0 = ', p0, 'p01 = ', p01)
    
    return vec_p, p0, Eq 

############ Simulation of the GI/M/K queue: Stationary distribution of queue length ####
# Simulation of the GI/M/K queue:
# Compute the stationary distribution, the mean queue length, and p0
def Simulation_StatDist_CT(K, a, b, mu, Lmax):
    
    Qmax = Lmax + 1 # use a larger Qmax (Qmax > Lmax)  to avoid issues caused by heavy tails (when \rho is close to 1) 
    t_q = np.zeros(Qmax)  # To record the total time spent in which of the queue length is 0, 1, ..., Qmax
    Nmax = 20000000  # The total number of events to be generated
    v = np.ones(K+1) * 1.e15  # The residual times of services and the interarrival time: v[K] for the interarrival times
                              # v[0], v[1], ..., v[K-1] are the residual times of servers 1, 2, ..., K
    n = 0   # Number of events generated
    t_n = 0  # The current time
    q_length = 0 # The queue length
    
    ## initialization: assume the initial queue length is 0 (i.e., all servers are idle)
    a0 = a + random.random()*(b-a)  # First interarrival time
    v[K] = a0    ## add the first inter-arrival time to to v_[K+1] (in [v_1, v_2,...,v_{K+1}])
    t_n = 0
    
    ## The main loop
    while n < Nmax:
          s_min, j_n = np.min(v), np.argmin(v)  ## revised: use np.argmin() to return the index of the minimal value 
          t_n = t_n + s_min                     # The next event time epoch
          v = v - s_min * np.ones(K+1)          # Updated residual times
          if q_length < Qmax-1: 
              t_q[q_length] += s_min            # Sojourn time in queue length q_length
          else: 
              t_q[Qmax-1] += s_min
          if j_n == K:  ## The next event is an arrival
              a0 = a + random.random()*(b-a)   ## The next interarrival time
              v[K] = a0     ## Update v[K] for the next interarrival time
              if q_length < K:  # At least one server is available for the new arrival, starting a service
                  j = np.argmax(v[0:K]) ## The first avaiable server
                  # print('j = ', j, 'v[0:K] = ', v[0:K], 'v = ', v)
                  v[j] = -math.log(random.random())/mu ## service time
              q_length += 1    ## Queue length is increased by one
          else:  # A service completion
              if q_length > K:
                  v[j_n] = -math.log(random.random())/mu ## service time
              else: 
                  v[j_n] = 1.e+15
              q_length -= 1     # Queue length is decreased by one      
          n += 1
          #print(n, j_n, s_min, t_n, np.sum(t_q), q_length)
          #print(v)
    
    #----------------------------------------------------------------
    # Convert to a pandas Series
    series_data = pd.Series(t_q)
    # print(t_n, np.sum(series_data))
    # Queue length distribution
    p_d = series_data / np.sum(series_data)
    p0 = p_d[0]
    Eq = 0 
    for i in range (Lmax):
        Eq = Eq + i*p_d[i]
    
    return p_d, p0, Eq


if __name__ == '__main__':
    # For testing
    K, a, b, mu = 1, 0, 2, 1.5
    Lmax = 1000
    
    p_d, Eq, p0 = GIMK_Queues(K, a, b, mu, Lmax)
    print(p_d[0:10], p0, Eq)
    p_d, Eq, p0 = Simulation_StatDist_CT(K, a, b, mu, Lmax)
    print(p_d[0:10], p0, Eq)
