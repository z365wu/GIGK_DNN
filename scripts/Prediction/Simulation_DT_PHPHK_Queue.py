# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:07:02 2024
      Simulation of the discrete time PH/PH/K queue: Distribution of the queue length
@author: z365wu and q7he
"""

import time
import numpy as np
import random
import math
import pandas as pd

 
#  Generate discrete time PH-random variable
def PH_Random_Variable_Generator_DT(ma, alpha, T):
    p = random.random()   # probability to determine the initial state of the underlying Markov chain
    j0 = 0                # The initial state
    ptotal = 0            # A tentative variable
    for i in range (0, ma):   # To determine the state for which the cumulated probability is p, starting from state 1.
        ptotal = ptotal + alpha[i]
        if ptotal < p:
            j0 += 1
    ## J0 can be {0, 1, ..., ma-1} for states {1, 2, ..., ma}
    x = 0  # The PH-random variable initialized at zero.
    while j0 < ma:  # J0 can be {0, 1, ..., ma-1, ma}
        # print(j0, T[j0][j0])
        x = x + 1  # The time elapsed (or already spent on states 0, 1, ..., ma-1)
        p = random.random()  # Probability to determine the next state
        # print(x, p)
        tempv = T[j0][:]  # The j0-th row in T to determine the next state of the underlying Markov chain
        #print(tempv, j0)
        jnew = 0
        # print(tempv, j0)
        ptotal = 0
        for i in range (0, ma): 
            ptotal = ptotal + tempv[i]
            if ptotal < p:
                jnew += 1
        j0 = jnew   # This can be 0, 1, 2, ..., ma-1, and ma (note: j0 = ma means the Markov chain is absorbed, end of the process)
        # print('state: ', j0, '. Time elapsed: ', x)
    return x
 
# Obtain the stationary distribution of a discrete queue using simulation method
# Variables for simulation 
# Qmax: The maximum number of queue length to be considered 
def Simulation_StatDist_DT(ma, alpha, T, ms, beta, S, K, Qmax):
    
    # use a larger Qmax + 1 is used to the absorbe all the probability for queue length longer than Qmax
    Qmax = Qmax + 1
    # Variables for simulation 
    t_q = np.zeros(Qmax)  # To record the total time spent in which of the queue length is 0, 1, ..., Qmax
    Nmax = 1000000  # The total number of events to be generated
    v = np.ones(K+1) * 1.e15  # The residual times of services and the interarrival time: v[K] for the interarrival times
                              # v[0], v[1], ..., v[K-1] are the residual times of servers 1, 2, ..., K
    n = 0   # Number of events generated
    t_n = 0  # The current time
    q_length = 0 # The queue length
    
    ## initialization: assume the initial queue length is 0 (i.e., all servers are idle)
    a = PH_Random_Variable_Generator_DT(ma, alpha, T)  # First interarrival time
    v[K] = a    ## add the first inter-arrival time to to v_[K+1] (in [v_1, v_2,...,v_{K+1}])
    t_n = 0
    
    ## The main loop
    while n < Nmax:
          s_min, j_n = np.min(v), np.argmin(v)  ## revised: use np.argmin() to return the index of the minimal value 
          t_n = t_n + s_min                     # The next event time epoch
          v = v - s_min * np.ones(K+1)          # Updated residual times
          if q_length < Qmax-1: 
              t_q[q_length] += s_min                # Sojourn time in queue length q_length
          else: 
              t_q[Qmax-1] += s_min
          if j_n == K:  ## The next event is an arrival
              a = PH_Random_Variable_Generator_DT(ma, alpha, T) ## The next interarrival time
              v[K] = a     ## Update v[K] for the next interarrival time
              if q_length < K:  # At least one server is available for the new arrival, starting a service
                  j = np.argmax(v[0:K]) ## The first avaiable server
                  # print('j = ', j, 'v[0:K] = ', v[0:K], 'v = ', v)
                  v[j] = PH_Random_Variable_Generator_DT(ms, beta, S) ## service time
              q_length += 1    ## Queue length is increased by one
          else:  # A service completion
              if q_length > K:
                  v[j_n] = PH_Random_Variable_Generator_DT(ms, beta, S) ## service time
              else: 
                  v[j_n] = 1.e+15
              q_length -= 1     # Queue length is decreased by one      
          n += 1

    #----------------------------------------------------------------
    # Convert to a pandas Series
    series_data = pd.Series(t_q)
    # Queue length distribution
    p_d = series_data / np.sum(series_data)
    
    return p_d    

# # Testing function in this file
# if __name__ == '__main__':
    
#     # ## System parameters
#     K = 3    # The number of servers
#     # PH-representation (ma, alpha, T) for the interarrival times
#     ma = 2   # Number of phases of (alpha, T) for interarrival times
#     alpha = np.array([0.2, 0.8])
#     T = np.array([[0.2, 0.5],  [0.2, 0.4]])
#     # PH-representation (ms, beta, S) for the service times
#     ms = 3   # Number of phases of (beta, S) for service times
#     beta = np.array([0.2, 0.5, 0.3])
#     S = np.array([[0.1, 0.3, 0.3], [0.4, 0.0, 0.5], [0.1, 0.5, 0.1]])
    
#     Qmax = 100   # The maximum number of queue length to be considered
#     stat_distr = Simulation_StatDist_DT(ma, alpha, T, ms, beta, S, K, Qmax)
#     print(f'sum of Queue length distribution: {sum(stat_distr)}')
