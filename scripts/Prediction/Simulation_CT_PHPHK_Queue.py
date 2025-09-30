# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:07:02 2024
      Simulation of continuous time PH/PH/K: Distribution of the queue length
@author: z365wu and q7he
"""
# continuous case

import time
import numpy as np
import random
import math
import pandas as pd ## revised
import matplotlib.pyplot as plt


#  Generate PH-random variable
def PH_Random_Variable_Generator(ma, alpha, T):
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
        x = x + (np.log(random.random())/T[j0][j0])  # The time elapsed (or already spent on states 0, 1, ..., ma-1)
        p = random.random()  # Probabilty to determine the next state
        # print(x, p)
        tempv = T[j0][:]  # The j0-th row in T to determine the next state of the underlying Markov chain
        #print(tempv, j0)
        tempv = [y / (-tempv[j0]) for y in tempv] ## add for loop to calculate list/int; for normalization?
        tempv[j0] = 0
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
 
# Variables for simulation 
# Qmax: The maximum number of queue length to be considered 
def Simulation_StatDist_CT(ma, alpha, T, ms, beta, S, K, Lmax):
    
    Qmax = Lmax + 1 # use a larger Qmax (Qmax > Lmax)  to avoid issues caused by heavy tails (when \rho is close to 1) 
    t_q = np.zeros(Qmax)  # To record the total time spent in which of the queue length is 0, 1, ..., Qmax
    Nmax = 1000000  # The total number of events to be generated
    v = np.ones(K+1) * 1.e15  # The residual times of services and the interarrival time: v[K] for the interarrival times
                              # v[0], v[1], ..., v[K-1] are the residual times of servers 1, 2, ..., K
    n = 0   # Number of events generated
    t_n = 0  # The current time
    q_length = 0 # The queue length
    
    ## initialization: assume the initial queue length is 0 (i.e., all servers are idle)
    a = PH_Random_Variable_Generator(ma, alpha, T)  # First interarrival time
    v[K] = a    ## add the first inter-arrival time to to v_[K+1] (in [v_1, v_2,...,v_{K+1}])
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
              a = PH_Random_Variable_Generator(ma, alpha, T) ## The next interarrival time
              v[K] = a     ## Update v[K] for the next interarrival time
              if q_length < K:  # At least one server is available for the new arrival, starting a service
                  j = np.argmax(v[0:K]) ## The first avaiable server
                  # print('j = ', j, 'v[0:K] = ', v[0:K], 'v = ', v)
                  v[j] = PH_Random_Variable_Generator(ms, beta, S) ## service time
              q_length += 1    ## Queue length is increased by one
          else:  # A service completion
              if q_length > K:
                  v[j_n] = PH_Random_Variable_Generator(ms, beta, S) ## service time
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
    
    return p_d # return Queue length distribution
    
####  Simulation of waiting time ##############
def Simulation_WaitingTime_PHPHK(ma, alpha, T, ms, beta, S, K):
    N_max = 1000000
    mean_waiting = 0
    Customers = np.ones([N_max, 2])   # N_max customers each with (arrival time, departure time)
    q_e = 0    ## the earliest cutomer in queue
    q_l = 0    ## The latest customer in queue
    Customers[0][0] = PH_Random_Variable_Generator(ma, alpha, T) ## The first customer's arrival time
    Customers[0][1] = Customers[q_e][0] + PH_Random_Variable_Generator(ms, beta, S) ## first customer's departure time
    current_time = Customers[0][0] 
    # print('t = 0: q_e, q_l ', q_e, q_l, '\n', Customers[q_e:q_l+1])
    while  q_l < N_max-1: 
        a = PH_Random_Variable_Generator(ma, alpha, T) ## The next interarrival time 
        current_time = current_time + a
        q_l += 1  # A new customer
        Customers[q_l][0] = current_time   # The arrival time of the q_l's customer
        if q_e + K > q_l:  # A server is available to serve the customer, schedule its service and set departure time
            Customers[q_l][1] = current_time + PH_Random_Variable_Generator(ms, beta, S)
        tempqe = q_e
        for i in range(tempqe, np.min([tempqe+K, q_l])):    # Remove some customers from the queue
            # print('i=', i, 'currenttime=', current_time, 'departuretime=', Customers[i][1])
            if current_time > Customers[i][1]:   # a departure
                if q_e + K < q_l+1:  # A server becomes available, a customer enters the server and set departure time
                    Customers[q_e+K][1] = np.max([Customers[q_e+K][0], Customers[i][1]]) + PH_Random_Variable_Generator(ms, beta, S)
                q_e += 1
        
        # print('before q_e, q_l ', q_e, q_l, '\n', Customers[q_e:q_l+1])
        tempM = Customers[q_e:np.min([q_e+K, q_l+1])] 
        tempM = tempM[tempM[:, 1].argsort()]
        Customers[q_e:np.min([q_e+K, q_l+1])] = tempM
        # print('after \n', Customers[q_e:q_l+1])
    
    Waiting_times = np.zeros(N_max-(q_l-q_e))
    for n in range (0, N_max-(q_l-q_e)):
        Waiting_times[n] = Customers[n][1] - Customers[n][0]
    mean_waiting = np.mean(Waiting_times)
    # print(Customers)
       
    return mean_waiting   # return mean waiting time
 

# ##### Testing functions in this file #########
# if __name__ == '__main__':
    
#     ## System parameters
#     K = 1    # The number of servers
#     # PH-representation (ma, alpha, T) for the interarrival times
#     ma = 2   # Number of phases of (alpha, T) for interarrival times
#     alpha = np.array([0.2, 0.8])
#     T = np.array([[-2, 1],  [2, -5]])
#     # PH-representation (ms, beta, S) for the service times
#     ms = 3   # Number of phases of (beta, S) for service times
#     beta = np.array([0.2, 0.5, 0.3])
#     S = np.array([[-6, 1, 1], [0.4, -6, 0.5], [1, 0.5, -8.5]])
    
#     # K = 1
#     # ma = 1
#     # alpha = np.array([1])
#     # T = np.array([[-1.96]])
#     # ms = 1
#     # beta = np.array([1])
#     # S = np.array([[-2]])

#     mean_a = np.sum(np.matmul(alpha, np.linalg.inv(-T)))  # Mean interarrival time
#     mean_s = np.sum(np.matmul(beta, np.linalg.inv(-S)))   # Mean service time
#     rho = mean_s/(mean_a*K)                      # Traffic intensiy
#     print('rho = ', rho, 'K = ', K, 'mean_a = ', mean_a, 'mean_s = ', mean_s)
    
    
#     if rho < 1:  # The queue can reach stability 
#         # Variables for simulation 
#         Qmax = 1000   # The maximum number of queue length to be considered
#         t_q = np.zeros(Qmax)  # To record the total time spent in which of the queue length is 0, 1, ..., Qmax
#         Nmax = 1000000  # The total number of events to be generated
#         v = np.ones(K+1) * 1.e15  # The residual times of services and the interarrival time: v[K] for the interarrival times
#                                   # v[0], v[1], ..., v[K-1] are the residual times of servers 1, 2, ..., K
#         n = 0   # Number of events generated
#         t_n = 0  # The current time
#         q_length = 0 # The queue length
        
#         ## initialization: assume the initial queue length is 0 (i.e., all servers are idle)
#         a = PH_Random_Variable_Generator(ma, alpha, T)  # First interarrival time
#         v[K] = a    ## add the first inter-arrival time to to v_[K+1] (in [v_1, v_2,...,v_{K+1}])
#         t_n = 0
        
#         ## The main loop
#         while n < Nmax:
#               s_min, j_n = np.min(v), np.argmin(v)  ## revised: use np.argmin() to return the index of the minimal value 
#               t_n = t_n + s_min                     # The next event time epoch
#               v = v - s_min * np.ones(K+1)          # Updated residual times
#               if q_length < Qmax-1: 
#                   t_q[q_length] += s_min                # Sojourn time in queue length q_length
#               else: 
#                   t_q[Qmax-1] += s_min
#               if j_n == K:  ## The next event is an arrival
#                   a = PH_Random_Variable_Generator(ma, alpha, T) ## The next interarrival time
#                   v[K] = a     ## Update v[K] for the next interarrival time
#                   if q_length < K:  # At least one server is available for the new arrival, starting a service
#                       j = np.argmax(v[0:K]) ## The first avaiable server
#                       # print('j = ', j, 'v[0:K] = ', v[0:K], 'v = ', v)
#                       v[j] = PH_Random_Variable_Generator(ms, beta, S) ## service time
#                   q_length += 1    ## Queue length is increased by one
#               else:  # A service completion
#                   if q_length > K:
#                       v[j_n] = PH_Random_Variable_Generator(ms, beta, S) ## service time
#                   else: 
#                       v[j_n] = 1.e+15
#                   q_length -= 1     # Queue length is decreased by one      
#               n += 1
#               #print(n, j_n, s_min, t_n, np.sum(t_q), q_length)
#               #print(v)
        
#         #----------------------------------------------------------------
#         # Convert to a pandas Series
#         series_data = pd.Series(t_q)
#         # print(t_n, np.sum(series_data))
#         # Queue length distribution
#         p_d = series_data  / np.sum(series_data)
#         print(p_d[0:5])
#         print(f'sum of Queue length distribution: {sum(p_d)}', 'rho= ', rho)
#         print('Simulation Mean queue length = ', np.dot(np.arange(Qmax), p_d), '\nTheoretical mean queue length = ', rho/(1-rho))

        
#         # Plot the occurrences
#         plt.figure(figsize=(10, 6))
#         p_d.iloc[0:].plot(kind='bar', color='blue', width=0.8)
#         plt.title("Queue length distribution")
#         plt.xlabel("Queue length")
#         plt.ylabel("Probability")
#         plt.xticks(rotation=45)  # Rotate x-axis labels for readability
#         plt.tight_layout()
#         plt.show()
#         #-------------------------
#     else: # Queue is unstable
#         print('Rho > 1. The queue is unstable.')


