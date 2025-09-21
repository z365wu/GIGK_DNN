# -*- coding: utf-8 -*-
"""
Created on Tue May  6 17:02:50 2025
    Test QBD(PHPHK) and Simulation 
@author: he201
""" 
import numpy as np
from numpy.linalg import inv

# All functions for the stationary distribution of the Continous time PH/PH/K queues by QBD
from Sampling.GIGK_Input_output_CSFP import PH_Rep_generator, PHPHK_Stationary_Queue_Length
# All functions for the stationary distribution of the Continous time PH/PH/K queues by simulation
from Prediction.Simulation_CT_PHPHK_Queue import Simulation_StatDist_CT, Simulation_WaitingTime_PHPHK
# All functions for the stationary distributions of the discrete time PH/PH/K queues
from Sampling.QBD_for_DT_PHPHK_CSFP import DTPH_Rep_generator, main_QBD_DTPHPHK_CSFP, Discrete_PH_PH_K  # Discrete_PH_PH_K
# Simulation for the discrete time PH/PH/K queue
from Prediction.Simulation_DT_PHPHK_Queue  import Simulation_StatDist_DT


K = 3
ma = 3
ms = 3
rho = 0.8
Lmax = 500

# Arrival time PH-distribution (alpha_a, T_a)
alpha_a, T_a = PH_Rep_generator(ma, K*rho)
# Service time PH-distribution (alpha_s, T_s)
alpha_s, T_s = PH_Rep_generator(ms, 1)

print(K, rho, Lmax, ma, alpha_a, T_a, ms, alpha_s, T_s)

QBD_v_stationary, _ = PHPHK_Stationary_Queue_Length(ma, alpha_a, T_a, ms, alpha_s, T_s, K, Lmax)
print('rho =', rho,  '\n QBD: mean_queue_length =', np.dot(np.arange(500), QBD_v_stationary), '\n', QBD_v_stationary[0:6])

#print(K, rho, Lmax, ma, alpha_a, T_a, ms, alpha_s, T_s)
p_d = Simulation_StatDist_CT(ma, alpha_a, T_a, ms, alpha_s, T_s, K, Lmax)
mean_waiting = Simulation_WaitingTime_PHPHK(ma, alpha_a, T_a, ms, alpha_s, T_s, K)
print('Simulation: mean_queue_length =', np.dot(np.arange(501), p_d), 'mean_waiting=', mean_waiting, mean_waiting*K*rho)
print(p_d[0:6])
print(len(p_d), np.sum(p_d), np.sum(QBD_v_stationary), np.dot(np.arange(501), p_d), '=?', np.dot(np.arange(500), QBD_v_stationary))



# K = 3
# ma = 5
# ms = 3
# rho = 0.6
# Lmax = 500
# # Arrival time PH-distribution (alpha_a, T_a)
# alpha_a, T_a = DTPH_Rep_generator(ma)
# # Service time PH-distribution (alpha_s, T_s)
# alpha_s, T_s = DTPH_Rep_generator(ms)
# mean_a = np.sum(np.matmul(alpha_a, inv(np.eye(ma,ma)-T_a)))
# mean_s = np.sum(np.matmul(alpha_s, inv(np.eye(ms,ms)-T_s)))
# print(mean_s/(mean_a*K), rho)

# # Interarrival/service rate of the ramdonly generated PH/PH/K
# lbd_0 = 1 / np.sum(np.matmul(alpha_a, inv(np.identity(T_a.shape[0])-T_a)))
# mu_0 = 1 / np.sum(np.matmul(alpha_s, inv(np.identity(T_s.shape[0])-T_s)))
# # Adjust ramdonly generated T_s and T_a to make the PH/PH/1 have Traffic intensity rho
# delta = lbd_0 / (rho * K* mu_0)
# T_a = (1 - min(1, 1/delta)) * np.identity(T_a.shape[0]) + min(1, 1/delta) * T_a
# T_s = (1 - min(1, delta)) * np.identity(T_s.shape[0]) + min(1, delta) * T_s
# mu = 1 / np.sum(np.matmul(alpha_s, inv(np.identity(T_s.shape[0])-T_s))) # service rate
# lbd = 1 / np.sum(np.matmul(alpha_a, inv(np.identity(T_a.shape[0])-T_a))) # interarrival rate    

# mean_a = np.sum(np.matmul(alpha_a, inv(np.eye(ma,ma)-T_a)))
# mean_s = np.sum(np.matmul(alpha_s, inv(np.eye(ms,ms)-T_s)))
# print(mean_s/(mean_a*K),  rho)

# # print(K, rho, Lmax, ma, alpha_a, T_a, ms, alpha_s, T_s)
# D0 = T_a
# tempM = np.ones([ma, 1]) - np.matmul(D0, np.ones([ma, 1]))
# D1 = tempM*alpha_a
# # print('enter QBD_v_stationary')
# QBD_v_stationary, _ = main_QBD_DTPHPHK_CSFP(ma, ms, D0, D1, alpha_s, T_s, K, Lmax)
# print('rho =', rho,  '\n QBD: mean_queue_length =', np.dot(np.arange(500), QBD_v_stationary), '\n', QBD_v_stationary[0:6])

# p_d = Simulation_StatDist_DT(ma, alpha_a, T_a, ms, alpha_s, T_s, K, Lmax)
# print('Simulation: mean_queue_length =', np.dot(np.arange(501), p_d), '\n', p_d[0:6])
# print(len(p_d), np.sum(p_d), np.sum(QBD_v_stationary), np.dot(np.arange(501), p_d), '=?', np.dot(np.arange(500), QBD_v_stationary))




