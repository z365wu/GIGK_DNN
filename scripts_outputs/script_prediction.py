#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 15:02:58 2025
    Part I:  
    Part II:
    Part III:
    Part IV:
        1. Predicting the stationary distribution of queue length for a new sample.
        2. Predicting the stationary distribution of queue length for a new sample located inside or outside the SCV_a and SCV_s regions of the DNN training samples.
@author: z365wu and q7he
"""
import pandas as pd
import os
import numpy as np
import random

from Sampling.QBD_for_DT_PHPHK_CSFP import Discrete_PH_PH_K
from Training.Class_training import NN_for_Queue
from Sampling.Class_sample_generation import PHPHK_sample_generation
from Sampling.QBD_for_CT_PHPHK_CSFP import CTPH_Rep_generator, CTPHD_Moments, CTPHPHK_Stationary_Queue_Length
from Prediction.Comparison_and_validation import PH_Represent, Compare_Queue_quantity_batch, Out_space_samples
from Prediction.Comparison_and_validation import Compare_Queue_quantity, df_quantity_Compare_to_latex, Compare_StatDist
from Prediction.Simulation_CT_PHPHK_Queue import Simulation_StatDist_CT # continuous case
from Prediction.Simulation_DT_PHPHK_Queue import Simulation_StatDist_DT # discrete case
from Prediction.Whitt_approximation_queue_length import Whitt1993
from Prediction.NN_model_Baron import Baron2024
from Prediction.Cross_comparison_DNNs import Cross_comparison_of_DNNs

##### Parameters #####
# K: The number of servers
queue_type = 1        #  =0.5 for mixed # =0 for discrete and =1 for continuous # queue type is a number in the range [0,1] ; proportion of continuous samples
n_max = 10            # The highest order of moments
Lmax = 500            # The maximum queue length
m_max = 10            # The maximum order of PH-representation (alpha, T)
input_dim = 2*n_max   # DNN input dimension: arrival n_max + service n_max

############ Class initilization and DNN training for ALL functions below ###########
DNN_net = NN_for_Queue(
    queue_type=queue_type, n_max=n_max, Lmax=Lmax, m_max=m_max, input_dim=input_dim, num_classes=Lmax
)
DNN_net.DNN() # creat a DNN structure


# # ####### Part I: Comparison and performance evaluation (for Section 5: Examples 1, 2, 3) #######################
# # ###### DNN Comparison for 1 sample (for continuous/discrete/mixed)
# # ###### For the mixed case, randomly generate a sample to be either discrete or continuous, each with a probability of 0.5.
# # # Figures: Queue length distribution; Tables: Queuing Quantity Comparision #####
# # # Figure: Queue length distribution by Whitt1993, Simulation, DNN, and QBD
# # # Table: Queuing quantity comparision across Whitt1993, Simulation, DNN, and QBD
# # # Plot queue length distribution of PH/PH/K by QBD, NN, simulation, and Whitt 1993
# # # for  rho = 0.27, rho = 0.66, and 0.95

# # # Generate samples for a give rho:  0.27, 0.66, 0.95
# Rho_list = [0.27, 0.66, 0.95]   # [0.66]  # List of traffic intensities (ρ) used for prediction
# for K in [1, 2, 3]: # number of servers
#     # Load NN model for each server number K
#     DNN_model = DNN_net.DNN()
#     DNN_model.load_weights(f'Output/models/model_GIGK({str(K)})_CSFP_saved_{DNN_net.queue_type}.weights.h5')
    
#     df_quantity_comparison = pd.DataFrame() # set up an empty dataframe to save queueing quantity comparision 
#     for rho_test in Rho_list:
#         # Plotting predicted stationary queue length distribution for the four methods 
#         DNN_net.Plotting_queue_length_distribution(
#             K, DNN_model, rho_lower=rho_test, rho_upper=rho_test, save_fig= True, fig_save_name = f'May_2025_{DNN_net.queue_type}'  #'Apr_2025_{DNN_net.queue_type}'
#             ) # fig saved in the directory 'Figures/prediction_compare_..._Give_a_name_you_like.png'
        
#         # Save queue length predictions
#         if DNN_net.queue_type == 'continuous':
#             os.makedirs('Output/samples/queue_length_sample', exist_ok=True)
#             file_route = f'Output/samples/queue_length_sample/Continuous_Test_queue_length_sample_K{K}_rho_{rho_test}.csv'
#         elif DNN_net.queue_type == 'discrete':
#             os.makedirs('Output/samples/queue_length_sample', exist_ok=True)
#             file_route = f'Output/samples/queue_length_sample/Discrete_Test_queue_length_sample_K{K}_rho_{rho_test}.csv'
#         elif DNN_net.queue_type == 'mixed':
#             os.makedirs('Output/samples/queue_length_sample', exist_ok=True)
#             file_route = f'Output/samples/queue_length_sample/Mixed_Test_queue_length_sample_K{K}_rho_{rho_test}.csv'
#         DNN_net.df_StatDist.to_csv(file_route, index=False)

#         # print(DNN_net.df_StatDist)

#         # Compare the queueing quantity across Whitt1993, Simulation, DNN, and QBD
#         # the probability of emprt system (q_0), the mean queue length (E[q_w]), 
#         # mean waiting time (E[W]), and MSE
#         df_sub = Compare_Queue_quantity(DNN_net.df_StatDist, Lmax, K, rho_test, DNN_net.queue_type)
#         df_quantity_comparison = pd.concat([df_quantity_comparison, df_sub], axis = 0)
        
#     # print('Compare the queueing quantity across Whitt1993, Simulation, DNN, and QBD')
#     # print(df_quantity_comparison)
#     # Convert to Latex form for Tables
#     df_compare_latex_table = df_quantity_Compare_to_latex(df_quantity_comparison, DNN_net.queue_type)
#     print(df_quantity_comparison)
#     # Save LaTeX table to a text file
#     file_name = f'Output/Tables/Test_df_Compare_latex_Tables_for_each_K{K}_{DNN_net.queue_type}.txt'
#     with open(file_name, "w", encoding="utf-8") as f:
#         f.write(df_compare_latex_table)

#     # Save queueing quantity comparision 
#     file_route_q_quantity = f'Output/Tables/Test_queue_quantity_comparison_sample_K{K}_{DNN_net.queue_type}.csv'
#     df_quantity_comparison.to_csv(file_route_q_quantity, index=False)
#     print('queue-type = ', DNN_net.queue_type)


# ########### Part II:  Batch Evaluation of samples (Section 5: Example 4) #########################
# ## Have mixed case here: For the mixed case, randomly generate samples to be either discrete or continuous, each with a probability of 0.5.#### 
# ##### DNN Comparision over batchs (only for continuous/discrete so far)
# Figures: Queue length distribution; Tables: Queuing Quantity Comparision #####
# Figure: Queue length distribution by Whitt1993, Simulation, DNN, and QBD
# Table: Queuing quantity comparision across Whitt1993, Simulation, DNN, and QBD
# Plot queue length distribution of PH/PH/K by QBD, NN, simulation, and Whitt 1993
# for  rho = 0.27, rho = 0.66, and 0.95
# Generate samples for a give rho:  0.27, 0.66, 0.95
Rho_list = [0.27, 0.66, 0.95] # List of traffic intensities (ρ) used for prediction
batch_size = 5    # to be changed to 50
epochs = 1         # to be changed to 10
df_all_quantity = pd.DataFrame()
for K in [1, 2, 3]:  #[1, 2, 3]: # number of servers   # Include K = 3 with m_max = 10 (see top)
    # Load NN model for each server number K
    DNN_model = DNN_net.DNN()
    DNN_model.load_weights(f'Output/models/model_GIGK({str(K)})_CSFP_saved_{DNN_net.queue_type}.weights.h5')
    df_summary = Compare_Queue_quantity_batch(Rho_list, batch_size, epochs, DNN_model, K, DNN_net.queue_type, m_max, n_max, Lmax)
    
    
# ############ Part III: Outlier evaluation (Section 5. Example 5.5) ###############################################
# #### Examples in or outside the SCV_a and SCV_s region of DNN training sampels
# #for continuous/discrete/mixed
# Parameters:
# - c_a_2_ubound : Upper bound for the squared coefficient of variation (SCV) of arrival times
# - c_s_2_ubound : Upper bound for the squared coefficient of variation (SCV) of service times.
# - out_sample : bool, default=True
#    If True, generates out-samples cases where SCV of arrival times (`c_a_2`) **exceeds** `c_a_2_ubound`
#    and SCV of service times (`c_s_2`) **exceeds** `c_s_2_ubound`.
#    If False, generates in-sample cases where SCV of arrival times (`c_a_2`) **is below** `c_a_2_ubound`
#    and SCV of service times (`c_s_2`) **is below** `c_s_2_ubound`.
# Rho_list = [0.27, 0.66, 0.95] # List of traffic intensities (ρ) used for prediction
# for K in [1, 2, 3]:  #, 2]  #, 3]:
#     for rho_test in Rho_list:
#         # Load NN model for each server number K
#         DNN_model = DNN_net.DNN()
#         DNN_model.load_weights(f'Output/models/model_GIGK({str(K)})_CSFP_saved_{DNN_net.queue_type}.weights.h5')
#         # Plot: outside the SCV_a and SCV_s region
#         rho_sample, c_a_2, c_s_2, df_StatDist = Out_space_samples(
#             DNN_net.queue_type, K, m_max, n_max, Lmax, DNN_model, c_a_2_ubound=0.5, c_s_2_ubound=0.5, out_sample = True, rho_lower=rho_test, rho_upper=rho_test
#         )
#         #print(f'{K} servers')
#         print(f'sample outside the SCV_a and SCV_s region: rho {rho_sample}, c_a_2 {c_a_2}, c_s_2 {c_s_2}')
#         print(f'Figures saved in: Output/Figures/Out_of_space_sample/K{K}_SCVa_{round(c_a_2,1)}_SCVs_{round(c_s_2,1)}_rho_{round(rho_sample,2)}')
        
        # # Plot: inside the SCV_a and SCV_s region
        # rho_sample, c_a_2, c_s_2, df_StatDist = Out_space_samples(
        #     DNN_net.queue_type, K, m_max, n_max, Lmax, DNN_model, c_a_2_ubound=1, c_s_2_ubound=1, out_sample = False, rho_lower=rho_test, rho_upper=rho_test
        # ) 
        # print(f'{K} servers')
        # print(f'sample in the SCV_a and SCV_s region: rho {rho_sample}, c_a_2 {c_a_2}, c_s_2 {c_s_2}')
        #print(f'Figures saved in: Output/Figures/Out_of_space_sample/K{K}_SCVa_{round(c_a_2,1)}_SCVs_{round(c_s_2,1)}_rho_{round(rho_sample,2)}')


# # ################ Part IV: Cross comparison of DNNs (Section 5, Example 6) ##################
# Rho_list = [0.27, 0.66, 0.95] # List of traffic intensities (ρ) used for prediction
# if queue_type == 1:
#     queue_type = 'continuous'
# elif queue_type == 0:
#     queue_type = 'discrete'
# for K in [1, 2, 3]:
#     for rho_test in Rho_list:
#         rho_lower = rho_test
#         rho_upper = rho_test
#         DNN_model = DNN_net.DNN()
#         df_statdist = Cross_comparison_of_DNNs(DNN_model, K, queue_type, m_max, n_max, Lmax, rho_lower, rho_upper)


# # ################## Part V: Find the accuracy rates for rho from 0, 0.1, 0.2, ..., to 1.  ##############

# # ##### Test accuracy on rho from 0 to 0.1, 0.1 to 0.2, ..., 0.9 to 1 ######
# def Accuracy_on_rho(queue_type, K, m_max, n_max, Lmax, NN_model, rho_lower, rho_upper, accuracy_bd, max_moment_bound = 1.0e+30):
   
#     if queue_type == 1 or queue_type == 'continuous': # generate continuous queue; queue_type = 1 means continuous queue.
#         # generate random paramters for continuous PD distribution
#         print('queue type: continuous')
#         rho, m_a, alpha_a, T_a, m_s, alpha_s, T_s, moments_Arrival, moments_Service = PH_Represent(1, K, m_max, n_max, Lmax, rho_lower, rho_upper, max_moment_bound = 1.0e+30)
#         # i) calculate the stationary distribution using QBD method
#         QBD_StatDist, _ = CTPHPHK_Stationary_Queue_Length(m_a, alpha_a, T_a, m_s, alpha_s, T_s, K, Lmax)
#         # iii) NN model prediction
#         Moments_A_S = np.concatenate((moments_Arrival, moments_Service)).reshape(1,-1)
#         Moments_A_S = np.log(Moments_A_S+1) # Transform: +1 and then log transform
#         NN_output = NN_model.predict(Moments_A_S)
           
#     elif queue_type == 0 or queue_type == 'discrete': # generate discrete queue; queue_type = 0 means discrete queue.
#         # generate random paramters for discrete PD distribution
#         print('queue type: discrete')
#         rho = np.random.random() * (rho_upper - rho_lower) + rho_lower
#         # i) calculate the stationary distribution using QBD method
#         m_a, alpha_a, T_a, m_s, alpha_s, T_s, moments_Arrival, moments_Service, QBD_StatDist = Discrete_PH_PH_K(0, 1, K, m_max, n_max, Lmax, rho_given = rho)
#         # iii) NN model prediction
#         moments_Arrival_log = np.log(1 + moments_Arrival)
#         moments_Service_log = np.log(1 + moments_Service) 
#         Moments_A_S = np.concatenate((moments_Arrival_log, moments_Service_log)).reshape(1,-1)
#         NN_output = NN_model.predict(Moments_A_S)
#     # combine QBD_StatDist and Simulation_StatDist_DT
#     df_StatDist = pd.DataFrame()
#     df_StatDist['NN'] = NN_output[0]
#     df_StatDist['QBD'] = QBD_StatDist
#     abs_diff = np.abs(df_StatDist['QBD'] - df_StatDist['NN']).sum()
    
#     if accuracy_bd >= abs_diff: # the prediction accuracy is below accuracy threshold
#         return 1, rho
#     else:
#         return 0, rho
    
# accuracy_bd = 0.05 # accuracy threshold
# sample_num = 1
# queue_type = 0
# for K in [1, 2, 3]:  #[1, 2, 3]: # number of servers   # Include K = 3 with m_max = 10 (see top)
#     # Load NN model for each server number K
#     Accounts = np.zeros(10)
#     DNN_model = DNN_net.DNN()
#     DNN_model.load_weights(f'Output/models/model_GIGK({str(K)})_CSFP_saved_{DNN_net.queue_type}.weights.h5')
#     for i in np.arange(0, 10):
#         rho_lower = i*0.1
#         rho_upper = i*0.1 + 0.1
#         for j in range (sample_num):
#              accuracy, rho_rand = Accuracy_on_rho(queue_type, K, m_max, n_max, Lmax, DNN_model, rho_lower, rho_upper, accuracy_bd, max_moment_bound = 1.0e+30)
#              #print(f"randomly generated rho: {rho_rand}")
#              if accuracy == 1:
#                  Accounts[i] = Accounts[i] + 1
#     print(f"Server numbers K = {K}; Accounts: {Accounts}")

