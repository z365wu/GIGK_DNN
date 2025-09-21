#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 20:07:29 2025
    Cross comparison of DNNs three types of DNN
@author: z365wu and q7he
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Prediction.Comparison_and_validation import PH_Represent
from Sampling.QBD_for_CT_PHPHK_CSFP import CTPHPHK_Stationary_Queue_Length
from Prediction.Simulation_CT_PHPHK_Queue import Simulation_StatDist_CT # continuous case
from Prediction.Simulation_DT_PHPHK_Queue import Simulation_StatDist_DT # discrete case
from Prediction.Whitt_approximation_queue_length import Whitt1993
from Prediction.NN_model_Baron import Baron2024
from Sampling.QBD_for_DT_PHPHK_CSFP import Discrete_PH_PH_K


################ Part IV: Cross comparison of DNNs (Section 5, Example 6) ##################
def Cross_comparison_of_DNNs(DNN_model, K, queue_type, m_max, n_max, Lmax, rho_lower, rho_upper):

    # generate random paramters for PH distribution
    rho, m_a, alpha_a, T_a, m_s, alpha_s, T_s, moments_Arrival, moments_Service = PH_Represent(queue_type, K, m_max, n_max, Lmax, rho_lower, rho_upper)
    
    Moments_A_S = np.concatenate((moments_Arrival, moments_Service)).reshape(1,-1)
    Moments_A_S = np.log(Moments_A_S+1) # Transform: +1 and then log transform
    
    if queue_type == 'continuous': # generate continuous queue
        # generate random paramters for PH distribution
        rho, m_a, alpha_a, T_a, m_s, alpha_s, T_s, moments_Arrival, moments_Service = PH_Represent(queue_type, K, m_max, n_max, Lmax, rho_lower, rho_upper)
        # i) calculate the stationary distribution using QBD method
        QBD_StatDist = CTPHPHK_Stationary_Queue_Length(m_a, alpha_a, T_a, m_s, alpha_s, T_s, K, Lmax)
        QBD_StatDist = QBD_StatDist[0]
        # ii) calculate the stationary distribution using simulation method
        Simu_StatDist = Simulation_StatDist_CT(m_a, alpha_a, T_a, m_s, alpha_s, T_s, K, Lmax)
        
    else: # queue_type == 'discrete':
        m_a, alpha_a, T_a, m_s, alpha_s, T_s, moments_Arrival, moments_Service, QBD_StatDist = Discrete_PH_PH_K(1, 1, K, m_max, n_max, Lmax, rho_given = rho)
        # ii) calculate the stationary distribution using simulation method
        Simu_StatDist = Simulation_StatDist_DT(m_a, alpha_a, T_a, m_s, alpha_s, T_s, K, Lmax)
    
    Moments_A_S = np.concatenate((moments_Arrival, moments_Service)).reshape(1,-1)
    Moments_A_S = np.log(Moments_A_S+1) # Transform: +1 and then log transform
    
    # iii) Stationary distribution by Whitt1993
    m = K
    c_a_2 = moments_Arrival[1]/moments_Arrival[0]**2 - 1 # SCV of arrival time
    c_s_2 = moments_Service[1]/moments_Service[0]**2 - 1 # SCV of service time
    print('SCV:', c_a_2, c_s_2)
    StatDistWhitt1993, E_W_Whitt = Whitt1993(Lmax, m, rho, c_a_2, c_s_2) # Return stationary queue length distribution and expected waiting time 

    # iv) using continuous DNN model
    DNN_model.load_weights(f'Output/models/model_GIGK({str(K)})_CSFP_saved_continuous.weights.h5')
    continuous_NN_output = DNN_model.predict(Moments_A_S)
    
    # v) using discrete DNN model; overwrite the previously loaded weights in DNN_model
    DNN_model.load_weights(f'Output/models/model_GIGK({str(K)})_CSFP_saved_discrete.weights.h5')
    discrete_NN_output = DNN_model.predict(Moments_A_S)
    
    # vi) using mixed DNN model; overwrite the previously loaded weights in DNN_model
    DNN_model.load_weights(f'Output/models/model_GIGK({str(K)})_CSFP_saved_mixed.weights.h5')
    mix_NN_output = DNN_model.predict(Moments_A_S)
    
    # vi) Opher2024
    if K == 1:
        Opher2024_dist = Baron2024(moments_Arrival, moments_Service)
    
    # combine QBD_StatDist and Simulation_StatDist
    df_StatDist = pd.DataFrame()
    df_StatDist['Continuous_DNN'] = continuous_NN_output[0]
    df_StatDist['Discrete_DNN'] = discrete_NN_output[0]
    df_StatDist['Mixed_DNN'] = mix_NN_output[0]
    df_StatDist['QBD'] = QBD_StatDist
    df_StatDist['Simulation'] = Simu_StatDist[:Lmax]
    df_StatDist['Whitt1993'] = StatDistWhitt1993
    mycolor = ['orange', 'green', 'brown', 'blue', 'yellow', 'red']
    if K == 1 and queue_type == 'continuous': # using opher2024 method only when K = 1
        df_StatDist['Baron2024'] = Opher2024_dist
        mycolor = ['orange', 'green', 'brown', 'blue', 'yellow', 'red', 'purple']
    # Set figure size
    plt.figure(figsize=(10, 6))
    # Plotting the bar plot
    #df_StatDist = np.round(df_StatDist,2)
    if rho < 0.4:
        df_StatDist.iloc[0:10,].plot(kind='bar', figsize=(10, 6), color =  mycolor)        
    elif rho < 0.8:
        df_StatDist.iloc[0:15,].plot(kind='bar', figsize=(10, 6), color =  mycolor)        
    else:
        df_StatDist.iloc[0:20,].plot(kind='bar', figsize=(10, 6), color =  mycolor)
    # Adding title and labels
    plt.title(f'Queue length distribution: cross comparison of DNNs, {queue_type} queue, $K$ = {K}, rho = {rho}', fontsize=16)


    plt.xlabel('Queue length', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    # Rotating x-axis labels for better visibility
    plt.xticks(rotation=45)
    # Adding legend
    plt.legend(title='Methods', fontsize=14)
    # Display the plot
    plt.tight_layout()

    # file name for saving the figure
    figure_path = f'Output/Figures/Cross_Comparison/cross_comparison_DNN_K{K}_{queue_type}_queue_{rho}.png'
    # Save the figure
    plt.savefig(figure_path, dpi=200, bbox_inches='tight')
    # Display the plot
    plt.show()
    
    return df_StatDist