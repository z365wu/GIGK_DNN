#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 11:12:13 2024

# Purposes:
    1. Compare QBD and Simulation codes for continuous queue
    2. Validate of NN and QBD for continuous queue
    3. Performance/comparison NN, QBD, Simulation, and Whitt1993 for continuous queue

@author: z365wu and q7he
"""

import numpy as np
import random
import os
import pandas as pd
#from numpy.linalg import inv
import matplotlib.pyplot as plt

from Sampling.QBD_for_CT_PHPHK_CSFP import CTPH_Rep_generator, CTPHD_Moments, CTPHPHK_Stationary_Queue_Length
#from Sampling.QBD_for_DT_PHPHK_CSFP import DTPH_Rep_generator, DTPHD_Moments, main_QBD_DTPHPHK_CSFP
from Sampling.QBD_for_DT_PHPHK_CSFP import Discrete_PH_PH_K
from Prediction.Simulation_CT_PHPHK_Queue import Simulation_StatDist_CT # continuous case
from Prediction.Simulation_DT_PHPHK_Queue import Simulation_StatDist_DT # discrete case
from Prediction.Whitt_approximation_queue_length import Whitt1993
from Prediction.NN_model_Baron import Baron2024
from Prediction.DNN_GGC import GGC


def PH_Represent(queue_type, K, m_max, n_max, Lmax, rho_lower, rho_upper, max_moment_bound = 1.0e+30):
    '''
    Generate parameters of a continuous/discrete time PH/PH/K queue: K, (m_a, alpha_a, T_a), (m_s, alpha_s, T_s) 
    m_max: The maximum order of PH-representation (alpha, T)
    n_max: The highest order of moments
    K:     The number of servers
    '''

    rho = random.random() * (rho_upper - rho_lower) + rho_lower               # Generate mean interarrival time and traffic intensity within [rho_lower, rho_upper]
    print('rho:', rho)

    if queue_type == 1 or queue_type == 'continuous': # generate continuous queue; queue_type = 1 means continuous queue. 
        m_a = random.randint(1, m_max) # Phase numbers of arrival process
        m_s = random.randint(1, m_max) # Phase number of service process
        alpha_a, T_a = CTPH_Rep_generator(m_a, rho*K)      # rho<K and mean for interarrival arrival time
        moments_Arrival = CTPHD_Moments(alpha_a, T_a, n_max)
        # Service time: for i in the range(1, m_max) for the distribution of service time
        alpha_s, T_s = CTPH_Rep_generator(m_s, 1)    # rho=1 and E[S] = 1 for service times 
        # Note: the system rho is guaranteed by rho*K/(K*1) = rho 
        moments_Service = CTPHD_Moments(alpha_s, T_s, n_max) # the moments have been log tranformed
        # Queueing quantities: Stationary distribution of queue length  

    elif queue_type == 0 or queue_type == 'discrete': # generate discrete queue; queue_type = 0 means discrete queue. 
        m_a, alpha_a, T_a, m_s, alpha_s, T_s, moments_Arrival, moments_Service, _ = Discrete_PH_PH_K(1, 1, K, m_max, n_max, Lmax, rho_given = rho)

    return rho, m_a, alpha_a, T_a, m_s, alpha_s, T_s, moments_Arrival, moments_Service # return paramters for continuous PD distribution


def Compare_StatDist(queue_type, K, m_max, n_max, Lmax, NN_model, rho_lower, rho_upper, max_moment_bound = 1.0e+30):
    '''
    Stationary distributions of a continuous/discrete time PH/PH/K queue #######
    i) QBD (exact); ii) Simulation; iii) NN prediction;  #######
    iv) Whitt1993; v) Baron2024 (only for a continuous queue with 1 server) #######
    vi) Sherzer2025
    '''
    
    if queue_type == 0.5 or queue_type == 'mixed': 
        print('queue type: mixed')
        #np.random.seed(42) # set seed for reproducibility
        rand_type = np.random.random()
        if rand_type < 0.5:
            # generate a discrete sample
            queue_type = 'discrete'
            print('queue type: mixed -- the random sample is discrete')
        elif rand_type >= 0.5:
            # generate a continuous sample
            queue_type = 'continuous'  
            print('queue type: mixed -- the random sample is continuous')
    
    if queue_type == 1 or queue_type == 'continuous': # generate continuous queue; queue_type = 1 means continuous queue. 
        # generate random paramters for continuous PD distribution
        print('queue type: continuous')
        rho, m_a, alpha_a, T_a, m_s, alpha_s, T_s, moments_Arrival, moments_Service = PH_Represent(1, K, m_max, n_max, Lmax, rho_lower, rho_upper, max_moment_bound = 1.0e+30)
        # i) calculate the stationary distribution using QBD method
        QBD_StatDist = CTPHPHK_Stationary_Queue_Length(m_a, alpha_a, T_a, m_s, alpha_s, T_s, K, Lmax)
        # ii) calculate the stationary distribution using simulation method
        Simu_StatDist = Simulation_StatDist_CT(m_a, alpha_a, T_a, m_s, alpha_s, T_s, K, Lmax)
        # iii) NN model prediction
        Moments_A_S = np.concatenate((moments_Arrival, moments_Service)).reshape(1,-1)
        Moments_A_S = np.log(Moments_A_S+1) # Transform: +1 and then log transform 
        NN_output = NN_model.predict(Moments_A_S)
        # iv) Stationary distribution by Whitt1993
        m = K
        c_a_2 = moments_Arrival[1]/moments_Arrival[0]**2 - 1 # SCV of arrival time
        c_s_2 = moments_Service[1]/moments_Service[0]**2 - 1 # SCV of service time
        print('SCV:', c_a_2, c_s_2)
        StatDistWhitt1993, E_W_Whitt = Whitt1993(Lmax, m, rho, c_a_2, c_s_2) # Return stationary queue length distribution and expected waiting time 
        # v) Baron2024
        if K == 1:
            Baron2024_dist = Baron2024(moments_Arrival, moments_Service)
        # vi) Sherzer2025   
        a1 = moments_Arrival[0]
        for i in range (n_max):
            moments_Service[i] = moments_Service[i]/np.power(a1, i+1)
            moments_Arrival[i] = moments_Arrival[i]/np.power(a1, i+1)
        GGC_dist = GGC(moments_Arrival, moments_Service, K)
        
        # combine QBD_StatDist and Simulation_StatDist_CT
        df_StatDist = pd.DataFrame()
        df_StatDist['DNN'] = NN_output[0]
        df_StatDist['QBD'] = QBD_StatDist[0]
        df_StatDist['Simulation'] = Simu_StatDist[:Lmax]          
        df_StatDist['S(2025)'] = GGC_dist     
        df_StatDist['W(1993)'] = StatDistWhitt1993
        if K == 1: # using Baron2024 method only when K = 1
            df_StatDist['B(2024)'] = Baron2024_dist
            
    elif queue_type == 0 or queue_type == 'discrete': # generate discrete queue; queue_type = 0 means discrete queue. 
        # generate random paramters for discrete PD distribution
        print('queue type: discrete')
        rho = np.random.random() * (rho_upper - rho_lower) + rho_lower

        # i) calculate the stationary distribution using QBD method
        m_a, alpha_a, T_a, m_s, alpha_s, T_s, moments_Arrival, moments_Service, QBD_StatDist = Discrete_PH_PH_K(0, 1, K, m_max, n_max, Lmax, rho_given = rho)
        # ii) calculate the stationary distribution using simulation method
        Simu_StatDist = Simulation_StatDist_DT(m_a, alpha_a, T_a, m_s, alpha_s, T_s, K, Lmax)
        # iii) NN model prediction
        moments_Arrival_log = np.log(1 + moments_Arrival)
        moments_Service_log = np.log(1 + moments_Service)  
        Moments_A_S = np.concatenate((moments_Arrival_log, moments_Service_log)).reshape(1,-1)
        NN_output = NN_model.predict(Moments_A_S)
        # iv) Stationary distribution by Whitt1993
        m = K
        c_a_2 = moments_Arrival[1]/moments_Arrival[0]**2 - 1 # SCV of arrival time
        c_s_2 = moments_Service[1]/moments_Service[0]**2 - 1 # SCV of service time
        print('SCV:', c_a_2, c_s_2)
        StatDistWhitt1993, E_W_Whitt = Whitt1993(Lmax, m, rho, c_a_2, c_s_2) # Return stationary queue length distribution and expected waiting time 
        # vi) Sherzer2025   
        a1 = moments_Arrival[0]
        for i in range (n_max):
            moments_Service[i] = moments_Service[i]/np.power(a1, i+1)
            moments_Arrival[i] = moments_Arrival[i]/np.power(a1, i+1)
        GGC_dist = GGC(moments_Arrival, moments_Service, K)
        if K == 1:
            Baron2024_dist = Baron2024(moments_Arrival, moments_Service)

        # combine QBD_StatDist and Simulation_StatDist_DT
        df_StatDist = pd.DataFrame()
        df_StatDist['DNN'] = NN_output[0]
        df_StatDist['QBD'] = QBD_StatDist
        df_StatDist['Simulation'] = Simu_StatDist[:Lmax]           
        df_StatDist['S(2025)'] = GGC_dist          
        df_StatDist['W(1993)'] = StatDistWhitt1993
        if K == 1: # using Baron2024 method only when K = 1
            df_StatDist['B(2024)'] = Baron2024_dist
            
    else:
        print('Wrong value range of queue_type')
    
    return df_StatDist, rho # return StatDist and \rho


##### Compare the probability of the queue length being 0, the mean, and the SCV of
##### the predecited stationary queue length distribution using QBD, Simulation, NN, and Whitt methods
def Compare_Queue_quantity(df_StatDist, Lmax, K, rho, queue_type):
    ### compare the mean and variance of QBD, NN, Simulation, Whitt1993, and Baron2024
    # Calculate the probability of the queue length being 0 (q = 0) using the four methods
    
    q_0_NN = round(float(df_StatDist['DNN'][0]),3)
    q_0_QBD = round(float(df_StatDist['QBD'][0]),3)
    q_0_Simu = round(float(df_StatDist['Simulation'][0]),3)
    q_0_Sherzer = round(float(df_StatDist['S(2025)'][0]),3)
    q_0_Whitt = round(float(df_StatDist['W(1993)'][0]),3)

    # set up queue length list
    queue_list = np.arange(0, Lmax).reshape(-1,1)

    # Calculate expected queue length (exlcude the customers in the servers) 
    E_qw_NN = np.dot(df_StatDist['DNN'][K:], queue_list[:(Lmax-K)])[0]
    E_qw_QBD = np.dot(df_StatDist['QBD'][K:], queue_list[:(Lmax-K)])[0]
    E_qw_Simu = np.dot(df_StatDist['Simulation'][K:], queue_list[:(Lmax-K)])[0]
    E_qw_Sherzer = np.dot(df_StatDist['S(2025)'][K:], queue_list[:(Lmax-K)])[0]   
    E_qw_Whitt = np.dot(df_StatDist['W(1993)'][K:], queue_list[:(Lmax-K)])[0]

    # Calculate mean waiting time (that a customer spends in the queue before being served) 
    # by Little's law: E[queue_length] = arrival_rate * E[waiting time]
    arrival_rate = rho * 1 * K # the service rate is 1 in our setting
    #E[W] = E[q_w] * E[X_a] (equation (25)) in Feb draft, where q_w is the number of customers not in the servers
    E_wait_NN = (E_qw_NN / arrival_rate).round(3)
    E_wait_QBD = (E_qw_QBD / arrival_rate).round(3)
    E_wait_Simu = (E_qw_Simu / arrival_rate).round(3)
    E_wait_Sherzer = (E_qw_Sherzer / arrival_rate).round(3)  
    E_wait_Whitt = (E_qw_Whitt / arrival_rate).round(3)

    # Compuate MSE base on QBD for NN, simulation, and Whitt1993
    mse_QBD_Simu = np.mean((df_StatDist['QBD'] - df_StatDist['Simulation']) ** 2).round(3)
    mse_QBD_NN = np.mean((df_StatDist['QBD'] - df_StatDist['DNN']) ** 2).round(3)
    mse_QBD_Sherzer = np.mean((df_StatDist['QBD'] - df_StatDist['S(2025)']) ** 2).round(3)  
    mse_QBD_Whitt = np.mean((df_StatDist['QBD'] - df_StatDist['W(1993)']) ** 2).round(3)
    mse_QBD_QBD = np.mean((df_StatDist['QBD'] - df_StatDist['QBD']) ** 2).round(3)

    # Apply Baron2024 method only for continuous queue with K = 1
    if str(queue_type) in ['continuous', '1']:
        # q_0_Sherzer2025 = round(float(df_StatDist['S(2025)'][0]),3)
        # E_qw_Sherzer2025 = np.dot(df_StatDist['S(2025)'][K:], queue_list[:(Lmax-K)])[0]
        # E_wait_Sherzer2025 = (E_qw_Sherzer2025 / arrival_rate).round(3)
        # if origin_queue_type == 1 or origin_queue_type == 'continuous':
        #   mse_QBD_Sherzer2025 = np.mean((df_StatDist['S(2025)'] - df_StatDist['S(2025)']) ** 2).round(3)
        
        if K==1: # include both B(2024)
            q_0_Baron = round(float(df_StatDist['B(2024)'].iloc[0]), 3)
            E_qw_Baron = np.dot(df_StatDist['B(2024)'][K:], queue_list[:(Lmax-K)])[0]
            E_wait_Baron = (E_qw_Baron / arrival_rate).round(3)
            mse_QBD_Baron = np.mean((df_StatDist['QBD'] - df_StatDist['B(2024)']) ** 2).round(3)
            df_predict_compare = pd.DataFrame(columns=['Quantity', 'DNN', 'QBD', 'Simul', 'S(2025)', 'W(1993)', 'B(2024)', 'rho'])
        else: # include B(2024)
            df_predict_compare = pd.DataFrame(columns=['Quantity', 'DNN', 'QBD', 'Simul', 'S(2025)', 'W(1993)', 'rho'])
    else:
        df_predict_compare = pd.DataFrame(columns=['Quantity', 'DNN', 'QBD', 'Simul', 'S(2025)', 'W(1993)', 'rho'])
    
    ## df for saving probability of empty queue q_0, mean queue length E[qw], waiting time E[W] and MSE
    df_predict_compare['Quantity'] = ['q_0', 'E[q_w]', 'E[W]','MSE']
    df_predict_compare['DNN'] = [q_0_NN, E_qw_NN.round(3), E_wait_NN, mse_QBD_NN]
    df_predict_compare['QBD'] = [q_0_QBD, E_qw_QBD.round(3), E_wait_QBD, mse_QBD_QBD]
    df_predict_compare['Simul'] = [q_0_Simu, E_qw_Simu.round(3), E_wait_Simu, mse_QBD_Simu]
    # if str(origin_queue_type) in ['continuous', '1']: # using S(2025) method only for continuous queue 
    df_predict_compare['S(2025)'] = [q_0_Sherzer, E_qw_Sherzer.round(3), E_wait_Sherzer, mse_QBD_Sherzer]
    df_predict_compare['W(1993)'] = [q_0_Whitt, E_qw_Whitt.round(3), E_wait_Whitt, mse_QBD_Whitt]
    if  str(queue_type) in ['continuous', '1'] and K == 1: # using Baron2024 method only for continuous queue with K = 1
            df_predict_compare['B(2024)'] = [q_0_Baron, E_qw_Baron.round(3), E_wait_Baron, mse_QBD_Baron]
    
    df_predict_compare['rho'] = rho
    df_predict_compare['ServerNumber'] = K
    
    return df_predict_compare
    

def df_quantity_Compare_to_latex(df, queue_type):
    """
    Convert a DataFrame into a LaTeX table, grouped by ServerNumber (K),
    formatted with headers for different rho values and computational methods.

    Parameters:
    - df (pd.DataFrame): The DataFrame to convert.

    Print:
    - str: LaTeX-formatted table for each K.
    """
    
    # Get unique values of ServerNumber (K)
    unique_K = sorted(df["ServerNumber"].unique())
    
    # Save all tables across Ks
    latex_tables = ""
    
    for K in unique_K:
        # Filter DataFrame for current K
        df_K = df[df["ServerNumber"] == K]
        df_K = df_K[df_K["Quantity"] != "MSE"] # remove the column "MSE"
    
        # Replace specific row names with LaTeX math formatting
        df_K.loc[:, "Quantity"] = df_K["Quantity"].replace({
            "E[q_w]": "$\\mathbb{E}[q_w]$",
            "q_0": "$q_0$",
            "E[W]": "$\\mathbb{E}[W]$"
        })
    
        # Pivot table depending on queue type and K
        if str(queue_type) in ['continuous', '1']:
            if K == 1:
                df_pivot = df_K.pivot(index="Quantity", columns="rho",
                                      values=["DNN", "QBD", "Simul", "S(2025)", "W(1993)", "B(2024)"])
                value_order = ["DNN", "QBD", "Simul", "S(2025)", "W(1993)", "B(2024)"]
            else:
                df_pivot = df_K.pivot(index="Quantity", columns="rho",
                                      values=["DNN", "QBD", "Simul", "S(2025)", "W(1993)"])
                value_order = ["DNN", "QBD", "Simul", "S(2025)", "W(1993)"]
        else:
            df_pivot = df_K.pivot(index="Quantity", columns="rho",
                                  values=["DNN", "QBD", "Simul", "S(2025)", "W(1993)"])
            value_order = ["DNN", "QBD", "Simul", "S(2025)", "W(1993)"]
    
        # Enforce row order
        quantity_order = ["$q_0$", "$\\mathbb{E}[q_w]$", "$\\mathbb{E}[W]$"]
        df_pivot = df_pivot.reindex(quantity_order)
    
        # Sort columns
        # df_pivot = df_pivot.sort_index(axis=1, level=1)
        
        # Note: The order of column names may change (it may not match the order specified in 'values'),
        # but each column name still correctly corresponds to its column values.
        
        # Extract methods and rhos
        # methods = list(df_pivot.columns.levels[0])
        methods = value_order
        rhos = list(df_pivot.columns.levels[1])
    
        # Start LaTeX table
        latex_table = (
            "\\begin{table}[H]\n"
            "\\centering\n"
            "\\tiny\n"
            f"\\caption{{Queueing quantity comparison for $K$ = {K}; {queue_type.capitalize()} time $PH/PH/{K}$ Queues}}\n"
            "\\begin{tabular}{|l|" + "r" * len(methods) + "|}\n"
            "\\hline\n"
        )
        
        # Loop through each rho block
        for i, rho in enumerate(rhos):
            df_block = df_pivot.xs(rho, axis=1, level=1)
            # Reorder columns according to value_order
            df_block = df_block.reindex(columns=[c for c in value_order if c in df_block.columns])
    
            # Add rho header
            latex_table += f" & \\multicolumn{{{len(methods)}}}{{c|}}{{$\\rho = {rho}$}}  \\\\\n"
            latex_table += " & " + " & ".join(methods) + "  \\\\\n"
            latex_table += "\\hline\n\n"
    
            # Add data rows
            for q in df_block.index:
                row_values = " & ".join([f"{v:.3f}" if pd.notna(v) else "" for v in df_block.loc[q]])
                latex_table += f"{q} & {row_values}  \\\\\n"
            latex_table += "\n\\hline\n"
    
        # End LaTeX table
        latex_table += (
            "\\end{tabular}\n"
            f"\\label{{tab:Queueing_Quantity_Comparison_K_{K}, {queue_type}}}\n"
            "\\end{table}\n\n"
        )
    
        latex_tables += latex_table
    

    return latex_tables


def Compare_Queue_quantity_batch(Rho_list, batch_size, epochs, DNN_model, K, queue_type, m_max, n_max, Lmax):
    ''' 
    Compare the min/max/avg SAE (sum of absolute errors) of NN, Simulation, Whitt1993, and Baronr2024 with QBD over batchs
    Specifically, first, calculate SAE/REM for each sample; Second, take the min/mean/max across each sample' SAE/REM 
    '''
    queue_list = np.array([i for i in range(0, Lmax)]) # queue length considered

    df_all_quantity = pd.DataFrame()
    for epoch in range(1, epochs + 1):
        for rho_test in Rho_list:
            for i in range(1, batch_size + 1):
                print('For K =', K, 'and rho =', rho_test, ', the ', i, '-th example of total ', batch_size, 'examples.')
                # for the mixed case
                mixed_type = False
                if queue_type == 0.5 or queue_type == 'mixed':
                    #print('i: ', i)
                    mixed_type = True    
                    print('queue type: mixed')
                    #np.random.seed(42) # set seed for reproducibility
                    rand_type = np.random.random()
                    #print('rand_type: ',  rand_type)
                    if rand_type < 0.5:
                        # generate a discrete sample
                        queue_type = 'discrete'
                        print('queue type: mixed -- the random sample is discrete')
                    elif rand_type >= 0.5:
                        # generate a continuous sample
                        queue_type = 'continuous'  
                        print('queue type: mixed -- the random sample is continuous')
                        
                # Calculate stationary distribution of queue length
                df_StatDist, rho = Compare_StatDist(queue_type, K, m_max, n_max, Lmax, DNN_model, rho_test, rho_test)
                
                if mixed_type == True: # recover the queue type for the mixed case
                    queue_type = 'mixed'
                
                # SAE for Whitt (1993), DNN, and the PH/PH/K queue. 
                # SAE = ||W(1993) – Label||, ||DNN – Label||, or ||Simulation – Label||, where label is from the results of QBD
                # SAE for the three methods
                SAE_DNN = np.abs(df_StatDist['QBD'] - df_StatDist['DNN']).sum() # summation over queue length
                # SAE_whitt = np.abs(df_StatDist['QBD']- df_StatDist['W(1993)']).sum() # summation over queue length
                SAE_simu = np.abs( df_StatDist['QBD'] - df_StatDist['Simulation']).sum()
                SAE_Sherzer2025 = np.abs(df_StatDist['QBD'] - df_StatDist['S(2025)']).sum()
                if queue_type == 'continuous' and K == 1:
                    SAE_Baron = np.abs(df_StatDist['QBD'] - df_StatDist['B(2024)']).sum()

                # Calculate REM for each prediction
                REM_DNN = np.abs(np.dot(queue_list, (df_StatDist['QBD']) - df_StatDist['DNN'])) / np.dot(queue_list, df_StatDist['DNN'])
                # REM_whitt = np.abs(np.dot(queue_list, (df_StatDist['QBD']) - df_StatDist['W(1993)'])) / np.dot(queue_list, df_StatDist['W(1993)'])
                REM_simu = np.abs(np.dot(queue_list, (df_StatDist['QBD']) - df_StatDist['Simulation'])) / np.dot(queue_list, df_StatDist['Simulation'])
                REM_Sherzer2025 = np.abs(np.dot(queue_list, (df_StatDist['QBD']) - df_StatDist['S(2025)'])) / np.dot(queue_list, df_StatDist['S(2025)'])
                if queue_type == 'continuous' and K == 1:
                    REM_Baron = np.abs(np.dot(queue_list, (df_StatDist['QBD']) - df_StatDist['B(2024)'])) / np.dot(queue_list, df_StatDist['B(2024)'])
                
                df_sub = pd.DataFrame()
                # SAE error
                df_sub['SAE_DNN'] = [SAE_DNN]
                df_sub['SAE_simu'] = [SAE_simu]
                # df_sub['SAE_whitt'] = [SAE_whitt]
                df_sub['SAE_Sherzer'] = [SAE_Sherzer2025]          
                if queue_type == 'continuous' and K == 1:
                    df_sub['SAE_Baron'] = [SAE_Baron]
                
                # REM
                df_sub['REM_DNN'] = [REM_DNN]
                df_sub['REM_simu'] = [REM_simu]
                # df_sub['REM_whitt'] = [REM_whitt]
                df_sub['REM_Sherzer'] = [REM_Sherzer2025]
                if queue_type == 'continuous' and K == 1:
                    df_sub['REM_Baron'] = [REM_Baron]
                
                df_sub['sample_index'] = [i]
                df_sub['queue_type'] = [queue_type]
                df_sub['epoch'] = [epoch]
                df_sub['rho'] = [rho]
                df_sub['K'] = [K]
                df_all_quantity = pd.concat([df_all_quantity, df_sub], axis = 0)
                
    # Save df of queueing quantity over batch
    file_route_q_quantity = f'Output/Tables/csv/queue_quantity_comparison_batch{batch_size}_{queue_type}_K{K}.csv'
    df_all_quantity.to_csv(file_route_q_quantity, index=False)
    
    # loop over rho
    df_summary_all_rho = pd.DataFrame()
    for epoch in range(1, epochs + 1):
        df_all_q_epoch = df_all_quantity[df_all_quantity['epoch'] == epoch] # subset for each rho
        for rho in Rho_list:
            df_all_q = df_all_q_epoch[df_all_q_epoch['rho'] == rho] # subset for each rho
            
            if K == 1 and queue_type == 'continuous': # for Baron2024 and Sherzer2025
    
                # First group by 'epoch' and calculate min, mean, max
                # df_min = df_all_q.groupby(['epoch','K'])[['SAE_DNN', 'SAE_simu', 'SAE_whitt', 'SAE_Sherzer', 'SAE_Baron','REM_DNN', 'REM_simu', 'REM_whitt', 'REM_Sherzer', 'REM_Baron']].min()
                # df_mean = df_all_q.groupby(['epoch', 'K'])[['SAE_DNN', 'SAE_simu', 'SAE_whitt', 'SAE_Sherzer', 'SAE_Baron', 'REM_DNN', 'REM_simu', 'REM_whitt', 'REM_Sherzer', 'REM_Baron']].mean()
                # df_max = df_all_q.groupby(['epoch', 'K'])[['SAE_DNN', 'SAE_simu', 'SAE_whitt', 'SAE_Sherzer', 'SAE_Baron', 'REM_DNN', 'REM_simu', 'REM_whitt', 'REM_Sherzer','REM_Baron']].max()        
                df_min = df_all_q.groupby(['epoch','K'])[['SAE_DNN', 'SAE_simu', 'SAE_Sherzer', 'SAE_Baron','REM_DNN', 'REM_simu',  'REM_Sherzer', 'REM_Baron']].min()
                df_mean = df_all_q.groupby(['epoch', 'K'])[['SAE_DNN', 'SAE_simu', 'SAE_Sherzer', 'SAE_Baron', 'REM_DNN', 'REM_simu',  'REM_Sherzer', 'REM_Baron']].mean()
                df_max = df_all_q.groupby(['epoch', 'K'])[['SAE_DNN', 'SAE_simu', 'SAE_Sherzer', 'SAE_Baron', 'REM_DNN', 'REM_simu', 'REM_Sherzer','REM_Baron']].max()
                
                # Rename columns to indicate statistic
                df_min = df_min.rename(columns={
                    'SAE_DNN': 'DNN_SAE_min',
                    'SAE_simu': 'Simul_SAE_min',
                    # 'SAE_whitt': 'W(1993)_SAE_min',
                    'SAE_Sherzer': 'S(2025)_SAE_min',
                    'SAE_Baron': 'B(2024)_SAE_min',
                    'REM_DNN': 'DNN_REM_min',
                    'REM_simu': 'Simul_REM_min',
                    #'REM_whitt':'W(1993)_REM_min', 
                    'REM_Sherzer': 'S(2025)_REM_min',
                    'REM_Baron': 'B(2024)_REM_min'
                })
                
                df_mean = df_mean.rename(columns={
                    'SAE_DNN': 'DNN_SAE_avg',
                    'SAE_simu': 'Simul_SAE_avg',
                    # 'SAE_whitt': 'W(1993)_SAE_avg',
                    'SAE_Sherzer': 'S(2025)_SAE_avg',
                    'SAE_Baron': 'B(2024)_SAE_avg',
                    'REM_DNN': 'DNN_REM_avg',
                    'REM_simu': 'Simul_REM_avg',
                    # 'REM_whitt':'W(1993)_REM_avg', 
                    'REM_Sherzer': 'S(2025)_REM_avg',
                    'REM_Baron': 'B(2024)_REM_avg'
                })
                
                df_max = df_max.rename(columns={
                    'SAE_DNN': 'DNN_SAE_max',
                    'SAE_simu': 'Simul_SAE_max',
                    # 'SAE_whitt': 'W(1993)_SAE_max',
                    'SAE_Sherzer': 'S(2025)_SAE_max',
                    'SAE_Baron': 'B(2024)_SAE_max',
                     'REM_DNN': 'DNN_REM_max',
                    'REM_simu': 'Simul_REM_max',
                    # 'REM_whitt':'W(1993)_REM_max', 
                    'REM_Sherzer': 'S(2025)_REM_max',
                    'REM_Baron': 'B(2024)_REM_max'
                })
            
            elif queue_type == 'continuous': # for Sherzer2025
        
                # First group by 'epoch' and calculate min, mean, max
                # df_min = df_all_q.groupby(['epoch','K'])[['SAE_DNN', 'SAE_simu', 'SAE_whitt', 'SAE_Sherzer', 'REM_DNN', 'REM_simu', 'REM_whitt', 'REM_Sherzer']].min()
                # df_mean = df_all_q.groupby(['epoch', 'K'])[['SAE_DNN', 'SAE_simu', 'SAE_whitt', 'SAE_Sherzer', 'REM_DNN', 'REM_simu', 'REM_whitt', 'REM_Sherzer']].mean()
                # df_max = df_all_q.groupby(['epoch', 'K'])[['SAE_DNN', 'SAE_simu', 'SAE_whitt', 'SAE_Sherzer', 'REM_DNN', 'REM_simu', 'REM_whitt', 'REM_Sherzer']].max()
                
                df_min = df_all_q.groupby(['epoch','K'])[['SAE_DNN', 'SAE_simu', 'SAE_Sherzer', 'REM_DNN', 'REM_simu', 'REM_Sherzer']].min()
                df_mean = df_all_q.groupby(['epoch', 'K'])[['SAE_DNN', 'SAE_simu', 'SAE_Sherzer', 'REM_DNN', 'REM_simu', 'REM_Sherzer']].mean()
                df_max = df_all_q.groupby(['epoch', 'K'])[['SAE_DNN', 'SAE_simu', 'SAE_Sherzer', 'REM_DNN', 'REM_simu', 'REM_Sherzer']].max()
        
                # Rename columns to indicate statistic
                df_min = df_min.rename(columns={
                    'SAE_DNN': 'DNN_SAE_min',
                    'SAE_simu': 'Simul_SAE_min',
                    # 'SAE_whitt': 'W(1993)_SAE_min',
                    'SAE_Sherzer': 'S(2025)_SAE_min',
                    'REM_DNN': 'DNN_REM_min',
                    'REM_simu': 'Simul_REM_min',
                    # 'REM_whitt':'W(1993)_REM_min', 
                    'REM_Sherzer': 'S(2025)_REM_min'
                })
                
                df_mean = df_mean.rename(columns={
                    'SAE_DNN': 'DNN_SAE_avg',
                    'SAE_simu': 'Simul_SAE_avg',
                    # 'SAE_whitt': 'W(1993)_SAE_avg',
                    'SAE_Sherzer': 'S(2025)_SAE_avg',
                    'REM_DNN': 'DNN_REM_avg',
                    'REM_simu': 'Simul_REM_avg',
                    # 'REM_whitt':'W(1993)_REM_avg', 
                    'REM_Sherzer': 'S(2025)_REM_avg'
                })
                
                df_max = df_max.rename(columns={
                    'SAE_DNN': 'DNN_SAE_max',
                    'SAE_simu': 'Simul_SAE_max',
                    # 'SAE_whitt': 'W(1993)_SAE_max',
                    'SAE_Sherzer': 'S(2025)_SAE_max',
                     'REM_DNN': 'DNN_REM_max',
                    'REM_simu': 'Simul_REM_max',
                    # 'REM_whitt':'W(1993)_REM_max', 
                    'REM_Sherzer': 'S(2025)_REM_max'
                })
            
            else: # exclude Baron2024
                # df_min = df_all_q.groupby(['epoch','K'])[['SAE_DNN', 'SAE_simu', 'SAE_whitt', 'REM_DNN', 'REM_simu', 'REM_whitt']].min()
                # df_mean = df_all_q.groupby(['epoch', 'K'])[['SAE_DNN', 'SAE_simu', 'SAE_whitt', 'REM_DNN', 'REM_simu', 'REM_whitt']].mean()
                # df_max = df_all_q.groupby(['epoch', 'K'])[['SAE_DNN', 'SAE_simu', 'SAE_whitt', 'REM_DNN', 'REM_simu', 'REM_whitt']].max()
                df_min = df_all_q.groupby(['epoch','K'])[['SAE_DNN', 'SAE_simu',  'SAE_Sherzer', 'REM_DNN', 'REM_simu', 'REM_Sherzer']].min()
                df_mean = df_all_q.groupby(['epoch', 'K'])[['SAE_DNN', 'SAE_simu', 'SAE_Sherzer', 'REM_DNN', 'REM_simu', 'REM_Sherzer']].mean()
                df_max = df_all_q.groupby(['epoch', 'K'])[['SAE_DNN', 'SAE_simu',  'SAE_Sherzer', 'REM_DNN', 'REM_simu', 'REM_Sherzer']].max()
       
                # Rename columns to indicate statistic
                df_min = df_min.rename(columns={
                    'SAE_DNN': 'DNN_SAE_min',
                    'SAE_simu': 'Simul_SAE_min',
                    'SAE_Sherzer': 'S(2025)_SAE_min',
                    # 'SAE_whitt': 'W(1993)_SAE_min',      
                    'REM_DNN': 'DNN_REM_min',
                    'REM_simu': 'Simul_REM_min',
                    'REM_Sherzer': 'S(2025)_REM_min',
                    # 'REM_whitt':'W(1993)_REM_min'
                })
                df_mean = df_mean.rename(columns={
                    'SAE_DNN': 'DNN_SAE_avg',
                    'SAE_simu': 'Simul_SAE_avg',
                    'SAE_Sherzer': 'S(2025)_SAE_avg',                   
                    # 'SAE_whitt': 'W(1993)_SAE_avg',
                    'REM_DNN': 'DNN_REM_avg',
                    'REM_simu': 'Simul_REM_avg',
                    'REM_Sherzer': 'S(2025)_REM_avg',
                    # 'REM_whitt':'W(1993)_REM_avg' 
                })
                
                df_max = df_max.rename(columns={
                    'SAE_DNN': 'DNN_SAE_max',
                    'SAE_simu': 'Simul_SAE_max',
                    'SAE_Sherzer': 'S(2025)_SAE_max',        
                    # 'SAE_whitt': 'W(1993)_SAE_max',
                    'REM_DNN': 'DNN_REM_max',
                    'REM_simu': 'Simul_REM_max',
                    'REM_Sherzer': 'S(2025)_REM_max',
                    # 'REM_whitt':'W(1993)_REM_max'
                })
            
            # Combine into one DataFrame (along columns)
            df_summary = pd.concat([df_min, df_mean, df_max], axis=1)
            df_summary = df_summary.round(3)
            df_summary.reset_index(inplace=True)
            # Assign rho
            df_summary.insert(0, 'rho', rho)
            # Concat data
            df_summary_all_rho =  pd.concat([df_summary_all_rho, df_summary], axis=0)
    
    # Drop all columns that contain 'REM' in their name
    df_summary_all_rho = df_summary_all_rho.loc[:, ~df_summary_all_rho.columns.str.contains('REM', case=False)]
    print(f'df_summary_all_rho columns: {df_summary_all_rho.columns}')  
    
    # Save df_summary of queueing quantity over batch
    file_route_q_quantity = f'Output/Tables/csv/summary_queue_quantity_comparison_batch{batch_size}_{queue_type}_K{K}.csv'
    df_summary_all_rho.to_csv(file_route_q_quantity, index=False)
    
    # covnert df_summary_all_rho into Latex Table
    tab_latex = df_to_latex_queue_quantity_batch_table(df_summary_all_rho, K, queue_type, model_name="SAE")
    
    return tab_latex


def df_to_latex_queue_quantity_batch_table(df_k, K, queue_type, model_name="SAE",
                            caption_prefix="Queueing Quantity Comparison (by Batch)"):
    """
    Convert a summary queue quantity (by batch) df (like df_summary_all_rho) into a LaTeX table block.

    Handles both 3-column (DNN, Simul, S(2025)) and 4-column (DNN, Simul, S(2025), B(2024)) formats
    based on K and queue_type.
    """
    
    # Determine subcolumns
    if K == 1 and queue_type == "continuous":
        subcols = ["DNN", "Simul", "S(2025)", "B(2024)"]
    else:
        subcols = ["DNN", "Simul", "S(2025)"]

    n_sub = len(subcols)
    groups = ["min", "avg", "max"]
    
    if queue_type == 'continuous' or queue_type == 1:
        type_DNN = 'C'
    elif queue_type == 'discrete' or queue_type == 0:
        type_DNN = 'D'
    else:
        type_DNN = 'M'

    # ----- Header -----
    header = [
        r"\begin{table}[H]",
        r"\centering",
        r"\tiny",
        rf"\caption{{{caption_prefix} for $K = {K}$; {queue_type} queues; $DNN_{({type_DNN})}^{{({K})}}$}}",
        r"\begin{tabular}{|c|" + "|".join(["c" * n_sub] * len(groups)).replace("c", "c") + "|}",
        r"\hline",
    ]

    # First row: grouped columns
    group_row = "     & " + " & ".join([
        rf"\multicolumn{{{n_sub}}}{{c|}}{{{model_name}$\_${g}}}" if g != groups[-1]
        else rf"\multicolumn{{{n_sub}}}{{c}}{{{model_name}$\_${g}}}"
        for g in groups
    ]) + r" \\ \hline"
    header.append(group_row)

    # Second row: subcolumns repeated
    header.append("    $\\rho$ & " + " & ".join(subcols * len(groups)) + r" \\ \hline")

    # ----- Body -----
    body = []
    for _, row in df_k.iterrows():
        rho = row["rho"]
        row_str = [f"{rho:.2f}"]
        for g in groups:
            for s in subcols:
                val = row.get(f"{s}_{model_name}_{g}", 0)
                row_str.append(f"{val:.3f}")
        body.append("\t" + "   &   ".join(row_str) + r"   \\ \hline")

    # ----- Footer -----
    footer = [
        r"\end{tabular}",
        rf"\label{{tab:Queueing_Quantity_batch_Comparison_K_{K}, {queue_type}}}",
        r"\end{table}"
    ]

    # Combine all parts
    return "\n".join(header + body + footer)


#### Generate samples that are not in the (SCV_a, SCV_s) space of training samples 
#### Prediction using QBD, NN, Simu for the samples
def Out_space_samples(queue_type, K, m_max, n_max, Lmax, DNN_model, c_a_2_ubound, c_s_2_ubound, out_sample = True, rho_lower=0, rho_upper=1, max_moment_bound = 1.0e+30):
    """
    Generate in-sample or out-of-sample queueing system scenarios and predict queue length distribution 
    using QBD, NN, and Simulation methods (and Baron2024 for K=1 and continuous queue).
    
    Parameters:
    ----------
    out_sample: if True, generate samples with SCV of arrival times > c_a_2_ubound and SCV of service times > c_s_2_ubound
        if False, generate in-sample with SCV of arrival times < c_a_2_ubound and SCV of service times < c_s_2_ubound
    
    c_a_2_ubound : float
        Upper bound for the squared coefficient of variation (SCV) of arrival times.
        for generate 
    c_s_2_ubound : float
        Upper bound for the squared coefficient of variation (SCV) of service times.
    out_sample : bool, default=True
        If True, generates samples where SCV of arrival times (`c_a_2`) **exceeds** `c_a_2_ubound`
        and SCV of service times (`c_s_2`) **exceeds** `c_s_2_ubound`.
        If False, generates in-sample cases where SCV of arrival times (`c_a_2`) **is below** `c_a_2_ubound`
        and SCV of service times (`c_s_2`) **is below** `c_s_2_ubound`.
    Returns:
    -------
    rho : float
        Traffic intensity (utilization factor).
    c_a_2 : float
        Squared coefficient of variation (SCV) of arrival times.
    c_s_2 : float
        Squared coefficient of variation (SCV) of service times.
    """
      
    # for mixed DNN
    mixed_type = False
    if queue_type == 0.5 or queue_type == 'mixed': 
        print('queue type: mixed')
        mixed_type = True
        #np.random.seed(42) # set seed for reproducibility
        rand_type = np.random.random()
        if rand_type < 0.5:
            # generate a discrete sample
            queue_type = 'discrete'
            print('queue type: mixed -- the random sample is discrete')
        elif rand_type >= 0.5:
            # generate a continuous sample
            queue_type = 'continuous'  
            print('queue type: mixed -- the random sample is continuous')
    
    # for continuous queue
    if queue_type == 'continuous':
        # Generate in/out-of-sample-space c_a_2 and c_s_2 for continuous PD distribution
        if out_sample == True: # Generate out-of-sample-space c_a_2 and c_s_2 for continuous PD distribution
            c_a_2 = 0 # Initialization of the SCV for arrival time
            c_s_2 = 0 # Initialization of the SCV for service time
            while c_a_2 < c_a_2_ubound or c_s_2 < c_s_2_ubound: # out of smaple c_a_2 < 30 or c_s_2 < 20
                rho, m_a, alpha_a, T_a, m_s, alpha_s, T_s, moments_Arrival, moments_Service = PH_Represent(queue_type, K, m_max, n_max, Lmax, rho_lower, rho_upper)
                c_a_2 = moments_Arrival[1]/moments_Arrival[0]**2 - 1 # SCV of arrival time
                c_s_2 = moments_Service[1]/moments_Service[0]**2 - 1 # SCV of service time
                
        else: # Generate in-of-sample-space c_a_2 and c_s_2 for continuous PD distribution
            condition1 = True
            while condition1 == True:
                rho, m_a, alpha_a, T_a, m_s, alpha_s, T_s, moments_Arrival, moments_Service = PH_Represent(queue_type, K, m_max, n_max, Lmax, rho_lower, rho_upper)
                c_a_2 = moments_Arrival[1]/moments_Arrival[0]**2 - 1 # SCV of arrival time
                c_s_2 = moments_Service[1]/moments_Service[0]**2 - 1 # SCV of service time
                if c_a_2 < c_a_2_ubound and c_s_2 < c_s_2_ubound: # out of smaple c_a_2 < 30 or c_s_2 < 20
                    condition1 = False
        # i) calculate the stationary distribution using QBD method
        QBD_StatDist = CTPHPHK_Stationary_Queue_Length(m_a, alpha_a, T_a, m_s, alpha_s, T_s, K, Lmax)
        QBD_StatDist = QBD_StatDist[0]
        # ii) calculate the stationary distribution using simulation method
        Simu_StatDist = Simulation_StatDist_CT(m_a, alpha_a, T_a, m_s, alpha_s, T_s, K, Lmax)
    
    # for discrete queue
    elif queue_type == 'discrete':
        # Generate in/out-of-sample-space c_a_2 and c_s_2 for continuous PD distribution
        rho = random.random() * (rho_upper - rho_lower) + rho_lower   
        if out_sample == True: # Generate out-of-sample-space c_a_2 and c_s_2 for continuous PD distribution
            c_a_2 = 0 # Initialization of the SCV for arrival time
            c_s_2 = 0 # Initialization of the SCV for service time
            while c_a_2 < c_a_2_ubound or c_s_2 < c_s_2_ubound: # out of smaple c_a_2 < 30 or c_s_2 < 20
                m_a, alpha_a, T_a, m_s, alpha_s, T_s, moments_Arrival, moments_Service, QBD_StatDist = Discrete_PH_PH_K(1, 1, K, m_max, n_max, Lmax, rho_given = rho)
                c_a_2 = moments_Arrival[1]/moments_Arrival[0]**2 - 1 # SCV of arrival time
                c_s_2 = moments_Service[1]/moments_Service[0]**2 - 1 # SCV of service time

        else: # Generate in-of-sample-space c_a_2 and c_s_2 for continuous PD distribution
            condition1 = True
            while condition1 == True:
                m_a, alpha_a, T_a, m_s, alpha_s, T_s, moments_Arrival, moments_Service, QBD_StatDist = Discrete_PH_PH_K(1, 1, K, m_max, n_max, Lmax, rho_given = rho)
                c_a_2 = moments_Arrival[1]/moments_Arrival[0]**2 - 1 # SCV of arrival time
                c_s_2 = moments_Service[1]/moments_Service[0]**2 - 1 # SCV of service time
                if c_a_2 < c_a_2_ubound and c_s_2 < c_s_2_ubound: # out of smaple c_a_2 < 30 or c_s_2 < 20
                    condition1 = False
        print('rho', rho)
        # ii) calculate the stationary distribution using simulation method
        Simu_StatDist = Simulation_StatDist_DT(m_a, alpha_a, T_a, m_s, alpha_s, T_s, K, Lmax)
        
    # iii) NN model prediction
    Moments_A_S = np.concatenate((moments_Arrival, moments_Service)).reshape(1,-1)
    Moments_A_S = np.log(Moments_A_S+1) # Transform: +1 and then log transform 
    NN_output = DNN_model.predict(Moments_A_S)
    
    # iv) Stationary distribution by Whitt1993
    # m = K
    #print('SCV:', c_a_2, c_s_2)
    # StatDistWhitt1993, E_W_Whitt = Whitt1993(Lmax, m, rho, c_a_2, c_s_2) # Return stationary queue length distribution and expected waiting time 
    
    # v) Baron2024
    if queue_type == 'continuous' and K == 1:
        Baron2024_dist = Baron2024(moments_Arrival, moments_Service)
    a1 = moments_Arrival[0]
    for i in range (n_max):
        moments_Service[i] = moments_Service[i]/np.power(a1, i+1)
        moments_Arrival[i] = moments_Arrival[i]/np.power(a1, i+1)
    Sherzer2025_dist = GGC(moments_Arrival, moments_Service, K)

    # combine QBD_StatDist and Simulation_StatDist_CT
    df_StatDist = pd.DataFrame()
    df_StatDist['DNN'] = NN_output[0]
    df_StatDist['QBD'] = QBD_StatDist
    df_StatDist['Simulation'] = Simu_StatDist[:Lmax]
    # df_StatDist['W(1993)'] = StatDistWhitt1993
    df_StatDist['S(2025)'] = Sherzer2025_dist
    mycolor = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
    if queue_type == 'continuous' and K == 1: # using Baron2024 method only when K = 1
        df_StatDist['B(2024)'] = Baron2024_dist
        mycolor = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#8c564b']
        
    ### Maximal MSE between QBD and Simulation: plot QBD_StatDist vs. Simulation_StatDist_CT vs. NN_StatDist
    # Set figure size
    plt.figure(figsize=(10, 6))
    # Plotting the bar plot
    #df_StatDist = np.round(df_StatDist,2)
    N = int(rho*100/4)
    if rho > 0.85:
        N = 35
    df_StatDist.iloc[0:N,].plot(kind='bar', figsize=(10, 6), color = mycolor, rot=0)
    # Adding title and labels
    if mixed_type:
        plt.title(f'Queue length distribution: $K$={K}, {queue_type.capitalize()} queue, mixed DNN, SCVa={round(c_a_2,1)}, SCVs={round(c_s_2,1)}, $\\rho={round(rho,2)}$', fontsize=16) # $\rho$={round(rho,2)}', fontsize=16)
    else:
        plt.title(f'Queue length distribution: $K$={K}, {queue_type.capitalize()} queue, {queue_type.capitalize()} DNN, SCVa={round(c_a_2,1)}, SCVs={round(c_s_2,1)}, $\\rho={round(rho,2)}$', fontsize=16)  #$\rho$={round(rho,2)}', fontsize=16)

    plt.xlabel('Queue length', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    # Rotating x-axis labels for better visibility
    #plt.xticks(rotation=45)
    # Adding legend
    plt.legend(fontsize=14) # title='Methods'
    # Display the plot
    plt.tight_layout()
    # Define sample name based on `out_sample`
    if out_sample == True:
        sample_name = 'example_out_sample_space'
    else:
        sample_name = 'example_in_sample_space'
    # file name for saving the figure
    if mixed_type:
        figure_path = f'Output/Figures/Out_of_space_sample/mixed_queue_K{K}_SCVa_{round(c_a_2,1)}_SCVs_{round(c_s_2,1)}_rho_{round(rho,2)}_{sample_name}.png'
    else:
        #figure_path = f'Output/Figures/Out_of_space_sample/{queue_type}_queue_K{K}_SCVa_{round(c_a_2,1)}_SCVs_{round(c_s_2,1)}_rho_{round(rho,2)}_{sample_name}.png'
        figure_path = f'Output/Figures/Out_of_space_sample/{queue_type}_queue_K{K}_rho_{round(rho,2)}_{sample_name}.png'
    # Save the figure
    os.makedirs('Output/Figures/Out_of_space_sample/', exist_ok=True)    
    plt.savefig(figure_path, dpi=200, bbox_inches='tight')
    # Display the plot
    plt.show()

    return rho, c_a_2, c_s_2, df_StatDist


def error_average_best_worst(NN_model, Moments_test, Stationary_test, Lmax, batch_size):
    '''
    Numerical examples and error analysis: average, min, and max 
    of Sum of absolute errors (SAE) and relative error of the mean (REM)
    '''
    
    # Load the trained NN model for the queue
    queue_list = np.array([i for i in range(0, Lmax)]).reshape(-1,1) # queue length considered
    
    df_error = pd.DataFrame(columns=['SAE_avg', 'SAE_min', 'SAE_max', 'REM_avg', 'REM_min', 'REM_max'])
    
    # loop over batch size samples
    for i in range(0, 10):
        index_f = i * batch_size
        index_e = (i + 1) * batch_size
        Moments_bacth, Stationary_bacth = Moments_test[index_f:index_e, :], Stationary_test[index_f:index_e, :]

        # Prediction using NN model
        nn_prediction = NN_model.predict(Moments_bacth)
            
        # Sum of absolute errors (SAE) for each sample
        abs_error = np.sum(np.abs(nn_prediction - Stationary_bacth), axis=1).reshape(-1,1) # summation over queue length
        
        # Compute average, min, and max SAE
        SAE_avg = abs_error.mean()
        SAE_min = abs_error.min()
        SAE_max = abs_error.max() 
        
        # relative error of the mean (REM)
        rem_error = np.abs(np.matmul((Stationary_bacth - nn_prediction), queue_list)) / np.dot(nn_prediction, queue_list)
        
        # Compute average, min, and max REM
        REM_avg = rem_error.mean()
        REM_min = rem_error.min()
        REM_max = rem_error.max() 
        
        # Save SAE for each batch
        df_error.loc[i] = [SAE_avg, SAE_min, SAE_max, REM_avg, REM_min, REM_max]
    
    return df_error


def df_error_to_latex_table(df, queue_type, K):
    """
    Convert a DataFrame into a LaTeX table with a custom table name.

    Parameters:
    - df (pd.DataFrame): The DataFrame to convert.
    - queue_type (float): The queue type to include in the table caption.
    - K (int): The number of servers to include in the table caption.

    Returns:
    - str: LaTeX-formatted table.
    """
    
    # Add index column with numbers from 1 to df.shape[0]
    df.insert(0, 'Number', range(1, df.shape[0] + 1))

    # Convert DataFrame to LaTeX table format
    latex_table = df.to_latex(index=False, float_format="%.6f", column_format="lrrrrrr", escape=False)
    #latex_table = df.to_latex(index=False, format="e", column_format="lrrrrrr", escape=False)

    # Remove unwanted LaTeX commands (\toprule, \midrule, \bottomrule)
    latex_table = latex_table.replace("\\toprule", "\\hline").replace("\\midrule", "").replace("\\bottomrule", "")

    # Define table caption and label
    table_caption = f"Error of test set: {queue_type.capitalize()}, $K$ = {K}"
    table_label = f"tab:error_metrics_{queue_type.replace(' ', '_')}_K_{K}"

    # Format the complete LaTeX table
    latex_code = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\tiny\n"
        f"\\caption{{{table_caption}}}\n"
        "\\begin{tabular}{|l|rrrrrr|}\n"
        "\\hline\n"
        "Number & SAE\\_avg & SAE\\_min & SAE\\_max & REM\\_avg & REM\\_min & REM\\_max \\\\\n"
        "\\hline\n"
        + "\n".join(latex_table.splitlines()[3:-1]) +  # Extract only the data part of the table
        "\n\\hline\n"
        "\\end{tabular}\n"
        f"\\label{{{table_label}}}\n"
        "\\end{table}"
    )
    
    return latex_code


