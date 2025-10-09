# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:37:48 2024
      i)  Save samples to a file 
      ii) Read samples from a file
@author: z365wu and q7he
"""
import numpy as np
import pandas as pd
from pathlib import Path

########### Save samples to a csv file (new or existing) ###################################################
def Save_Samples_to_File(n_max, Lmax, K, Moments_train, Stationary_train, SCVs, Rhos, queue_time_type, R_Iter_num):
    # Reshape queue_time_type as a column matrix
    queue_time_type = np.array(queue_time_type).reshape(-1, 1)
    # Combine multiple sample datasets horizontally into a single matrix
    samples_matrix = np.hstack((Moments_train, Stationary_train, SCVs, Rhos, queue_time_type, R_Iter_num.astype(int)))
    ## Create column names for saving samples
    # Arrival moments
    col_names = []
    for i in range(0+1, n_max+1):
        col_names.append(str(i)+'th_moments_arrival')
    # serving moments
    for i in range(0+1, n_max+1):
        col_names.append(str(i)+'th_moments_service')
    # stationary queue length
    for i in range(0, Lmax):
        col_names.append('stationary_queue_length_'+str(i))  
    # SCV for the arrival time
    col_names.append('SCV_arrial')  
    # SCV for the service time
    col_names.append('SCV_service')
    # Rhos
    col_names.append('Rho')
    # queue_time_type
    col_names.append('queue_time_type')   
    # R iteration number
    col_names.append('R_Iter_num') 
    # Create dataframe for the samples
    df_samples = pd.DataFrame(data=samples_matrix, columns=col_names)
    # add a column named 'server_number'
    df_samples['server_number'] = K
    
    '''
    ## create fold 'samples' to save samples if the fold does not exist
    Path('samples').mkdir(parents=True, exist_ok=True)
    
    try:
        # Attempt to read the CSV file if it exists.
        df_samples_old = pd.read_csv('samples/df_continuous_' + 'server_number_' + str(K) + '.csv')
        # Combine the DataFrame and the new DataFrame row-wise ignoring the index
        df_samples_all = pd.concat([df_samples_old, df_samples], axis=0, ignore_index=True)
        # Save DataFrame to CSV
        df_samples_all.to_csv('samples/df_continuous_' + 'server_number_' + str(K) + '.csv', index=False)
    except FileNotFoundError:
        # This block will execute if the CSV does not exist
        # Save DataFrame to CSV
        df_samples.to_csv('samples/df_continuous_' + 'server_number_' + str(K) + '.csv', index=False)
    '''
    
    return df_samples


################# Load samples from a csv file and use them to train an existing NN model  #########################
# Reload saved CSV samples for training the neural network model.
# Load data over a range of K values (K = 1, 2) for testing purposes.
def Load_Samples_from_file(n_max, Lmax, df_all):
    moment_idx_first = df_all.columns.get_loc('1th_moments_arrival')
    # Get the column index of the last "moments service" column.
    moment_idx_last = df_all.columns.get_loc(str(n_max) + 'th_moments_service')
    # Get the column index of the first stationary column.
    stationary_idx_first = df_all.columns.get_loc('stationary_queue_length_0')
    # Get the column index of the last stationary column.
    stationary_idx_last = df_all.columns.get_loc('stationary_queue_length_' + str(Lmax - 1))
    # Obtain Moments_train as a NumPy array.
    Moments_train_new = df_all.iloc[:, moment_idx_first:(moment_idx_last + 1)].to_numpy()
    # Obtain Stationary_train as a NumPy array.
    Stationary_train_new = df_all.iloc[:, stationary_idx_first:(stationary_idx_last + 1)].to_numpy()
    
    # obtain queue type 
    # idx_queue_type = df_all.columns.get_loc('queue_time_type') 
    # queue_time_type = df_all.iloc[:, idx_queue_type:(idx_queue_type+1)].to_numpy()
    
    # Get the rho index
    rho_idx = df_all.columns.get_loc('Rho')
    rho_train_new = df_all.iloc[:, rho_idx].to_numpy()
    
    # Get inter-arrival SCV
    SCV_arrival_idx = df_all.columns.get_loc('SCV_arrial')
    SCV_arrial = df_all.iloc[:, SCV_arrival_idx].to_numpy()
    
    # Get service time SCV
    SCV_service_idx = df_all.columns.get_loc('SCV_service')
    SCV_service = df_all.iloc[:, SCV_service_idx].to_numpy()   
    
    return Moments_train_new, Stationary_train_new, rho_train_new  #, queue_time_type, SCV_arrial, SCV_service
