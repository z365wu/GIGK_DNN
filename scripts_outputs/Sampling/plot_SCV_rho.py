#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 19:36:49 2024

@author: z365wu

Purposes:
    1. plotting the SCV of inter-arrial time and service time for training samples of NN
    2. plotting the distrbution of the traffic intensy 'rho' of training samples

"""

import pandas as pd
import matplotlib.pyplot as plt

### Plot SCV of inter-arrival times vs. SCV of service times
# df_all: samples
# queue_type: queue type (continuous or discrete)
# K: number of servers
def plot_scv_a_s(df_all, queue_type):
    SCV_a = df_all['SCV_arrial'] # SCV of inter-arrival times
    SCV_s = df_all['SCV_service'] # SCV of service times
    
    ### plot SCV_a vs SCV_s
    # Create a scatter plot
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.scatter(SCV_a, SCV_s, color='blue', alpha=0.3)  # Plot SCV_arrial vs SCV_service
    
    # Adding labels and title
    plt.title(f'SCV in Inter-arrival vs. Service Times: {queue_type}', fontsize=14)
    plt.xlabel('SCV of Inter-arrival Times', fontsize=14)
    plt.ylabel('SCV of Service Times', fontsize=14)
    
    # Show the plot
    plt.show()
    
## plotting the distrbution of rho of training samples over server numbers
# df_all: samples
# K: number of servers
# color: plotting color
def plot_rho(df_all, K, color):
    
    # Select the 'Rho' column
    rho_values = df_all['Rho']
    
    # Setup the figure and plot
    plt.figure(figsize=(10, 6))
    
    # Create histogram for 'Rho' with density=True for probability distribution
    plt.hist(rho_values, bins=1000, alpha=1, density=True, color = color)
    plt.title(r'Probability Density of $\rho$' + f': {queue_type}, {K} server(s)', fontsize=14)
    plt.xlabel(r'$\rho$', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.xlim(0, 0.2)
    
    # Show plot
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    
    # K: number of servers
    # number of df_rho_close_to_0 to concat for continuous case
    Rho_0 = [20000, 6000, 1200]
    
    # initial data
    df_all_K = pd.DataFrame()
    
    # concat data over K = 1, 2, 3
    for K in [1, 2, 3]:
        
        queue_type = 'continuous'
        # continuous case only: samples with upper bound of max(moments_Arrival[n_max-1], moments_Service[n_max-1]) < 1.0e+30:
        df_all = pd.read_csv(f'samples/df_{queue_type}_server_number_' + str(K) + '.csv')
        # continuous case only: samples with rho close to 0
        df_rho_close_to_0 = pd.read_csv(f'samples/df_{queue_type}_server_number_' + str(K) + '_rho_close_0.csv')
        
        # concat df_all and df_rho_close_to_0
        df_all = pd.concat([df_all, df_rho_close_to_0.iloc[:Rho_0[K-1],:]], axis = 0)
        
        # concat over K
        df_all_K = pd.concat([df_all_K, df_all], axis = 0)
        
        # For each K: Plot the traffic intensity 'rho' for the system from the samples loaded.
        # plot_rho(df_all, K, colors[K-1])
    
        # For each K: Plot SCV of inter-arrival times vs. SCV of service times
        # plot_scv_a_s(df_all2, queue_type, K)
    
    df_all_K = df_all_K.reset_index(drop=True)
    
    # Plot the traffic intensity 'rho' for the system from the samples loaded.
    # plot rho over server number
    fig, axs = plt.subplots(nrows=3, figsize=(10, 12))
    
    server_numbers = df_all_K['server_number'].unique()
    colors = ['blue', 'green', 'red']  # Colors for different servers
    
    for i, server in enumerate(server_numbers):
        ax = axs[i]
        data = df_all_K[df_all_K['server_number'] == server]['Rho']
        ax.hist(data, bins=300, color=colors[i], alpha=0.5, density=True)
        ax.set_title(r'Histogram of $\rho$' + f': {server} server(s), {queue_type}', fontsize=14)
        ax.set_xlabel(r'$\rho$', fontsize=14)
        ax.set_ylabel('Probability', fontsize=14)
        #ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot SCV of inter-arrival times vs. SCV of service times
    plot_scv_a_s(df_all_K, queue_type)
