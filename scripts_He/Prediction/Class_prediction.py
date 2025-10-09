#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:31:03 2025

1. Predicting the stationary distribution of queue length for a new sample.

@author: z365wu and q7he
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Prediction.Comparison_and_validation import Compare_StatDist

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float64')


class DNN_prediction:
    
    def __init__(self, queue_type, n_max, Lmax, m_max):
        self.queue_type = queue_type
        self.n_max = n_max
        self.Lmax = Lmax
        self.m_max = m_max


    #### Plotting predicted stationary queue length distribution for the four methods    
    # Randomly generate a sample (moments and stationary distribution of queue length) with the reqiured rho
    # Generate the DataFrame containing stationary distributions using QBD, Simulation, NN, and Whitt methods
    def Queue_length_preds(self, queue_type, K, NN_model, rho_lower, rho_upper, save_fig= False, fig_save_name = None):
        
        self.df_StatDist, rho = Compare_StatDist(
            self.queue_type, K, self.m_max, self.n_max, self.Lmax, NN_model, rho_lower, rho_upper
        )
            
        ### Maximal MSE between QBD and Simulation: plot QBD_StatDist vs. Simulation_StatDist vs. NN_StatDist
        # Plotting the bar plot
        N = int(rho*100/4)
        self.df_StatDist.iloc[0:N,].plot(kind='bar', figsize=(10, 6), rot=0)
         
        # Adding title and labels
        if queue_type == 0.5:
            plt.title(f'Queue Length Distribution: Mixed Queue, $K$={K}, $\\rho = {rho}$', fontsize=14)
        else:
            plt.title(f'Queue Length Distribution: {queue_type.capitalize()} Queue, $K$={K}, $\\rho = {rho}$', fontsize=14)
        plt.xlabel('Queue length', fontsize=14)
        plt.ylabel('Probability', fontsize=14)
        # Rotating x-axis labels for better visibility
        #plt.xticks(rotation=45)
        # Adding legend
        plt.legend(fontsize=14)
        # Adjust layout
        plt.tight_layout()
        # Save the figure
        if save_fig == True:
            # file name for saving the figure
            figure_path = f'Figures/prediction_compare_K{K}_rho_{rho}_' + fig_save_name + '.png'
            # Save the figure
            plt.savefig(figure_path, dpi=200, bbox_inches='tight')
        
        # Displaying the plot
        plt.show()
        
        return self.df_StatDist, rho
     
