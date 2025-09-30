#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:38:39 2025
DNN Training:  (Plots and tables for Section 4)
        Part I. Train DNN models (Section 4.3)
        Part II. DNN validation (Section 4.4)
@author: z365wu and q7he
"""

import pandas as pd
from Training.Class_training import NN_for_Queue
import os

def DNN_training(DNN_net):
    '''
    Part I: Training the DNN (for Section 4.3)
    Plot training loss and accuracy for all K over epochs (for Section 4.3)
    '''
    
    DNN_net.DNN() # creat a DNN structure
    # Load the saved DNN model for further training if avaiable; otherwise train a new DNN model 
    for K in [1, 2, 3]:   
         DNN_net.training(K=K)

    # Plot training loss and accuracy for all K over epochs (for Section 4.3)  #######################
    # Queue-type is specified in DNN_net() defined above
    DNN_net.Plot_DNN_training_accuracy_loss(K_list=[1, 2, 3])


def DNN_validation(DNN_net):
    '''
    1. Validation Test: DNN's mean loss and accuracy over test samples (20% of out-of-traning samples)
    2. Table: Accuracy of the test set, saved in 'Output/Tables/accuracy_of_test_set_K{K}.txt'  ######
    Numerical examples and error analysis: average, min, and max 
    of Sum of absolute errors (SAE) and Percent relative error of the mean (REM)
    Parameters: 
    - K: number of servers
    - batch_size: bath size of samples for each of the 10 tests
    '''
        
    # DNN validation for (Section 4.4)
    df_loss_accuracy = pd.DataFrame(columns=['K', 'Queue_type', 'loss', 'accuracy'])
    row_ind = 0
    batch_size = 50
    for K in [1, 2, 3]:
        DNN_model = DNN_net.DNN()
        DNN_model.load_weights(f'Output/models/model_GIGK({str(K)})_CSFP_saved_{DNN_net.queue_type}.weights.h5')
        test_loss, test_accuracy, df_accuracy_test, df_accuracy_latex_table = DNN_net.Validation_test_DNN(K, DNN_model, batch_size)
        # Record results into DataFrame
        df_loss_accuracy.loc[row_ind] = [K, DNN_net.queue_type, test_loss, test_accuracy]
        row_ind += 1
        
        # Save LaTeX table to a text file
        print('DNN accuracy test:\n', df_accuracy_test)
        # print('DNN accuracy test:\n', df_accuracy_latex_table)                                                      
        file_name = f'Output/Tables/accuracy_of_test_set_K{K}_{DNN_net.queue_type}.txt'
        os.makedirs('Output/Tables/', exist_ok=True)
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(df_accuracy_latex_table)
    
    # # Save overall loss and accuracy results
    df_loss_accuracy.to_csv(f'Output/Tables/Validation_lossa_accuracy_{DNN_net.queue_type}.csv')


if __name__ == '__main__':
    
    ##### Parameters
    n_max = 10           # The highest order of moments
    Lmax = 500           # The maximum queue length
    m_max = 15           # The maximum order of PH-representation (alpha, T)
    input_dim = 2*n_max  # DNN input dimension: arrival n_max + service n_max

    # Note: queue type can be a number in the range [0,1] ; proportion of continuous samples)
    # queue_type = 0   # Discrete time = 0; mixed = 0.5; 0 <= queue_type <= 1
    for queue_type in ['continuous', 'discrete', 0.5]:  # 0.5 is for the mixed case with 0.5*100% continous samples and (1 - 0.5) * 100% discrete samples
        print(f'queue type: {queue_type}')
        
        # Class initilization
        DNN_net = NN_for_Queue(
            queue_type=queue_type, n_max=n_max, Lmax=Lmax, m_max=m_max, input_dim=input_dim, num_classes=Lmax
        )
        
        # Part I: Training the DNN (for Section 4.3)
        DNN_training(DNN_net)
        
        # Part II: DNN validation for (Section 4.4)
        DNN_validation(DNN_net)