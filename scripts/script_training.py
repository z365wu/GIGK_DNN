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
    df_loss_accuracy.to_csv(f'Output/Tables/csv/Validation_loss_accuracy_{DNN_net.queue_type}.csv')


def Table_DNN_training_accuracy_loss(mode):
    """
    Generate a LaTeX table summarizing DNN accuracy and loss results.
    
    Parameters
    ----------
    mode : str
        'train' to summarize training accuracy/loss across K and q_type;
        'validation' to summarize validation accuracy/loss.
    
    Output
    ------
    Writes a .txt file with LaTeX table in Output/Tables/.
    Prints the LaTeX table to console.
    """
    
    df_all = pd.DataFrame()
    if mode == 'train':
        for queue_type in ['continuous', 'discrete', 'mixed']:
            for K in [1, 2, 3]:
                df_sub = pd.read_csv(f'Output/models/training_history_GIGK({str(K)})_{queue_type}.csv')
                df_sub = df_sub.loc[[df_sub['epoch'].idxmax()]]
                df_sub['q_type'] = queue_type
                df_all = pd.concat([df_all,df_sub], axis = 0, ignore_index=True)
    elif mode == 'validation':
        for queue_type in ['continuous', 'discrete', 'mixed']:
            df_sub = pd.read_csv(f'Output/Tables/csv/Validation_loss_accuracy_{queue_type}.csv')
            df_sub['q_type'] = queue_type
            df_sub = df_sub.rename(columns={'accuracy': 'accuracy_fn'})
            df_all = pd.concat([df_all,df_sub], axis=0, ignore_index=True)

    # Pivot the DataFrame
    table = df_all.pivot(index="K", columns="q_type", values=["accuracy_fn", "loss"])
    
    # Build LaTeX manually (no booktabs)
    latex_str = "\\begin{table}[H]\n"
    latex_str += "\\centering\n"
    if mode == 'train':
        latex_str += "\\caption{Training Accuracy and Loss for $DNN_{(C)}^{(K)}$, $DNN_{(D)}^{(K)}$, and $DNN_{(M)}^{(K)}$, for $K=1, 2, 3$.}\n"
    elif mode == 'validation':
        latex_str += "\\caption{Validation Accuracy and Loss for $DNN_{(C)}^{(K)}$, $DNN_{(D)}^{(K)}$, and $DNN_{(M)}^{(K)}$, for $K=1, 2, 3$.}\n"
    
    latex_str += "\\begin{tabular}{|c|c|c|c|c|c|c|}\n"
    latex_str += "\\hline\n"
    latex_str += " & \\multicolumn{2}{c|}{$DNN_{(C)}^{(K)}$} & \\multicolumn{2}{c|}{$DNN_{(D)}^{(K)}$} & \\multicolumn{2}{c|}{$DNN_{(M)}^{(K)}$} \\\\\n"
    latex_str += "\\hline\n"
    latex_str += "$K$ & Accuracy & Loss & Accuracy & Loss & Accuracy & Loss \\\\\n"
    latex_str += "\\hline\n"
    
    # Fill table rows
    for k in table.index:
        row = [
            f"{k}",
            f"{table.loc[k, ('accuracy_fn', 'continuous')]:.5f}", f"{table.loc[k, ('loss', 'continuous')]:.5f}",
            f"{table.loc[k, ('accuracy_fn', 'discrete')]:.5f}",   f"{table.loc[k, ('loss', 'discrete')]:.5f}",
            f"{table.loc[k, ('accuracy_fn', 'mixed')]:.5f}",      f"{table.loc[k, ('loss', 'mixed')]:.5f}",
        ]
        latex_str += " & ".join(row) + " \\\\\n\\hline\n"
    
    latex_str += "\\end{tabular}\n"
    if mode == 'train':
        latex_str += "\\label{tb: Training errors and accuracy for all nine cases}\n"
    elif mode == 'validation':
        latex_str += "\\label{tb: Validation errors and accuracy for all DNNs}\n"
    
    latex_str += "\\end{table}"
    
    if mode == 'train':
        file_name = 'Output/Tables/DNN_training_accuracy_and_loss.txt'
    elif mode == 'validation':
        file_name = 'Output/Tables/DNN_validation_accuracy_and_loss.txt'
        
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(latex_str)
    
    print(latex_str)


if __name__ == '__main__':
    
    ##### Parameters
    n_max = 10           # The highest order of moments
    Lmax = 500           # The maximum queue length
    m_max = 15           # The maximum order of PH-representation (alpha, T)
    input_dim = 2*n_max  # DNN input dimension: arrival n_max + service n_max
    
    # make directories for output csv files
    os.makedirs('Output/Tables/csv/', exist_ok=True)
    
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
        
    # Generate LaTeX table for training accuracy and loss
    Table_DNN_training_accuracy_loss(mode='train')
    
    # Generate LaTeX table for validation accuracy and loss
    Table_DNN_training_accuracy_loss(mode='validation')
