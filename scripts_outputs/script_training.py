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
from Prediction.Comparison_and_validation import Compare_Queue_quantity, df_quantity_Compare_to_latex, Compare_StatDist, Compare_Queue_quantity_batch, Out_space_samples
from Sampling.Class_sample_generation import PHPHK_sample_generation  # For both continuous and discrete cases


##### Parameters #####
# K: The number of servers
queue_type = 0.5     # Continuous time = 1 (Note: queue type is a number in the range [0,1] ; proportion of continuous samples)
                     # queue_type = 0   # Discrete time = 0; mixed = 0.5; 0 <= queue_type <= 1
n_max = 10           # The highest order of moments
Lmax = 500           # The maximum queue length
m_max = 15           # The maximum order of PH-representation (alpha, T)
input_dim = 2*n_max  # DNN input dimension: arrival n_max + service n_max


############ Class initilization and DNN training for ALL functions below ###########
DNN_net = NN_for_Queue(
    queue_type=queue_type, n_max=n_max, Lmax=Lmax, m_max=m_max, input_dim=input_dim, num_classes=Lmax
)
DNN_net.DNN() # creat a DNN structure

##########  Part I: Training the DNN (for Section 4.3) ###################
##### Using generated samples ##
# ## Load the saved DNN model for further training if avaiable; otherwise train a new DNN model 
for K in [1, 2, 3]:   
     DNN_net.training(K=K)

# #### Using newly generated samples (uncomment the following code to run)
# ## (Not completed for the mixed case yet) ******
# # Class initilization for sample generation
# DNN_sample = PHPHK_sample_generation(queue_type = queue_type, n_max = n_max, Lmax = Lmax, m_max = m_max)
# for K in [1]:  # [1, 2, 3]
#     file_name = 'Give_a_file_name_you_like_test0'
#     DNN_sample.Sample_Generation(Sample_size = 10, K = K, file_name = file_name)
#     # Load NN model for each server number K
#     DNN_model = DNN_net.DNN()
#     # Load DNN weights
#     DNN_model.load_weights(f'Output/models/model_GIGK({str(K)})_CSFP_saved_{queue_type}.weights.h5')
#     # Retrain the DNN using new samples
#     DNN_net.Training_using_new_samples(DNN_sample.df_sample)
#     # Save the trained model
#     DNN_net.model.save_weights(f'Output/models/model_GIGK({str(K)})_CSFP_saved_{queue_type}.weights.h5')    

# ####### Plot training loss and accuracy for all K over epochs (for Section 4.3)  #######################
# #### Queue-type is specified in DNN_net() defined above
DNN_net.Plot_DNN_training_accuracy_loss(K_list=[1, 2, 3])


##### Part II: DNN validation for (Section 4.4) ################################################
# 1. Validation Test: DNN's mean loss and accuracy over test samples (20% of out-of-traning samples)
# 2. Table: Accuracy of the test set, saved in 'Output/Tables/accuracy_of_test_set_K{K}.txt'  ######
# Numerical examples and error analysis: average, min, and max 
# of Sum of absolute errors (SAE) and Percent relative error of the mean (REM)
# Parameters: 
# - K: number of servers
# - batch_size: bath size of samples for each of the 10 tests

# df_loss_accuracy = pd.DataFrame(columns=['K', 'Queue_type', 'loss', 'accuracy'])
# row_ind = 0
# batch_size = 50
# for K in [1, 2, 3]:
#     DNN_model = DNN_net.DNN()
#     DNN_model.load_weights(f'Output/models/model_GIGK({str(K)})_CSFP_saved_{DNN_net.queue_type}.weights.h5')
#     test_loss, test_accuracy, df_accuracy_test, df_accuracy_latex_table = DNN_net.Validation_test_DNN(K, DNN_model, batch_size)
#     # Record results into DataFrame
#     df_loss_accuracy.loc[row_ind] = [K, DNN_net.queue_type, test_loss, test_accuracy]
#     row_ind += 1
    
#     # Save LaTeX table to a text file
#     print('DNN accuracy test:\n', df_accuracy_test)
#     # print('DNN accuracy test:\n', df_accuracy_latex_table)                                                      
#     file_name = f'Output/Tables/accuracy_of_test_set_K{K}_{DNN_net.queue_type}.txt'
#     with open(file_name, "w", encoding="utf-8") as f:
#         f.write(df_accuracy_latex_table)

# # # Save overall loss and accuracy results
# df_loss_accuracy.to_csv(f'Output/Tables/Validation_lossa_accuracy_{DNN_net.queue_type}.csv')
