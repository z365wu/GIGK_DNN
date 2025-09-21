#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 19:33:01 2025
import Python modules for 
    1. Sample generation (run this part if you want more samples)
    2. DNN training
    3. Prediction
NOTE: You can run this file for all three parts. It is suggested to run one part at a time. 
      On the other hand, you can run 'script.sampling', 'script.training', or 'script.prediction' individually for 
      individual parts with more controls. We suggest that you run those individual files individually.
@author: z365wu and q7he
"""

from Sampling.Class_sample_generation import PHPHK_sample_generation
from Training.Class_training import NN_for_Queue
from Prediction.Class_prediction import DNN_prediction    

##### Parameters #####
# K: The number of servers
queue_type = 'discrete' # queue type: 'continuous' or 'discrete'
n_max = 10              # The highest order of moments
Lmax = 500              # The maximum queue length
m_max = 15              # The maximum order of PH-representation (alpha, T)
input_dim = 2*n_max     # DNN input dimension: arrival n_max + service n_max


# ### 1. Generate samples -------------------------------
# #### Class initilization for sample generation #####
# NN_sample1 = PHPHK_sample_generation(queue_type = queue_type, n_max = n_max, Lmax = Lmax, m_max = m_max)

# # ##### Generate Samples and Save #####
# for i in range(1, 2):
#     for K in [1, 2, 3]:  # K = number of servers: [1], [2], [3], [1, 2], [1, 3], [2, 3], or [1, 2, 3]:
#         NN_sample1.Sample_Generation(Sample_size = 2, K = K, file_name = f'df_{queue_type}_{K}_servers_sample_{i}_He')


# ### 2. Trainging DNN -------------------------------
# #### Class initilization for DNN training #####
# DNN_net = NN_for_Queue(
#     queue_type=queue_type, n_max=n_max, Lmax=Lmax, m_max=m_max, input_dim=input_dim, num_classes=Lmax
# )
# #### Creat a DNN network
# DNN_model = DNN_net.DNN()
# #### Load the saved DNN model for further training if avaiable; otherwise train a DNN model    
# for K in [1, 2, 3]:    # K = number of servers: [1], [2], [3], [1, 2], [1, 3], [2, 3], or [1, 2, 3]
#     DNN_net.training(K=K)


### 3. Prediction -------------------------------
#### Class initilization for DNN training #####
DNN_net = NN_for_Queue(
    queue_type=queue_type, n_max=n_max, Lmax=Lmax, m_max=m_max, input_dim=input_dim, num_classes=Lmax
)
#### Class initilization for DNN prediction #####
DNN_preds = DNN_prediction(queue_type=queue_type, n_max=n_max, Lmax=Lmax, m_max=m_max)
#### Predict queue length stationary distribution and plot using a newly generated sample
for K in [1, 2, 3]:   # K = number of servers: [1], [2], [3], [1, 2], [1, 3], [2, 3], or [1, 2, 3]
    DNN_model = DNN_net.DNN()
    DNN_model.load_weights(f'Output/models/model_GIGK({str(K)})_CSFP_saved_{DNN_net.queue_type}.weights.h5')
                      # [rho_lower, rho_upper]: the range of the traffic intensity of the GI/G/K queues
    df_StatDist_preds, rho = DNN_preds.Queue_length_preds(queue_type, K=K, NN_model = DNN_model, rho_lower=0.27, rho_upper=0.27)

