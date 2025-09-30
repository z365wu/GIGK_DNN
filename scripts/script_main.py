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
# import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in matmul") # only required for Mac chips
# warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in matmul") # only required for Mac chips


### 1. Generate samples -------------------------------
def sample_generation(queue_type, n_max, Lmax, m_max, sample_size, file_save_name):
    '''
    Generate training samples
    '''
    
    #### Class initilization for sample generation #####
    NN_sample1 = PHPHK_sample_generation(queue_type = queue_type, n_max = n_max, Lmax = Lmax, m_max = m_max)
    
    # ##### Generate Samples and Save #####
    for i in range(1, 2): # number of sample files
        for K in [1, 2, 3]:  # K = number of servers: [1], [2], [3], [1, 2], [1, 3], [2, 3], or [1, 2, 3]:
            print(f'number of servers: {K}')
            NN_sample1.Sample_Generation(Sample_size = sample_size, K = K, file_name = f'df_{queue_type}_{K}_servers_' + file_save_name)


### 2. Trainging DNN -------------------------------
def training_DNN(queue_type, n_max, Lmax, m_max):
    '''
    Trainging DNN using 80% of all generated samples and save trained parameters in the folder 'Output/models/'
    '''
    #### Class initilization for DNN training #####
    DNN_net = NN_for_Queue(
        queue_type=queue_type, n_max=n_max, Lmax=Lmax, m_max=m_max, input_dim=input_dim, num_classes=Lmax
    )
    #### Construct a DNN model
    DNN_net.DNN()
    
    #### Load the saved DNN model for further training if avaiable; otherwise train a DNN model    
    for K in [1, 2, 3]:    # K = number of servers: [1], [2], [3], [1, 2], [1, 3], [2, 3], or [1, 2, 3]
        DNN_net.training(K=K)


### 3. Prediction -------------------------------
def Prediction(queue_type, n_max, Lmax, m_max, rho):
    '''
    Predict stationary distribution of queue length and generate plots
    '''
    
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
        df_StatDist_preds, rho = DNN_preds.Queue_length_preds(queue_type, K=K, NN_model = DNN_model, rho_lower=rho, rho_upper=rho)


if __name__ == '__main__':
    
    # Build DNN over queue types
    for queue_type in ['continuous', 'discrete', 0.5]: # 0.5 is for the mixed case with 0.5*100% continous samples and (1 - 0.5) * 100% discrete samples 
        print(f'queue type: {queue_type}')
        
        n_max = 10              # The highest order of moments
        Lmax = 500              # The maximum queue length
        m_max = 15              # The maximum order of PH-representation (alpha, T)
        input_dim = 2*n_max     # DNN input dimension: arrival n_max + service n_max
        
        # generate samples for continuous and discrete queues
        if queue_type in ['continuous', 'discrete']:
            sample_generation(queue_type, n_max, Lmax, m_max, sample_size=6, file_save_name='sample_test_Sept_25_2025')
        
        # train a DNN model
        training_DNN(queue_type, n_max, Lmax, m_max)
        
        # prediction based on the trained DNN models
        Prediction(queue_type, n_max, Lmax, m_max, rho=0.27) # rho: traffic intensity
