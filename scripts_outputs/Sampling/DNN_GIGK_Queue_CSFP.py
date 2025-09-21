# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 19:23:26 2024
    This program builds a DNN model for the GI/G/K queue.
    Input:    The first 15 (n_max) moments of the interarrival and service times; K: number of servers
    Output:   The stationary distribution of the queue length
    Approach: The model is trained by using PH/PH/K queue with its matrix-geometric solutions 
              of the continuous/discrete time QBD process of its queue length. The CSFP method is used. 
@author: z365wu and q7he
"""

import matplotlib.pyplot as plt
import time
from keras.models import Sequential
from keras.layers import Dense

from GIGK_Input_output_CSFP import Input_Output_Moments_Generator
from Save_Combine_Read_for_CSV_files import Save_Samples_to_File, Load_Samples_from_file


# Train or retrain a DNN model for a queue with K server(s)
# The input 'queue_type' can be 'continuous' or 'discrete', representing a continuous or discrete queue, respectively.
# Assume that the DNN has 30 inputs (15 log-transfermed moments of (random) arrival and service times)
# and 100 outputs (the stationary distribution of queue lengths from 0 to 99)
# The first moment of service time of continuous time is normalzied to 1 
# K: the number of servers
def DNN(queue_type, K, n_max, input_dim, Lmax, num_classes, m_max):
    # n_max = 15             # The maximum order of moments
    # input_dim = 2*n_max    # DNN input dimension: arrival n_max + service n_max
    # Lmax = 100             # The maximum queue length
    # num_classes = Lmax     # DNN output dimension
    # m_max = 10        # The maximum order size of PH-representations
    
    # Structure of the DNN model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(70, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(350, activation='relu'))
    model.add(Dense(350, activation='relu'))
    model.add(Dense(350, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))   

    # Coding model, selection, and loss function 
    model.compile(optimizer='adam', loss = 'mean_squared_error', metrics=['accuracy']) 
    # Load the saved DNN model for further training if avaiable; otherwise train a DNN model
    try:
        model.load_weights(f'models/model_GIGK({str(K)})_CSFP_saved_{queue_type}.weights.h5')
    except:
        None
        
    # Generate new samples or read saved samples to train the NN model
    # If Generate_samples = 1, generate new samples to train the NN model.
    # If Generate_samples ≠ 1 (e.g., Generate_samples = 2), read saved samples from a csv file to train the NN model
    Generate_samples = 2
    if Generate_samples == 1: # Generate new samples
        Sample_size = 50000   # The sample size for training
        # Assume there are Sample_size training samples; i) Generate PHDs; ii) Compute moments; iii) Compute stationary distribution of the QBD
        Moments_train, Stationary_train, SCVs, Rhos, queue_time_type = Input_Output_Moments_Generator(Sample_size, K, m_max, n_max, Lmax)
        # Plot SCVs: To check the versatility of the PH/PH/K queues used in training
        plt.scatter(SCVs[:,0], SCVs[:,1])   # Plot the SCVs: should be all over the place
        plt.scatter(Rhos, Rhos)             # Should be spread out on the diagonal line (0, 1)
        ## Save samples to csv file: samples_all/df_continuous_server_number_{K}.csv
        Save_Samples_to_File(n_max, Lmax, K, Sample_size, Moments_train, Stationary_train, SCVs, Rhos, queue_time_type)
    else:  # Read samples from file: samples_all/df_continuous_server_number_{K}.csv 
        Moments_all, Stationary_all, queue_time_type_all, rho_all, _, _ = Load_Samples_from_file(n_max, Lmax, K, queue_type)
    
        # Use the first 80 percentage of the saved samples as training samples
        sample_num = int(Moments_all.shape[0] * 0.8)
        Moments_train, Stationary_train = Moments_all[:sample_num,:], Stationary_all[:sample_num,:]
    
    st = time.time()    # Record the total computing time
    
    # Train or retrain the DNN model
    model.fit(Moments_train, Stationary_train, batch_size= 512, epochs= 100)  
    # Save the trained model
    model.save_weights(f'models/model_GIGK({str(K)})_CSFP_saved_{queue_type}.weights.h5')
    print('A DNN model for GI/G/K model with K =', K, 'has been trained and saved')
    
    # Print the training time
    et = time.time()
    Elapsed_time = et - st
    print('Execution time is ', Elapsed_time, 'seconds.')
    
    ####### Steps to train the NN model and to use the NN model ###### 
    #  1) Run the file "DNN_GIGK_Queue_CSFP.py" without the line "model .load_weights()"; 
    #        1.0) You need to choose K = 1, 2, 3, ... (number of servers)
    #        1.1) Choose sample size to be 5000 or more; batch_size = 1000 or more; epochs = 100 or more;
    #        1.2) This file calls "Input_Output_Moments_Generator()" from GIGK_Input_output_CSFP.py to generate samples;
    #        1.3) For continuous time sample, the above file calls Construction_QBD_for_GIGK_CSFP.py
    #        1.4) The "Input_Output_Moments_Generator()" calls "Discrete_PH_PH_K()" from QBD_for_DT_PHPHK_CSFP.py to generae discrete samples
    #        1.5) Save weights to the file "model_GIGK(K)_CSFP_saved.weights.h5"
    #        1.6) Generated samples are saved in a csv file. 
    #  2) Run the file "DNN_GIGK_Queue_CSFP.py" with the line "model.load_weights()" to further train the NN model
    #        Do this many times so that the total samples is 100,000 or more.
    #        2.1) You can generate new samples to train the NN model and save new samples into file df_server_number_K.csv.
    #        2.2) You can read samples from file df.server_number_K.csv and train the NN model. (change Generate_samples != 1)
    #  3) Run Reloaded_GIGK_DNN_CSFP.py to use the NN model for the prediction of the stationary distribution of the queue length. 
    #        Note: The trained NN model is saved in "model_GIGK(K)_CSFP_saved.weights.h5". The model is reloaded to do prediction.
    #        Note: Please change all K to the same value (e.g., 2)
    #        Note: You need to choose K in file 'DNN_GIGK_Queue_CSFP.py' and 'Reloaded_GIGK_DNN_CSFP.py'. 
    #
    ###### Files involved #####
    #      1) DNN_GIGK_Queue_CSFP.py
    #      2) GIGK_Input_output_CSFP.py
    #      3) Construction_QBD_for_GIGK_CSFP.py
    #      4) QBD_for_DT_PHPHK_CSFP.py
    #      5) Reloaded_GIGK_DNN_CSFP.py
    #      6) Save_Combine_Read_for_CSV_files.py
    #      7) model_GIGK(K)_CSFP_saved_{queue_type}.weights.h5; K = 1, 2, 3, ...
    #      8) df_continuous_server_number_K.csv: K = 1, 2, 3, ...


# #### Test functions in this file ####
# if __name__ == '__main__': 
#     # Train or retrain DNN model for a continuous queue with K server(s)
#     for K in [1, 2, 3]:
#         queue_type = 'continuous'
#         DNN(queue_type, K)
