#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:31:03 2025

DNN Training:
        1. Train DNN models
        2. DNN validation
@author: z365wu and q7he
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import glob
from keras.models import Sequential
from keras.layers import Dense, Input
import tensorflow as tf

from Prediction.Comparison_and_validation import error_average_best_worst, df_error_to_latex_table, Compare_StatDist
from Sampling.Save_Combine_Read_for_CSV_files import Load_Samples_from_file


def total_plus_max_absolute_error(y_true, y_pred):
    ### Custom loss function:
    ### Sum of total absolute error and max absolute error per sample, averaged over batch.
    ### y_true, y_pred: tensors of shape (B, l)
    
    abs_diff = tf.abs(y_true - y_pred)                 # shape (B, l)
    total_abs_error = tf.reduce_sum(abs_diff, axis=1)  # shape (B,)
    max_abs_error = tf.reduce_max(abs_diff, axis=1)    # shape (B,)
    loss_per_sample = total_abs_error + max_abs_error
    return tf.reduce_mean(loss_per_sample)             # scalar    


def bounded_loss_accuracy(threshold=0.05):
    ### Custom accuracy: percentage of samples whose absolute loss is below a threshold.
    ### Args:     threshold (float): Maximum allowed loss to be counted as 'accurate'.
    ### Returns:  function: A TensorFlow function usable as a Keras metric.
    
    def accuracy_fn(y_true, y_pred):
        # Ensure both tensors are float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # Compute sample-wise absolute error
        abs_diff = tf.abs(y_true - y_pred)             # shape (B, l)
        total_abs_error = tf.reduce_sum(abs_diff, axis=1)  # shape (B,)
        # max_abs_error = tf.reduce_max(abs_diff, axis=1)    # shape (B,)
        # loss_per_sample = total_abs_error + max_abs_error
        loss_per_sample = total_abs_error
        
        # Check if error is within threshold
        accurate = tf.cast(loss_per_sample < threshold, tf.float32)
        
        # Average accuracy over batch
        return tf.reduce_mean(accurate)
    
    return accuracy_fn


#### This is where we define our deep neural network and related operations (functions) ##############
class NN_for_Queue: 
    """
    Neural network model for queueing systems, inheriting PH sample generation.
    Attributes:
    - queue_type (str)
    - n_max (int)
    - Lmax (int)
    - m_max (int)
    - input_dim (int): Dimension of input features.
    - num_classes (int): Number of output classes.
    """
    
    def __init__(self, queue_type, n_max, Lmax, m_max, input_dim, num_classes):
        
        if str(queue_type).lower() in ['1', 'continuous']:
            self.queue_type = 'continuous'
        elif str(queue_type).lower() in ['0', 'discrete']:
            self.queue_type = 'discrete'
        elif isinstance(queue_type, (int, float)) and 0 < queue_type < 1:
            self.queue_type = 'mixed'
            self.mix_q_ratio = queue_type
        else:
            raise ValueError("queue_type must be a number in [0, 1] or 'continuous' or 'discrete'.")
        print('The queue type is', self.queue_type)
        
        self.n_max = n_max # The highest order of moments
        self.Lmax = Lmax
        self.m_max = m_max # The maximum order of continous PH-representation (alpha, T)
        self.input_dim = input_dim
        self.num_classes = num_classes
     
    ### The definition and structure of the DNN model 
    def DNN(self):
        """
        Constructs a deep neural network model.
        """
        # Define the DNN model structure
        self.model = Sequential([
            Input(shape=(self.input_dim,)),
            Dense(512, activation='relu'),
            Dense(512, activation='relu'),
            Dense(512, activation='relu'),
            Dense(1024, activation='relu'),
            Dense(1024, activation='relu'),
            Dense(512, activation='relu'),
            Dense(512, activation='relu'),
            Dense(512, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])  
        
        # Coding model, selection, and loss function 
        self.model.compile(optimizer='adam', 
                           loss = total_plus_max_absolute_error, 
                           metrics = [bounded_loss_accuracy(threshold=0.05)])
        
        return self.model
    
    
    ### load_and_merge all samples for the queue type
    def Load_and_Merge_All_Samples_for_K(self, K, queue_type = True, save_route='Output/samples/'):
        
        if self.queue_type != 'mixed':
            queue_type = self.queue_type
            print(f'load samples: queue type is {queue_type}')
        else:
            print(f'load samples: queue type is {queue_type}')
        
        # Define the file pattern
        file_pattern = os.path.join(save_route, queue_type, 'K' + str(K), '*.csv')
    
        # Find all matching files
        files = glob.glob(file_pattern)
        self.files = files

        if not files:
            print("No matching files found.")
            return None
    
        # Load and concatenate all CSV files
        df_list = [pd.read_csv(file) for file in files]
        merged_df = pd.concat(df_list, ignore_index=True)
        
        return merged_df
    
    ##### Training DNN models: continuous, discrete, and mixed, for GI/G/K
    def training(self, K):
        """
        Trains a deep neural network model.
        Parameters:
            - K (int): Number of servers.
        Returns:
            - model (Sequential): Compiled (trained) Keras model.
        """
        # Load the saved DNN model for further training if avaiable; otherwise train a DNN model
        try:
            self.model.load_weights(f'Output/models/model_GIGK({str(K)})_CSFP_saved_{self.queue_type}.weights.h5')
            print('Load the saved DNN model for further training or prediction with K =', K)
        except:
            print('train a new DNN model with K = ', K)
        
        if self.queue_type != 'mixed': # for continuous or discrete queue
            # Obtain all samples for the queue with K servers
            self.df_all_K = self.Load_and_Merge_All_Samples_for_K(K) 
            # Break samples into Moments, stationary distribution, and rho blocks for NN training
            self.Moments_all, self.Stationary_all, self.rho_all = Load_Samples_from_file(self.n_max, self.Lmax, self.df_all_K)
            # convert to float
            self.Moments_all = self.Moments_all.astype(float)
            self.Stationary_all = self.Stationary_all.astype(float)
            
            # Use the first 80 percentage of the saved samples as training samples
            sample_num = int(self.Moments_all.shape[0] * 0.8)
            Moments_train, Stationary_train = self.Moments_all[:sample_num,:], self.Stationary_all[:sample_num,:]
        
        elif self.queue_type == 'mixed': # for mixed DNN: continuous time PH/PHKs  + discrete time PH/PH/Ks 
            ## Samples of 'discrete' queue
            # Obtain all samples for the queue with K servers  
            self.df_all_K = self.Load_and_Merge_All_Samples_for_K(K, queue_type = 'discrete')  
            # Break samples into Moments, stationary distribution, and rho blocks for NN training
            self.Moments_all, self.Stationary_all, self.rho_all = Load_Samples_from_file(self.n_max, self.Lmax, self.df_all_K)
            # convert to float
            self.Moments_all = self.Moments_all.astype(float)
            self.Stationary_all = self.Stationary_all.astype(float)
            sample_num = int(self.Moments_all.shape[0] * 0.8) # Use the first 80 percentage of the saved samples as training samples
            Moments_train_DT, Stationary_train_DT = self.Moments_all[:sample_num,:], self.Stationary_all[:sample_num,:]
            
            ## Samples of 'continuous' queue
            # Obtain all samples for the queue with K servers  
            self.df_all_K = self.Load_and_Merge_All_Samples_for_K(K, queue_type = 'continuous')  
            # Break samples into Moments, stationary distribution, and rho blocks for NN training
            self.Moments_all, self.Stationary_all, self.rho_all = Load_Samples_from_file(self.n_max, self.Lmax, self.df_all_K)
            # convert to float
            self.Moments_all = self.Moments_all.astype(float)
            self.Stationary_all = self.Stationary_all.astype(float)
            sample_num = int(self.Moments_all.shape[0] * 0.8) # Use the first 80 percentage of the saved samples as training samples
            Moments_train_CT, Stationary_train_CT = self.Moments_all[:sample_num,:], self.Stationary_all[:sample_num,:]
                        
            # Get the minimal number of 'discrete' samples and 'continuous' samples
            DT_num = Moments_train_DT.shape[0]
            CT_num = Moments_train_CT.shape[0]
            
            if DT_num / CT_num <= (1 - self.mix_q_ratio)  / self.mix_q_ratio: # continuous samples more than required
                # use all 'discrete' samples
                # reduce continuous samples
                ct_s = int(DT_num * self.mix_q_ratio / (1 - self.mix_q_ratio))
                Moments_train_CT, Stationary_train_CT = Moments_train_CT[:ct_s,:], Stationary_train_CT[:ct_s,:]
            else: # discrete samples more than required
                # use all 'continuous' samples
                dt_s = int(CT_num * (1 - self.mix_q_ratio) / self.mix_q_ratio)
                Moments_train_DT, Stationary_train_DT = Moments_train_DT[:dt_s,:], Stationary_train_DT[:dt_s,:]
            
            print(f'Mixed ration: {self.mix_q_ratio}; DT_samples: {Moments_train_DT.shape[0]}; CT_samples: {Moments_train_CT.shape[0]}')
            
            # concat discrete and continuous samples
            Moments_train = np.vstack((Moments_train_CT, Moments_train_DT))
            Stationary_train = np.vstack((Stationary_train_CT, Stationary_train_DT))
        
        st = time.time()    # Record the total computing time
        # #### Train or retrain the DNN model ###################
        # #### Btoh batch_size and epochs are important hyper-parameters for training ##########################
        self.train_history = self.model.fit(Moments_train, Stationary_train, batch_size= 512, epochs= 100, verbose=1)  
        
        # Save the trained model
        self.model.save_weights(f'Output/models/model_GIGK({str(K)})_CSFP_saved_{self.queue_type}.weights.h5')
        print('A DNN model for GI/G/K model with K =', K, 'has been trained and saved')
        
        ## Save training loss and accuracy
        # Convert the training history into a DataFrame
        df_history = pd.DataFrame(self.train_history.history)
        df_history['K'] = K
        df_history['epoch'] = df_history.index + 1  # Add epoch number
        
        # Save it as a CSV (optional)
        df_history.to_csv(f'Output/models/training_history_GIGK({str(K)})_{self.queue_type}.csv', index=False)
        
        # Print the training time
        et = time.time()
        Elapsed_time = et - st
        print('Execution time is ', Elapsed_time, 'seconds.')
        
        # test
        self.Moments_train = Moments_train
    
    
    def Plot_DNN_training_accuracy_loss(self, K_list):
        '''Plot the DNN's training_accuracy and loss over epochs'''
        # Load accuracy and loss data over K
        df_all_history = pd.DataFrame()
        for K in K_list:
            df_history = pd.read_csv(f'Output/models/training_history_GIGK({str(K)})_{self.queue_type}.csv')
            df_all_history = pd.concat([df_all_history, df_history], axis = 0)
            
        # Plot Loss
        plt.figure(figsize=(10,6))
        for k_val in df_all_history['K'].unique():
            df_k = df_all_history[df_all_history['K'] == k_val]
            plt.plot(df_k['epoch'], df_k['loss'], marker='o', label=f'Loss (K={k_val})')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title(f'Training Loss over Epochs for $K=1, 2, 3$, DNNs for {self.queue_type} queues', fontsize=14)
        plt.legend()
        plt.grid(True)
        os.makedirs('Output/Figures/Training_accuracy_loss/', exist_ok=True)
        plt.savefig(f'Output/Figures/Training_accuracy_loss/training_loss_over_epochs_{self.queue_type}.png', dpi=200)
        plt.show()
        
        # Plot Accuracy
        plt.figure(figsize=(10,6))
        for k_val in df_all_history['K'].unique():
            df_k = df_all_history[df_all_history['K'] == k_val]
            acc_col = [col for col in df_k.columns if 'accuracy' in col][0]  # Auto-detect accuracy column
            plt.plot(df_k['epoch'], df_k[acc_col], marker='o', label=f'Accuracy (K={k_val})')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Bounded Accuracy', fontsize=14)
        plt.title(f'Training Accuracy over Epochs for each $K$: {self.queue_type} queue', fontsize=14)
        plt.legend()
        plt.grid(True)
        os.makedirs('Output/Figures/Training_accuracy_loss/', exist_ok=True)
        plt.savefig(f'Output/Figures/Training_accuracy_loss/training_accuracy_over_epochs_{self.queue_type}.png', dpi=200)
        plt.show()
    
    
    def Validation_test_DNN(self, K, DNN_model, batch_size = 10):
        '''
        1. We use 20 percent of generated samples for validation test: mean loss and accuracy of DNN prediction 
        2. Numerical examples and error analysis: average, min, and max 
           of Sum of absolute errors (SAE) and Percent relative error of the mean (REM)
           using the last 20 percentage of the saved samples to do out-of-sample test
        Parameter:
            - K: number of servers
            - batch_size: bath size of samples for each of the 10 tests
        '''
        
        # Use the last 20 percentage of the saved samples to do out-of-sample test
        # load data
        if self.queue_type != 'mixed': # for continuous and discrete queue
            # Obtain all samples for the queue with K servers
            self.df_all_K = self.Load_and_Merge_All_Samples_for_K(K) 
            # Break samples into Moments, stationary distribution, and rho blocks for NN training
            self.Moments_all, self.Stationary_all, self.rho_all = Load_Samples_from_file(self.n_max, self.Lmax, self.df_all_K)
            # convert to float
            self.Moments_all = self.Moments_all.astype(float)
            self.Stationary_all = self.Stationary_all.astype(float)
            
            # Use the last 20 percentage of the saved samples to do out-of-sample test
            sample_num = int(self.Moments_all.shape[0] * 0.8)
            Moments_test, Stationary_test = self.Moments_all[sample_num:,:], self.Stationary_all[sample_num:,:]
            
            # Normalize the moments for discrete queue
            if self.queue_type == 'discrete':
                Moments_test = np.exp(Moments_test) -1 # transfer back to moments
                Moments_test = Moments_test / Moments_test[:, self.n_max][:, None]  # normalize by dividing the 1st moment of service times
                Moments_test = np.log(Moments_test+1) # log transform
        
        if self.queue_type == 'mixed': # for mixed queue
        
            ## Samples of 'discrete' queue
            # Obtain all samples for the queue with K servers  
            self.df_all_K = self.Load_and_Merge_All_Samples_for_K(K, queue_type = 'discrete')  
            # Break samples into Moments, stationary distribution, and rho blocks for NN training
            self.Moments_all, self.Stationary_all, self.rho_all = Load_Samples_from_file(self.n_max, self.Lmax, self.df_all_K)
            # convert to float
            self.Moments_all = self.Moments_all.astype(float)
            self.Stationary_all = self.Stationary_all.astype(float)
            sample_num = int(self.Moments_all.shape[0] * 0.8) # Use the last 20 percentage of the saved samples to do out-of-sample test
            Moments_train_DT, Stationary_train_DT = self.Moments_all[sample_num:,:], self.Stationary_all[sample_num:,:]
            
            ## Samples of 'continuous' queue
            # Obtain all samples for the queue with K servers  
            self.df_all_K = self.Load_and_Merge_All_Samples_for_K(K, queue_type = 'continuous')  
            # Break samples into Moments, stationary distribution, and rho blocks for NN training
            self.Moments_all, self.Stationary_all, self.rho_all = Load_Samples_from_file(self.n_max, self.Lmax, self.df_all_K)
            # convert to float
            self.Moments_all = self.Moments_all.astype(float)
            self.Stationary_all = self.Stationary_all.astype(float)
            sample_num = int(self.Moments_all.shape[0] * 0.8) # Use the last 20 percentage of the saved samples to do out-of-sample test
            Moments_train_CT, Stationary_train_CT = self.Moments_all[sample_num:,:], self.Stationary_all[sample_num:,:]
                        
            # Get the minimal number of 'discrete' samples and 'continuous' samples
            DT_num = Moments_train_DT.shape[0]
            CT_num = Moments_train_CT.shape[0]
            
            if DT_num / CT_num <= (1 - self.mix_q_ratio)  / self.mix_q_ratio: # continuous samples more than required
                # use all 'discrete' samples
                # reduce continuous samples
                ct_s = int(DT_num * self.mix_q_ratio / (1 - self.mix_q_ratio))
                Moments_train_CT, Stationary_train_CT = Moments_train_CT[:ct_s,:], Stationary_train_CT[:ct_s,:]
            else: # discrete samples more than required
                # use all 'continuous' samples
                dt_s = int(CT_num * (1 - self.mix_q_ratio) / self.mix_q_ratio)
                Moments_train_DT, Stationary_train_DT = Moments_train_DT[:dt_s,:], Stationary_train_DT[:dt_s,:]
            
            print(f'Mixed ration: {self.mix_q_ratio}; DT_samples: {Moments_train_DT.shape[0]}; CT_samples: {Moments_train_CT.shape[0]}')
            
            # concat discrete and continuous samples
            Moments_test = np.vstack((Moments_train_CT, Moments_train_DT))
            Stationary_test = np.vstack((Stationary_train_CT, Stationary_train_DT))
            
            # randomly shuffle row index to mix continuous and discrete samples
            np.random.seed(42) # set seed for reproducibility
            random_index = np.random.permutation(Moments_test.shape[0])
            Moments_test = Moments_test[random_index]
            Stationary_test = Stationary_test[random_index]
            
        # prediction
        self.NN_output = DNN_model.predict(Moments_test)
        
        # Calculate loss
        self.test_loss = total_plus_max_absolute_error(Stationary_test, self.NN_output)
        
        ## calculate accuracy
        # Instantiate the accuracy function with your desired threshold
        accuracy_fn = bounded_loss_accuracy(threshold=0.05)
        self.test_accuracy = accuracy_fn(Stationary_test, self.NN_output)
        
        # Accuracy of Test Set: Use the last 20 percentage of the saved samples to do out-of-sample test
        df_error = error_average_best_worst(
            DNN_model, Moments_test, Stationary_test, self.Lmax, batch_size = batch_size
        )
        
        # Convert df_error into a LeTax Table
        df_error_latex_table = df_error_to_latex_table(df_error, self.queue_type, K)
        
        # print('samples test size: ', self.Moments_all.shape[0], Moments_test.shape[0])
        
        return tf.reduce_mean(self.test_loss).numpy(), tf.reduce_mean(self.test_accuracy).numpy(), df_error, df_error_latex_table
    
    
    #### Plotting predicted stationary queue length distribution for the four methods    
    # Randomly generate a sample (moments and stationary distribution of queue length) with the reqiured rho
    # Generate the DataFrame containing stationary distributions using QBD, Simulation, NN, and Whitt methods
    def Plotting_queue_length_distribution(self, K, NN_model, rho_lower, rho_upper, save_fig= False, fig_save_name = None):
        
        self.df_StatDist, rho = Compare_StatDist(self.queue_type, K, self.m_max, self.n_max, self.Lmax, NN_model, rho_lower, rho_upper)
            
        ### Maximal MSE between QBD and Simulation: plot QBD_StatDist vs. Simulation_StatDist vs. NN_StatDist
        # Plotting the bar plot
        self.df_StatDist.iloc[0:30,].plot(kind='bar', figsize=(10, 6), rot=0)
        
        # Adding title and labels
        plt.title(f'Queue Length Distribution: $K$={K}, $\\rho = {rho}$, {self.queue_type.capitalize()} Queue', fontsize=14)
        plt.xlabel('Queue length', fontsize=14)
        plt.ylabel('Probability', fontsize=14)
        # Rotating x-axis labels for better visibility
        #plt.xticks(rotation=90)
        # Adding legend
        plt.legend(fontsize=14) # title='Methods'
        # Adjust layout
        plt.tight_layout()
        # Save the figure
        if save_fig == True:
            # file name for saving the figure
            figure_path = f'Output/Figures/DNN_Comparison/prediction_compare_K{K}_rho_{rho}_' + fig_save_name + '.png'
            # Save the figure
            os.makedirs('Output/Figures/DNN_Comparison/', exist_ok=True)
            plt.savefig(figure_path, dpi=200, bbox_inches='tight')
        
        # Displaying the plot
        plt.show()
        
    
    
    