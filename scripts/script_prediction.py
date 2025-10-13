#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 15:02:58 2025
    Part I:  Comparison and performance evaluation
    Part II: Batch Evaluation of samples
    Part III: Outlier evaluation
        1. Predicting the stationary distribution of queue length for a new sample.
        2. Predicting the stationary distribution of queue length for a new sample located inside or outside the SCV_a and SCV_s regions of the DNN training samples.
    Part IV:Cross comparison of DNNs
    Part V: Shipped
    Part VI: Compare with GI/M/K
@author: z365wu and q7he
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from Sampling.QBD_for_DT_PHPHK_CSFP import Discrete_PH_PH_K
from Training.Class_training import NN_for_Queue
from Sampling.QBD_for_CT_PHPHK_CSFP import CTPHPHK_Stationary_Queue_Length
from Prediction.Comparison_and_validation import PH_Represent, Compare_Queue_quantity_batch, Out_space_samples
from Prediction.Comparison_and_validation import Compare_Queue_quantity, df_quantity_Compare_to_latex
from Prediction.Cross_comparison_DNNs import Cross_comparison_of_DNNs
from Prediction.GIMK_Queue import GIMK_Queues, Simulation_StatDist_CT
from Prediction.NN_model_Baron import Baron2024
from Prediction.DNN_GGC import GGC


def Comparison_performance_evaluation(Rho_list, DNN_net):
    '''
    Part I: Comparison and performance evaluation (for Section 5: Examples 1, 2, 3) #######################
    DNN Comparison for 1 sample (for continuous/discrete/mixed)
    For the mixed case, randomly generate a sample to be either discrete or continuous, each with a probability of 0.5.
    Figures: Queue length distribution; Tables: Queuing Quantity Comparision #####
    Figure: Queue length distribution by Whitt1993, Simulation, DNN, and QBD
    Table: Queuing quantity comparision across Whitt1993, Simulation, DNN, and QBD
    Plot queue length distribution of PH/PH/K by QBD, NN, simulation, and Whitt 1993
    for  rho = 0.27, rho = 0.66, and 0.95
    '''
    
    # # Generate samples for a give rho:  0.27, 0.66, 0.95
    for K in [1, 2, 3]: # number of servers
        # Load NN model for each server number K
        DNN_model = DNN_net.DNN()
        DNN_model.load_weights(f'Output/models/model_GIGK({str(K)})_CSFP_saved_{DNN_net.queue_type}.weights.h5')
        
        df_quantity_comparison = pd.DataFrame() # set up an empty dataframe to save queueing quantity comparision 
        for rho_test in Rho_list:
            # Plotting predicted stationary queue length distribution for the four methods 
            DNN_net.Plotting_queue_length_distribution(
                K, DNN_model, rho_lower=rho_test, rho_upper=rho_test, save_fig= True, fig_save_name = f'May_2025_{DNN_net.queue_type}'  #'Apr_2025_{DNN_net.queue_type}'
                ) # fig saved in the directory 'Figures/prediction_compare_..._Give_a_name_you_like.png'
            
            # Save queue length predictions
            if DNN_net.queue_type == 'continuous':
                file_route = f'Output/samples/queue_length_sample/Continuous_Test_queue_length_sample_K{K}_rho_{rho_test}.csv'
            elif DNN_net.queue_type == 'discrete':
                file_route = f'Output/samples/queue_length_sample/Discrete_Test_queue_length_sample_K{K}_rho_{rho_test}.csv'
            elif DNN_net.queue_type == 'mixed':
                file_route = f'Output/samples/queue_length_sample/Mixed_Test_queue_length_sample_K{K}_rho_{rho_test}.csv'
            DNN_net.df_StatDist.to_csv(file_route, index=False)
    
            # print(DNN_net.df_StatDist)
    
            # Compare the queueing quantity across Whitt1993, Simulation, DNN, and QBD
            # the probability of emprt system (q_0), the mean queue length (E[q_w]), 
            # mean waiting time (E[W]), and MSE
            df_sub = Compare_Queue_quantity(DNN_net.df_StatDist, Lmax, K, rho_test, DNN_net.queue_type)
            df_quantity_comparison = pd.concat([df_quantity_comparison, df_sub], axis = 0)
            
        # print('Compare the queueing quantity across Whitt1993, Simulation, DNN, and QBD')
        # print(df_quantity_comparison)
        # Convert to Latex form for Tables
        df_compare_latex_table = df_quantity_Compare_to_latex(df_quantity_comparison, DNN_net.queue_type)
        #print(df_compare_latex_table)
        # Save LaTeX table to a text file
        file_name = f'Output/Tables/Queueing_quantity_comparison_for_each_K{K}_{DNN_net.queue_type}.txt'
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(df_compare_latex_table)

        # Save queueing quantity comparision 
        file_route_q_quantity = f'Output/Tables/csv/Queue_quantity_comparison_sample_K{K}_{DNN_net.queue_type}.csv'
        df_quantity_comparison.to_csv(file_route_q_quantity, index=False)
        print('queue-type = ', DNN_net.queue_type)


def Batch_Evaluation_of_samples(Rho_list, DNN_net):
    '''
    Part II:  Batch Evaluation of samples (Section 5: Example 4) #########################
    Have mixed case here: For the mixed case, randomly generate samples to be either discrete or continuous, each with a probability of 0.5.#### 
    DNN Comparision over batchs (only for continuous/discrete so far)
    Figures: Queue length distribution; Tables: Queuing Quantity Comparision #####
    Figure: Queue length distribution by Whitt1993, Simulation, DNN, and QBD
    Table: Queuing quantity comparision across Whitt1993, Simulation, DNN, and QBD
    Plot queue length distribution of PH/PH/K by QBD, NN, simulation, and Whitt 1993
    for  rho = 0.27, rho = 0.66, and 0.95
    '''
    
    # Generate samples for a give rho:  0.27, 0.66, 0.95
    batch_size = 50     # to be changed to 50
    epochs = 1         # to be changed to 10
    for K in [1, 2, 3]:  #[1, 2, 3]: # number of servers   # Include K = 3 with m_max = 10 (see top)
        # Load NN model for each server number K
        DNN_model = DNN_net.DNN()
        DNN_model.load_weights(f'Output/models/model_GIGK({str(K)})_CSFP_saved_{DNN_net.queue_type}.weights.h5')
        # Note for epochs > 1, may need to add one more columns named 'batch number' to help understanding
        tab_latex = Compare_Queue_quantity_batch(Rho_list, batch_size, epochs, DNN_model, K, DNN_net.queue_type, m_max, n_max, Lmax)
    
        # Save LaTeX table to a text file
        file_name = f'Output/Tables/Queue_quantity_comparison_batch{batch_size}_{queue_type}_K{K}.txt'
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(tab_latex)    
    
        
def Outlier_evaluation(Rho_list, DNN_net): 
    '''
    # Part III: Outlier evaluation (Section 5. Example 5.5) ###############################################
    # Examples in or outside the SCV_a and SCV_s region of DNN training sampels
    # for continuous/discrete/mixed
    Parameters:
    - c_a_2_ubound : Upper bound for the squared coefficient of variation (SCV) of arrival times
    - c_s_2_ubound : Upper bound for the squared coefficient of variation (SCV) of service times.
    - out_sample : bool, default=True
       If True, generates out-samples cases where SCV of arrival times (`c_a_2`) **exceeds** `c_a_2_ubound`
       and SCV of service times (`c_s_2`) **exceeds** `c_s_2_ubound`.
       If False, generates in-sample cases where SCV of arrival times (`c_a_2`) **is below** `c_a_2_ubound`
       and SCV of service times (`c_s_2`) **is below** `c_s_2_ubound`.
    '''
    
    for K in [1]:  #, 2]  #, 3]:
        for rho_test in Rho_list:
            # Load NN model for each server number K
            DNN_model = DNN_net.DNN()
            DNN_model.load_weights(f'Output/models/model_GIGK({str(K)})_CSFP_saved_{DNN_net.queue_type}.weights.h5')
            # Plot: outside the SCV_a and SCV_s region
            rho_sample, c_a_2, c_s_2, df_StatDist = Out_space_samples(
                DNN_net.queue_type, K, m_max, n_max, Lmax, DNN_model, c_a_2_ubound=0.5, c_s_2_ubound=0.5, out_sample = True, rho_lower=rho_test, rho_upper=rho_test
            )
            #print(f'{K} servers')
            print(f'sample outside the SCV_a and SCV_s region: rho {rho_sample}, c_a_2 {c_a_2}, c_s_2 {c_s_2}')
            print(f'Figures saved in: Output/Figures/Out_of_space_sample/K{K}_SCVa_{round(c_a_2,1)}_SCVs_{round(c_s_2,1)}_rho_{round(rho_sample,2)}')
            
            # # Plot: inside the SCV_a and SCV_s region
            # rho_sample, c_a_2, c_s_2, df_StatDist = Out_space_samples(
            #     DNN_net.queue_type, K, m_max, n_max, Lmax, DNN_model, c_a_2_ubound=1, c_s_2_ubound=1, out_sample = False, rho_lower=rho_test, rho_upper=rho_test
            # ) 
            # print(f'{K} servers')
            # print(f'sample in the SCV_a and SCV_s region: rho {rho_sample}, c_a_2 {c_a_2}, c_s_2 {c_s_2}')
            # print(f'Figures saved in: Output/Figures/Out_of_space_sample/K{K}_SCVa_{round(c_a_2,1)}_SCVs_{round(c_s_2,1)}_rho_{round(rho_sample,2)}')


def Cross_comparison_DNNs(Rho_list, queue_type, DNN_net):
    '''
    Part IV: Cross comparison of DNNs (Section 5, Example 6) ##################
    '''
    
    if queue_type == 1:
        queue_type = 'continuous'
    elif queue_type == 0:
        queue_type = 'discrete'
    for K in [1, 2, 3]:
        for rho_test in Rho_list:
            rho_lower = rho_test
            rho_upper = rho_test
            DNN_model = DNN_net.DNN()
            Cross_comparison_of_DNNs(DNN_model, K, queue_type, m_max, n_max, Lmax, rho_lower, rho_upper)


def GIMK_queue_Example_Comparison(Lmax):  #  This function can be put in the file "GIMK_Queue.py"

    #### Uniform distribution on [a, b] for the interarrival time
    a, b = 0, 2    ### Then mean interarrival time = (a+b)/2
    arr_rate = 2/(a+b) ### Arrival rate
    DNN_net = NN_for_Queue(
        queue_type=queue_type, n_max=n_max, Lmax=Lmax, m_max=m_max, input_dim=input_dim, num_classes=Lmax
    )
    
    df_all = pd.DataFrame()
    # set up queue length list
    queue_list = np.arange(0, Lmax).reshape(-1,1)
    
    for K in [1, 2]:  # number of servers   # Include K = 3 with m_max = 10 (see top)
        # Load NN model for each server number K
        DNN_model_C = DNN_net.DNN()
        DNN_model_C.load_weights(f'Output/models/model_GIGK({str(K)})_CSFP_saved_continuous.weights.h5')
        DNN_model_D = DNN_net.DNN()
        DNN_model_D.load_weights(f'Output/models/model_GIGK({str(K)})_CSFP_saved_discrete.weights.h5')        
        DNN_model_M = DNN_net.DNN()
        DNN_model_M.load_weights(f'Output/models/model_GIGK({str(K)})_CSFP_saved_mixed.weights.h5') 

        for rho in [0.33, 0.66]:
            mu = arr_rate/(K*rho)
            
            # GI/M/K output
            Stat_Dis_GIMK, P0_GIMK, EStateDis_GIMK = GIMK_Queues(K, a, b, mu, Lmax)
            # Simulation output
            Stat_Dis_Simu, P0_Simu, EStateDis_Simu = Simulation_StatDist_CT(K, a, b, mu, Lmax)       
            
            Arrival_moments = np.zeros(10)
            Service_moments = np.zeros(10)
            Arrival_moments[0] = (a+b)/2
            Service_moments[0] = 1/mu
            for i in range (1, 10):
                Arrival_moments[i] = (np.power(b, i+2) - np.power(a, i+2))/((i+2)*(b-a))
                Service_moments[i] = Service_moments[i-1]*(i+1)/mu
            
            # DNN model output
            # normalize the first moment of service distribution to 1
            s1 = Service_moments[0]
            Arrival_norm = np.zeros(10)
            Service_norm = np.zeros(10)
            for i in range (n_max):
                Service_norm[i] = Service_moments[i]/np.power(s1, i+1)
                Arrival_norm[i] = Arrival_moments[i]/np.power(s1, i+1)
            Moments_A_S = np.concatenate((Arrival_norm, Service_norm)).reshape(1,-1)
            Moments_A_S = np.log(Moments_A_S+1) # Transform: +1 and then log transform 
            
            Stat_Dis_DNNC = DNN_model_C.predict(Moments_A_S)
            Stat_Dis_DNND = DNN_model_D.predict(Moments_A_S)
            Stat_Dis_DNNM = DNN_model_M.predict(Moments_A_S)
            
            # S(2025) output
            if Arrival_moments[0] == 1:
                Arrival_norm_s2025 = Arrival_moments
                Service_norm_s2025 = Service_moments
            else: # normalize Arrival_moments[0] to 1
                a1 = Arrival_moments[0]
                Arrival_norm_s2025 = np.zeros(10)
                Service_norm_s2025 = np.zeros(10)
                for i in range (n_max):
                    Arrival_norm_s2025[i] = Service_moments[i]/np.power(a1, i+1)
                    Service_norm_s2025[i] = Arrival_moments[i]/np.power(a1, i+1)
            Stat_Dis_GGC = GGC(Arrival_norm_s2025, Service_norm_s2025, K)
            
            # Simulation output
            #Simulation_StatDist_CT(m_a, alpha_a, T_a, m_s, alpha_s, T_s, K, Lmax)
                
            # combine Stationary distribution
            df_StatDist = pd.DataFrame()
            df_StatDist['DNNC'] = Stat_Dis_DNNC[0]
            df_StatDist['DNND'] = Stat_Dis_DNND[0]
            df_StatDist['DNNM'] = Stat_Dis_DNNM[0]  
            df_StatDist['GIMK'] = Stat_Dis_GIMK
            df_StatDist['Simulation'] = Stat_Dis_Simu[:Lmax]  
            df_StatDist['S(2025)'] = Stat_Dis_GGC  
            
            if K == 1: # using Baron2024 method only when K = 1
                print("B(2024)")
                #B(2024) output for K = 1 only    
                Stat_Dis_B2024 = Baron2024(Arrival_moments, Service_moments)
                df_StatDist['B(2024)'] = Stat_Dis_B2024
            
            mycolor = ['#1f77b4', '#bcbd22', '#7f7f7f', 'pink', '#2ca02c', '#d62728']
            if K ==1: # for Baron2024
                mycolor = ['#1f77b4', '#bcbd22', '#7f7f7f', 'pink', '#2ca02c', '#d62728', '#8c564b']
            
            # Plotting the bar plot
            N = int(rho*100/4)
            df_StatDist.iloc[0:N,].plot(kind='bar', figsize=(10, 6), color =  mycolor, rot=0)
            # Adding title and labels
            plt.title(f'Queue Length Distribution: $K$={K}, $\\rho={round(rho,2)}$', fontsize=14)
            plt.xlabel('Queue length', fontsize=14)
            plt.ylabel('Probability', fontsize=14)
            # Rotating x-axis labels for better visibility
            #plt.xticks(rotation=90)
            # Adding legend
            plt.legend(fontsize=14) # title='Methods'
            # Adjust layout
            plt.tight_layout()
            
            # file name for saving the figure
            figure_path = f'Output/Figures/GIMK/statDist_comparison_GIMK_K{K}_{queue_type}_queue_{rho}.png'
            # Save the figure
            plt.savefig(figure_path, dpi=200, bbox_inches='tight')
            
            # Displaying the plot
            plt.show()

            # Calculate expected queue length (exlcude the customers in the servers) 
            E_qw_DNNC = np.dot(df_StatDist['DNNC'][K:], queue_list[:(Lmax-K)])[0]
            E_qw_DNND = np.dot(df_StatDist['DNND'][K:], queue_list[:(Lmax-K)])[0]
            E_qw_DNNM = np.dot(df_StatDist['DNNM'][K:], queue_list[:(Lmax-K)])[0]
            E_qw_Sherzer = np.dot(df_StatDist['S(2025)'][K:], queue_list[:(Lmax-K)])[0]   
            if K == 1:
                E_qw_Baron = np.dot(df_StatDist['B(2024)'][K:], queue_list[:(Lmax-K)])[0]
            else: # it is empty fro B(2024)
                E_qw_Baron = '--'
            
            df_eq = pd.DataFrame([{
                'K': K,
                'rho': rho,
                'DNNC': E_qw_DNNC,
                'DNND': E_qw_DNND,
                'DNNM': E_qw_DNNM,
                'Simu': EStateDis_Simu,
                'GIMK': EStateDis_GIMK,
                'S(2025)': E_qw_Sherzer,
                'B(2024)': E_qw_Baron if K == 1 else '--'
            }])
            # concat data
            df_all = pd.concat([df_all, df_eq], axis=0)
            
    return df_all
    

def df_to_latex_GIMK(df):
    """
    Convert df with columns ['K', 'rho', 'DNNC', 'DNND', 'DNNM', 'GIMK', 'Simulation', 'S(2025)', 'B(2024)']
    into a LaTeX table grouped by K, with:
        - multirow for K centered
        - rho formatted as 'ρ=0.xx'
        - rho column header left empty
        - all numeric values rounded to 2 decimals
        - no booktabs rules
    """
    # Convert df into a LaTeX table
    # Select the desired columns (ensure correct order)
    cols = ['GIMK', 'Simu', 'DNNC', 'DNND', 'DNNM', 'S(2025)', 'B(2024)']
    df = df[['K', 'rho'] + cols].copy()

    # Round numeric columns to 2 decimals, convert to string
    for c in cols:
        df[c] = df[c].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float, np.floating)) else x)

    # Format rho column as ρ=xx.xx
    df['rho'] = df['rho'].apply(lambda x: f"$\\rho$={x:.2f}" if isinstance(x, (int, float, np.floating)) else x)

    # Build LaTeX table
    latex = []
    latex.append(r'\begin{table}[H]')
    latex.append(r'\centering')
    latex.append(r'\tiny')
    latex.append(r'\caption{Mean Queue Lengths by $K$ and $\rho$}')
    latex.append(r'\begin{tabular}{c|c|' + 'c' * len(cols) + '}')
    latex.append(r'\hline')
    latex.append(r' &  & ' + ' & '.join(cols) + r' \\')  # Empty header for rho
    latex.append(r'\hline')
    
    # Group by K
    for K, group in df.groupby('K'):
        group_sorted = group.sort_values('rho')
        n = len(group_sorted)
        first_row = True
        for _, row in group_sorted.iterrows():
            # format rho
            rho_val = row["rho"]
            try:
                rho_str = f'{float(rho_val):.2f}'
            except (ValueError, TypeError):
                rho_str = f'{rho_val}'
                
            if first_row:
                # multirow for K
                line = rf'\multirow{{{n}}}{{*}}{{\centering K={int(K)}}} & {rho_str}'
                first_row = False
            else:
                line = f' & {rho_str}'
            
            # format each cell value
            for c in cols:
                val = row[c]
                if isinstance(val, (int, float, np.number)):
                    val = f'{val:.2f}'
                elif isinstance(val, str) and val.replace('.', '', 1).isdigit():
                    val = f'{float(val):.2f}'
                # otherwise leave as is (like '--')
                line += f' & {val}'
            line += r' \\'
            latex.append(line)
        latex.append(r'\hline')

    latex.append(r'\end{tabular}')
    latex.append(r'\label{tab:Mean_Queue_Length_GIMK}')
    latex.append(r'\end{table}')

    return '\n'.join(latex)
            
            
if __name__ == '__main__':
    
    ##### Parameters
    n_max = 10           # The highest order of moments
    Lmax = 500           # The maximum queue length
    m_max = 15           # The maximum order of PH-representation (alpha, T)
    input_dim = 2*n_max  # DNN input dimension: arrival n_max + service n_max
    Rho_list = [0.27, 0.66, 0.95]  # List of traffic intensities (ρ) used for prediction
    
    # make directories for output tables and figures
    #os.makedirs('Output/Tables/', exist_ok=True)
    os.makedirs('Output/Figures/', exist_ok=True)
    os.makedirs('Output/Figures/GIMK/', exist_ok=True)
    os.makedirs('Output/Tables/csv/', exist_ok=True)
    os.makedirs('Output/samples/queue_length_sample', exist_ok=True)
    
    # Note: queue type can be a number in the range [0,1] ; proportion of continuous samples)
    # queue_type = 0   # Discrete time = 0; mixed = 0.5; 0 <= queue_type <= 1
    for queue_type in ['continuous', 'discrete', 0.5]:  # 0.5 is for the mixed case with 0.5*100% continous samples and (1 - 0.5) * 100% discrete samples
        print(f'queue type: {queue_type}')
        
        # Class initilization and DNN training for ALL functions below ###########
        DNN_net = NN_for_Queue(
            queue_type=queue_type, n_max=n_max, Lmax=Lmax, m_max=m_max, input_dim=input_dim, num_classes=Lmax
        )
        
        # # Part I: Comparison and performance evaluation (for Section 5: Examples 1, 2, 3)
        # print("Part I: Comparison and performance evaluation")
        Comparison_performance_evaluation(Rho_list, DNN_net)
        
        # # Part II:  Batch Evaluation of samples (Section 5: Example 4)
        # print("Part II:  Batch Evaluation of samples")
        # Batch_Evaluation_of_samples(Rho_list, DNN_net)
            
        # # Part III: Outlier evaluation (Section 5. Example 5.5) 
        # print("Part III: Outlier evaluation")
        # Outlier_evaluation(Rho_list, DNN_net)
        
        # # Part IV: Cross comparison of DNNs (Section 5, Example 6) 
        # print("Part IV: Cross comparison of DNNs")
        # if queue_type in ['continuous', 'discrete']:
        #     Cross_comparison_DNNs(Rho_list, queue_type, DNN_net)
        
        # Part V: Find the accuracy rates for rho from 0, 0.1, 0.2, ..., to 1.
        # print('Part V: Find the accuracy rates')
        # accuracy_rates(queue_type, DNN_net, accuracy_bd = 0.05)

    # Part VI: GI/M/K queues
    print("Part VI: GI/M/K queues")
    df_all = GIMK_queue_Example_Comparison(Lmax)
    df_latex_GIMK = df_to_latex_GIMK(df_all)
    # Save LaTeX table to a text file
    file_name = 'Output/Tables/GIMK_queue_Latex_table.txt'
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(df_latex_GIMK)
    

    ###################################################################################
    ##### This is for Part V that has not been used in the paper ###################
    # def Accuracy_on_rho(queue_type, K, m_max, n_max, Lmax, NN_model, rho_lower, rho_upper, accuracy_bd, max_moment_bound = 1.0e+30):
    #     '''
    #     Test accuracy on rho from 0 to 0.1, 0.1 to 0.2, ..., 0.9 to 1
    #     '''
        
    #     if queue_type == 1 or queue_type == 'continuous': # generate continuous queue; queue_type = 1 means continuous queue.
    #         # generate random paramters for continuous PD distribution
    #         print('queue type: continuous')
    #         rho, m_a, alpha_a, T_a, m_s, alpha_s, T_s, moments_Arrival, moments_Service = PH_Represent(1, K, m_max, n_max, Lmax, rho_lower, rho_upper, max_moment_bound = 1.0e+30)
    #         # i) calculate the stationary distribution using QBD method
    #         QBD_StatDist, _ = CTPHPHK_Stationary_Queue_Length(m_a, alpha_a, T_a, m_s, alpha_s, T_s, K, Lmax)
    #         # iii) NN model prediction
    #         Moments_A_S = np.concatenate((moments_Arrival, moments_Service)).reshape(1,-1)
    #         Moments_A_S = np.log(Moments_A_S+1) # Transform: +1 and then log transform
    #         NN_output = NN_model.predict(Moments_A_S)
               
    #     elif queue_type == 0 or queue_type == 'discrete': # generate discrete queue; queue_type = 0 means discrete queue.
    #         # generate random paramters for discrete PD distribution
    #         print('queue type: discrete')
    #         rho = np.random.random() * (rho_upper - rho_lower) + rho_lower
    #         # i) calculate the stationary distribution using QBD method
    #         m_a, alpha_a, T_a, m_s, alpha_s, T_s, moments_Arrival, moments_Service, QBD_StatDist = Discrete_PH_PH_K(0, 1, K, m_max, n_max, Lmax, rho_given = rho)
    #         # iii) NN model prediction
    #         moments_Arrival_log = np.log(1 + moments_Arrival)
    #         moments_Service_log = np.log(1 + moments_Service) 
    #         Moments_A_S = np.concatenate((moments_Arrival_log, moments_Service_log)).reshape(1,-1)
    #         NN_output = NN_model.predict(Moments_A_S)
    #     # combine QBD_StatDist and Simulation_StatDist_DT
    #     df_StatDist = pd.DataFrame()
    #     df_StatDist['NN'] = NN_output[0]
    #     df_StatDist['QBD'] = QBD_StatDist
    #     abs_diff = np.abs(df_StatDist['QBD'] - df_StatDist['NN']).sum()
        
    #     if accuracy_bd >= abs_diff: # the prediction accuracy is below accuracy threshold
    #         return 1, rho
    #     else:
    #         return 0, rho
        
        
    # def accuracy_rates(queue_type, DNN_net, accuracy_bd = 0.05):    
    #     '''
    #     Part V: Find the accuracy rates for rho from 0, 0.1, 0.2, ..., to 1.
    #     Parameters
    #     - accuracy_bd: accuracy threshold
    #     '''
        
    #     sample_num = 1
    #     #queue_type = 0
        
    #     for K in [1, 2, 3]:  #[1, 2, 3]: # number of servers   # Include K = 3 with m_max = 10 (see top)
    #         # Load NN model for each server number K
    #         Accounts = np.zeros(10)
    #         DNN_model = DNN_net.DNN()
    #         DNN_model.load_weights(f'Output/models/model_GIGK({str(K)})_CSFP_saved_{DNN_net.queue_type}.weights.h5')
    #         for i in np.arange(0, 10):
    #             rho_lower = i*0.1
    #             rho_upper = i*0.1 + 0.1
    #             for j in range (sample_num):
    #                  accuracy, rho_rand = Accuracy_on_rho(queue_type, K, m_max, n_max, Lmax, DNN_model, rho_lower, rho_upper, accuracy_bd, max_moment_bound = 1.0e+30)
    #                  #print(f"randomly generated rho: {rho_rand}")
    #                  if accuracy == 1:
    #                      Accounts[i] = Accounts[i] + 1
    #         print(f"Server numbers K = {K}; Accounts: {Accounts}")
