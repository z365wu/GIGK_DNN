#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:26:22 2025
Sample generation: (Note: Those four parts can run independently.)
        Part I:    Generate training samples for all
        Part II:   Generate pdf of samples (for Section 3.4.2 in the paper) 
        Part III:  Plot rho distribution of samples (for Section 3.4.3 in the paper)
        Part IV:   Plot SCV and its (convex) hull (for Section 3.4.4 in the paper) 
@author: z365wu and q7he
"""

from Sampling.Class_sample_generation import PHPHK_sample_generation  # For BOTH continuous and discrete PH/PH/K queues


def Generate_samples_for_all(NN_sample1, sample_size):
    '''
    Part I: Generate samples for all
    Generate Samples and Save for queues with K number servers.
    Specify the name of the CSV file to save the generated samples. The filename should be a string and should not 
    include the '.csv' extension. If a CSV file with the specified name already exists in the same directory, 
    append the newly generated samples to the existing file.
    Otherwise, create a new CSV file and save the generated samples.
    '''

    for K in [1, 2, 3]:  # K = [1], [2], [3], [1, 2], [1, 3], [2, 3], or [1, 2, 3]:
        file_name = f'Give_a_file_name_you_like_test1_K{K}'
        NN_sample1.Sample_Generation(Sample_size = sample_size, K = K, file_name = file_name)
    
    # Load and merge all samples for a given server number K; exclude later-generated samples where rho is near 0
    # df_sample_K = NN_sample1.Load_and_Merge_All_Samples_for_K(K = 3)   # K = 1, 2, or 3


def Generate_pdfs_of_PH_samples(NN_sample1, num_new_sample):
    '''
    Part II: Generate pdfs of PH-samples (for Section 3.4.2)
    Generate a PDF for a random variable X using a Continuous-Time Phase-Type (PH) representation
    Parameters:
    - generate_new_samples (bool, optional): If True, new samples will be generated. Default is True.
    - file_name_for_new_samples (str, optional): The file name for saving new samples. 
      Required only if generate_new_samples is True.
    - number_of_new_samples (int, optional): The number of new samples to generate. 
      Must be >= 1. Required only if generate_new_samples is True.
    Continuous or discrete is specified in NN_sample1() above.
    Plot sample PDF using new generated samples
    The code will save data to files and generate one plot:
       - A .pkl file containing a list of [sample_index, alpha, T, pdf_values].
       - A .csv file containing only 'x' values and corresponding pdf_values for plotting (plot 15 samples).
    ## Generate new continous/discrete time samples and plot
    '''
    
    NN_sample1.plot_sample_pdf(
        generate_new_samples = True, number_of_new_samples = num_new_sample, file_name_for_new_samples='Give_a_name_you_like'
    )
    # ## Plot sample PDF using existed continuous/discrete time PH-samples (plot 15 samples)
    NN_sample1.plot_sample_pdf(generate_new_samples = False)


def Plot_distribution_of_rho(NN_sample1):
    '''
    Part III: Plot the distribution of rho for the samples (for Section 3.4.3) ##########
    Plot the distribution of rho for training samples across different server numbers
    Parameters:
    - colors: List of colors corresponding to each K value
    - server_num_list: List of server numbers to include in the plot
    ### queue-type is specified in the NN_sample1() defined above
    '''
    
    NN_sample1.plot_rho(colors = ['blue', 'green', 'red'], server_num_list = [1, 2, 3])
    

def Plot_SCV_convex_hull(NN_sample1):
    '''
    Part IV: Plot SCV and its (convex) hull (for Section 3.4.4)
    Plot SCV and rho for all samples; exclude later-generated samples where rho is near 0
    Plot SCV of inter-arrival times vs. SCV of service times for all samples
    Exclude later-generated samples where rho is near 0
    The type of queues (continuous, discrete) is specified by queue-type in NN_sample1() defined above
    '''
    
    NN_sample1.plot_scv_a_s()
    
    # # Draw the convex hull around the main body of the samples (excluding outliers).
    # # threshold_percentage (float): Percentile threshold (e.g., 90) to filter out distant outliers before building the convex hull.
    # # (95%, 98%, 99.5%, 99.7% for the continuous case; 95%, 98%, 99.5%, 99.9% for the discrete case)
    
    NN_sample1.plot_scv_a_s_with_convex_hull(threshold_percentage = 95)


if __name__ == '__main__':
    ##### Parameters ##### 
    n_max = 10                # The highest order of moments
    Lmax = 500                # The maximum queue length
    m_max = 15                # The maximum order of PH-representation (alpha T)
    input_dim = 2*n_max       # DNN input dimension: arrival n_max + service n_max
    sample_size= 2            # The number of samples to be generated

    for queue_type in ['continuous', 'discrete']:  # generate samples for continuous and discrete queues
        print(f'queue type: {queue_type}')
        
        ## Class initilization for sample generation: Used by all parts in this file (Keep it) ###
        NN_sample1 = PHPHK_sample_generation(queue_type = queue_type, n_max = n_max, Lmax = Lmax, m_max = m_max)
        
        # Part I: Generate samples for all
        Generate_samples_for_all(NN_sample1, sample_size)
        
        # # Part II: Generate pdfs of PH-samples (for Section 3.4.2)
        # if queue_type =='continuous':
        #     num_new_sample = 15 # numbers of new samples
        #     Generate_pdfs_of_PH_samples(NN_sample1, num_new_sample)
        
        # # Part III: Plot the distribution of rho for the samples (for Section 3.4.3)
        # Plot_distribution_of_rho(NN_sample1)
        
        # Part IV: Plot SCV and its (convex) hull (for Section 3.4.4)
        # Plot_SCV_convex_hull(NN_sample1)