#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:31:03 2025
Sample generation:
        1. Generate training samples
        2. Generate pdf of samples
        3. Plot scv and rho distribution of samples
@author: z365wu and q7he
"""
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import pickle
from scipy.spatial import ConvexHull

from Sampling.QBD_for_CT_PHPHK_CSFP import Input_Output_Moments_Generator_Continuous, CTPH_Rep_generator, CTPH_pdf
from Sampling.QBD_for_DT_PHPHK_CSFP import Input_Output_Moments_Generator_Discrete   #, DTPH_Rep_generator,
from Sampling.Save_Combine_Read_for_CSV_files import Save_Samples_to_File


# #### PH/PH/K samples generation for BOTH continuous and discrete time cases ###
class PHPHK_sample_generation:   # BOTH continuous and discrete
    
    #    Lmax: The maximum queue length
    #    m_max: The maximum order of PH-representation (alpha, T)
    #    n_max: The highest order of moments
    def __init__(self, queue_type, n_max, Lmax, m_max):
        self.queue_type = queue_type
        self.n_max = n_max
        self.Lmax = Lmax
        self.m_max = m_max
    
    
    def Sample_Generation(self, Sample_size, K, file_name, rho_lower = 0, rho_upper = 1, max_moment_bound = 1.0e+30, save_route='Output/samples/'):
        '''Generate samples for training DNN
        arrival moments, service moments, and stationary distribution of queue length
        '''
        #print(self.n_max, Sample_size, K, rho_lower, rho_upper)
        
        # Generate samples for continuous PH
        if self.queue_type == 'continuous':
            self.Moments, self.StaDist, self.SCVs, self.Rhos, queue_type_list, self.R_Iter_num = Input_Output_Moments_Generator_Continuous(
                Sample_size, K, self.m_max, self.n_max, self.Lmax, max_moment_bound, rho_lower, rho_upper
            )
        
        # Generate samples for discrete PH
        elif self.queue_type == 'discrete':
            self.Moments, self.StaDist, self.SCVs, self.Rhos, queue_type_list, self.R_Iter_num = Input_Output_Moments_Generator_Discrete(
                Sample_size, K, self.m_max, self.n_max, self.Lmax, max_moment_bound, rho_lower, rho_upper
            )
        
        # merge the generated samples into a DataFrame
        self.df_sample = Save_Samples_to_File(
            self.n_max, self.Lmax, K, self.Moments, self.StaDist, self.SCVs, self.Rhos, queue_type_list, self.R_Iter_num
        )
        
        # Construct the directory path
        save_directory = os.path.join(save_route, self.queue_type, 'K' + str(K))
        # Create the directory if it does not exist
        os.makedirs(save_directory, exist_ok=True)
        # Construct the full save path
        save_path = os.path.join(save_directory, file_name + '.csv')
        
        # Save samples
        try:
            # Attempt to read the CSV file if it exists.
            df_sample_exist = pd.read_csv(save_path)
            # Combine the DataFrame and the new DataFrame row-wise ignoring the index
            self.df_sample = pd.concat([df_sample_exist, self.df_sample], axis=0, ignore_index=True)
            # Save DataFrame to CSV
            self.df_sample.to_csv(save_path, index=False)
            print("The file already exists, appending new samples to the existing file.")
        except FileNotFoundError:
            # This block will execute if the CSV does not exist
            self.df_sample.to_csv(save_path, index=False)
            print("File not found. Creating a new file.")
        print(f"File saved successfully at: {save_path}")
        
        
    ### load_and_merge all samples for the queue type
    def Load_and_Merge_All_Samples_for_K(self, K, save_route='Output/samples/'):
        # Define the file pattern
        file_pattern = os.path.join(save_route, self.queue_type, 'K' + str(K), '*.csv')
    
        # Find all matching files
        files = glob.glob(file_pattern)
            
        if not files:
            print("No matching files found.")
            return None
    
        # Load and concatenate all CSV files
        df_list = [pd.read_csv(file) for file in files]
        merged_df = pd.concat(df_list, ignore_index=True)
        
        return merged_df
    
    
    ### Plot SCV of inter-arrival times vs. SCV of service times
    def plot_scv_a_s(self):
        # Concate all samples across K
        df_sample_K1 = self.Load_and_Merge_All_Samples_for_K(K=1)
        df_sample_K2 = self.Load_and_Merge_All_Samples_for_K(K=2)
        df_sample_K3 = self.Load_and_Merge_All_Samples_for_K(K=3)
        df_sample_all = pd.concat([df_sample_K1, df_sample_K2, df_sample_K3], ignore_index=True)
        
        SCV_a = df_sample_all['SCV_arrial'] # SCV of inter-arrival times
        SCV_s = df_sample_all['SCV_service'] # SCV of service times
        
        ### plot SCV_a vs SCV_s
        # Create a scatter plot
        plt.figure(figsize=(10, 6))  # Set the figure size
        plt.scatter(SCV_a, SCV_s, color='blue', alpha=0.3)  # Plot SCV_arrial vs SCV_service
        
        # Adding labels and title
        plt.title(f'SCV in Inter-arrival vs. Service Times: {self.queue_type} queue', fontsize=14)
        plt.xlabel('SCV of Inter-arrival Times', fontsize=14)
        plt.ylabel('SCV of Service Times', fontsize=14)
        
        # Set axis limits for continuous queues
        if str(self.queue_type) in ['1', 'continuous']:
            plt.xlim(0, 150)
            plt.ylim(0, 150)
        
        plt.grid(True)
        # Save the plot
        os.makedirs('Output/Figures/SCV_and_rho/', exist_ok=True)
        plt.savefig(f'Output/Figures/SCV_and_rho/SCVa_vs_SCVs_{self.queue_type}.png')
        # Show the plot
        plt.show() 
        
        
    def plot_scv_a_s_with_convex_hull(self, threshold_percentage):
        """
        Plot SCV (Squared Coefficient of Variation) of inter-arrival times vs service times,
        and draw the (convex) hull around the main body of the samples (excluding outliers).
        
        Args:
            threshold_percentage (float): Percentile threshold (e.g., 90) to filter out distant outliers
                                           based on distance from the center before building the convex hull.
        """

        # Concate all samples across K
        df_sample_K1 = self.Load_and_Merge_All_Samples_for_K(K=1)
        df_sample_K2 = self.Load_and_Merge_All_Samples_for_K(K=2)
        df_sample_K3 = self.Load_and_Merge_All_Samples_for_K(K=3)
        df_sample_all = pd.concat([df_sample_K1, df_sample_K2, df_sample_K3], ignore_index=True)
        
        # Extract (SCV_arrival, SCV_service) pairs as 2D points
        points = df_sample_all[['SCV_arrial', 'SCV_service']].values
        
        # Compute center
        center = np.mean(points, axis=0)
        
        # Compute Euclidean distances from center
        distances = np.linalg.norm(points - center, axis=1)

        ## Filter points within the specified threshold percentile distance
        threshold = np.percentile(distances, 95)   #threshold_percentage)
        filtered_points095 = points[distances <= threshold]
        threshold = np.percentile(distances, 98)   #threshold_percentage)
        filtered_points097 = points[distances <= threshold]
        threshold = np.percentile(distances, 99.5)   #threshold_percentage)
        filtered_points0995 = points[distances <= threshold]
        threshold = np.percentile(distances, 99.7)   #threshold_percentage)
        filtered_points0997 = points[distances <= threshold]
        
        # Plot
        plt.figure(figsize=(8,6))
        plt.scatter(points[:, 0], points[:, 1], color='lightgray', label='All Samples')
        plt.scatter(filtered_points0997[:, 0], filtered_points0997[:, 1], color='blue', label='Filtered Samples ($p$=0.997)')
        plt.scatter(filtered_points0995[:, 0], filtered_points0995[:, 1], color='green', label='Filtered Samples ($p$=0.995)')
        plt.scatter(filtered_points097[:, 0], filtered_points097[:, 1], color='yellow', label='Filtered Samples ($p$=0.98)')        
        plt.scatter(filtered_points095[:, 0], filtered_points095[:, 1], color='black', label='Filtered Samples ($p$=0.95)')
   
        ## Convex Hull on filtered points
        hull = ConvexHull(filtered_points0997)
        for simplex in hull.simplices:
            plt.plot(filtered_points0997[simplex, 0], filtered_points0997[simplex, 1], 'r-')
        hull = ConvexHull(filtered_points0995)
        for simplex in hull.simplices:
            plt.plot(filtered_points0995[simplex, 0], filtered_points0995[simplex, 1], 'r-')       
        hull = ConvexHull(filtered_points097)
        for simplex in hull.simplices:
            plt.plot(filtered_points097[simplex, 0], filtered_points097[simplex, 1], 'r-')
        hull = ConvexHull(filtered_points095)
        for simplex in hull.simplices:
            plt.plot(filtered_points095[simplex, 0], filtered_points095[simplex, 1], 'r-')

        plt.title(f'$SCV$ in Inter-arrival vs. Service Times: {self.queue_type} queue', fontsize=14)
        plt.xlabel('$SCV$ of Inter-arrival Time', fontsize=14)
        plt.ylabel('$SCV$ of Service Time', fontsize=14)
        # Set axis limits for continuous queues
        if str(self.queue_type) in ['1', 'continuous']:
            plt.xlim(0, 60)
            plt.ylim(0, 60)
            
        plt.legend()
        plt.grid(True)
        os.makedirs('Output/Figures/SCV_and_rho/', exist_ok=True)
        plt.savefig(f'Output/Figures/SCV_and_rho/SCVa_vs_SCVs_{self.queue_type}_with_hull.png')
        plt.show()
        
        # test only
        self.df_sample_all = df_sample_all
        
    ## Plotting the distrbution of rho of training samples over server numbers
    # colors: color for each K
    # server_num_list: list of the server numbers
    def plot_rho(self, colors, server_num_list, bins=200, alpha=0.5):
        
        fig, axs = plt.subplots(nrows=len(server_num_list), figsize=(10, 12))
        for i, server in enumerate(server_num_list):
            ax = axs[i]
            df_sample_K = self.Load_and_Merge_All_Samples_for_K(K=server)
            data = df_sample_K['Rho']
            ax.hist(data, bins=bins, color=colors[i], alpha=alpha, density=True)
            ax.set_title(r'Histogram of $\rho$' + f': {server} server(s), {self.queue_type}', fontsize=14)
            ax.set_xlabel(r'$\rho$', fontsize=14, fontstyle='italic')
            ax.set_ylabel('Empirical density function', fontsize=14)
            #ax.legend()
        #fig.tight_layout()
        os.makedirs('Output/Figures/SCV_and_rho/', exist_ok=True)
        plt.savefig(f'Output/Figures/SCV_and_rho/rho_distrbution_{self.queue_type}.png', dpi=200)
        plt.show()
        
        
    # Generate a PDF for a random variable X using a Continuous-Time Phase-Type (PH) representation
    def plot_sample_pdf(self, generate_new_samples = True, number_of_new_samples = None, file_name_for_new_samples=None):
        """
        Plots the sample PDF (probability density function).
        Parameters:
        - generate_new_samples (bool, optional): If True, new samples will be generated. Default is True.
        - file_name_for_new_samples (str, optional): The file name for saving new samples. 
          Required only if generate_new_samples is True.
          - number_of_new_samples (int, optional): The number of new samples to generate. 
          Must be >= 1. Required only if generate_new_samples is True.
        """
        
        # generate multiple pdfs of PH distribution samples
        self.df_save = [] # initialize a empty set to save the samples
        if generate_new_samples == True:
            self.df_pdf = pd.DataFrame(columns=['x'])
            x_values = np.arange(0, 3.1, 0.01)
            self.df_pdf['x'] = x_values
            df_sub = []
            
            for k in range(0, number_of_new_samples):
                m = np.random.randint(1, self.m_max)
                alpha, T = CTPH_Rep_generator(m, 1)
                one_vector = np.ones([m, 1])
                t_0 = np.dot(-T, one_vector)
                # Generate x values and compute the PDF
                pdf_values = [CTPH_pdf(x, alpha, T, t_0)[0] for x in x_values]
                self.df_pdf[f'pdf_{k}'] = pdf_values
                df_sub = [k, alpha, T, pdf_values]
                self.df_save.append(df_sub)

            # Save the DataFrame as a pickle (.pkl) file
            pkl_file_path = 'Output/samples/continuous/pdf_samples/' + file_name_for_new_samples + '.pkl'
            with open(pkl_file_path, 'wb') as pkl_file:
                pickle.dump(self.df_save, pkl_file)
            # Save df_pdf as a csv
            csv_file_path = 'Output/samples/continuous/pdf_samples/' + file_name_for_new_samples + '.csv'
            self.df_pdf.to_csv(csv_file_path,index=False)
                
            ### Plot all PDF samples
            plt.figure(figsize=(10, 6))
            for idx, col in enumerate(self.df_pdf.columns[1:]):  # Exclude the 'x' column
                plt.plot(self.df_pdf['x'], self.df_pdf[col], label=col)
            
        else: # load generated multiple pdfs of PH distribution samples
            pdf_sample_path = "Sampling/PH-distribution_samples.csv"
            self.df_pdf = pd.read_csv(pdf_sample_path)
    
            ### Plot multiple PDF samples
            plt.figure(figsize=(10, 6))
            for idx, col in enumerate(self.df_pdf.columns[[3, 6, 15, 33, 35, 79, 83,99, 10, 20, 85, 89, 91, 93, 94]]):  # Exclude the 'x' column
                plt.plot(self.df_pdf['x'], self.df_pdf[col], label=col)
            
        # Add title, labels, and legend (optinal)
        plt.title('Examples of sampled PH distributions', fontsize=14)
        plt.xlabel('x', fontsize=14, fontstyle='italic')
        plt.ylabel('Probability density function (PDF)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 4)  # Limit y-axis from 0 to 4
        #plt.legend(fontsize=10, title='PDF Samples')
        if generate_new_samples == True:
            save_path = 'Output/Figures/PH_density_functions_new_samples.png'
        else:
            save_path = 'Output/Figures/PH_density_functions.png'
        plt.savefig(save_path, dpi=200)
        plt.show()        

        return None
