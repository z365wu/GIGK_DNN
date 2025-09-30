## The directory includes codes and outputs for discrete/continuous/mixed queues: 
### 0. Setup Python environment
### 1. Main code
### 2. Sample generation
### 3. DNN training
### 4. DNN prediction

##########################

### Setup Python environment
0. The file '../requirements.txt' contains the version of the used Python environment and relevant packages.

### Main code
1. The script `script_main.py` imports Python modules for:
    1.1 Sample generation  
    1.2 DNN training  
    1.3 Prediction
    
Note: The specific scripts for the three purposes (1.1-1.3) are described below. The queue type must be specified at the beginning of the code as one of continuous, discrete, or mixed.
The mixed type is applicable only for DNN training and prediction.

### Sample generation
2. The code `script_sampling.py` generate DNN training samples based on PH distribution. The samples include
	2.1 Give the algorithm of constructing/calculating PH/PH/K distribution, its moments, and stationary distribution. The PH/PH/K distrition is used to generate GI/G/K samples, including its moments and stationary distribution. 
	2.2 The first 10 log-transfermed moments of (random) arrival and service times.
	2.3 The stationary distribution of queue lengths from 0 to 499. Functions to construct QBD
using the CSFP method and to compute the matrix-geometric solution.
	2.4 Saved in the folder 'Output/sampels'
	
Additional plots include:
    Probability density functions for continuous samples (Part II)
    The distribution of system load ρ (Part III)
    Plots of squared coefficient of variation (SCV) and its convex hull (Part IV)
    
### DNN training
3. The code `script_training.py`
	3.1 Builds DNN models for the GI/G/K queue.
	3.2 Part I: Trains or retrains the NN model and save its parameters in the folder 'Output/models'.
	3.3 Part II: Validates the performance of the DNN models through
	    3.3.1 Measuring DNN's mean loss and accuracy on test samples
	    3.3.2 Comparing queue length prediction between DNN, QBD, Simulation, Whitt1993, and Opher2023 methods.
	
### DNN prediction
4. The code `script_prediction.py`    
	4.1 Part I: Predictes the stationary distribution of queue length for a new sample, and compares the results with other methods..
	4.2 Part II: Predictes the stationary distribution of queue length for a batch of new samples, and compares the results with other methods.
	4.2 Part III: Predictes the stationary distribution of queue length for a new sample located inside or outside the SCV_a and SCV_s regions of the DNN training samples.
	4.4 Part IV: Performs cross-comparison among DNNs trained using continuous, discrete, and mixed sample types.

