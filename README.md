# Error Distribution Smoothing

This project provides Error Distribution Smoothing (EDS), a data management and processing framework, to tackle imbalanced regression using Python. It includes two main components and configurations for four experiments:

## Project Overview

The **`Manager`** class is designed to manage and update data points for training and testing. It uses **Delaunay** triangulation and **cKDTree** for efficient data management and interpolation. The **`process.py`** script demonstrates how to use the `Manager` class to process data in batches and save the results. Additionally, this project includes configurations for **4 experiments** to demonstrate EDS's effectiveness by selecting a representative subset from the entire dataset, reducing redundant data while ensuring the dataset remains balanced and representative.

## Project Components

1. **`manager.py`**: 

	A class to manage and update data points for training and testing, using Delaunay triangulation and cKDTree for efficient data management and interpolation.

2. **`process.py`**: 

	A script for batch data processing that updates the Manager and showcases how to utilize the class to process and save data in batches.

3. **Experiment Configurations**:
	
	- **1_Motivation_Example:** 
	
		This experiment illustrates Error Distribution Smoothing (EDS). 
	
		It includes a `dataset`, two scripts, `manager.py`  and `process.py` , for processing the dataset using the Extended Data Manager (EDS), and four additional scripts for evaluating and plotting the results.
	
	- **2_Dynamics_System_Identification**
	
		This experiment evaluates Error Distribution Smoothing (EDS) in the context of dynamic system identification.
	
		It includes a `dataset`, two scripts, `manager.py`  and `process.py` , for processing the dataset using the Extended Data Manager (EDS). Also, It includes a script `sindy_algorithm.py` using SIDNy to regress the datasets and another script `lorenz_sindy_comparison.py` to compare the results. The `lorenz_data_visualization.py` is used to visualize the results.
	
	- **3_Polar_Moment_of_Inertia**
	
		This experiment evaluates EDS in high-dimensional settings through using a synthetic dataset of white rectangles on a blak background with each rectangle labeled by its polar moment of inertia for regression.
	
		It includes a `dataset` directory, and places the features and labels for training and testing separately into the `rectangles_dataset` directory for plotting purposes. And again it includes two scripts, `manager.py`  and `process.py` to process the dataset using the Extended Data Manager (EDS). Additional scripts are utilized for tasks including regression analysis, performance evaluation, and data visualization.
	
	- **4_Real_World_Experiment**
		
		This experiment features two dynamic systems and evaluates the effectiveness of Error Distribution Smoothing (EDS) in addressing real-world issues characterized by high noise and significant imbalance.
		
		- 4_1_Cartpole
		
			It includes a `dataset`  and the same two scripts to `manager`  and `process` the dataset. Additional scripts are utilized for tasks including regression analysis, performance evaluation, and data visualization.
		
		- 4_2_Quadcopter
		
			It includes a `dataset`  and the same two scripts to `manager`  and `process` the dataset. Additional scripts are utilized for tasks including regression analysis, performance evaluation, and data visualization.

## Installation

To install the required dependencies, run the following command:

```bash
pip install numpy scipy
```

## Usage

### Manager Class

The `Manager` class is initialized with initial input and output data points, and an error threshold `delta`. It provides methods to update the data points, calculate prediction errors, and perform barycentric interpolation.

### Process Script

The `process.py` script demonstrates how to use the `Manager` class to process data in batches. It loads data from text files, initializes the `Manager` with an initial batch, and updates it with new data points in batches.

### Experiment Configurations

Each experiment configuration is designed to demonstrate a specific use case of the EDS.

