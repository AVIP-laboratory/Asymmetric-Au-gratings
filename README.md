# Asymmetric-Au-gratings

This project focuses on learning-based wave propagation approaches for optimizing asymmetric double-layer gold (Au) grating structures. The goal is to identify geometric parameter combinations that maximize TM transmission and enhance polarization performance.

## Datasets description

The datasets used for training, simulation, and testing are available in the data folder. Below is a detailed description of each dataset:

### Geometric Parameters Definition
The geometric parameters of the double-layer Au grating structure are defined as follows:  
- **Upper grating period (p_u)**: The period of the upper grating layer.  
- **Lower grating period (p_l)**: The period of the lower grating layer.  
- **Dielectric spacer thickness (t_d)**: The thickness of the dielectric spacer between the grating layers.

![structure](data/fig%201.png)

### 1. `geometric_parameters_train.csv`
- This dataset contains the geometric parameters used for training the deeplearning models. It includes combinations of three parameters.
- **Parameter Range**:  
  - **p_u**, **p_l**, **t_d**: Vary from 0.05 µm to 1 µm.
- **Step Size**: 0.05 µm increments.
- **Total Combinations**: 2,600 unique parameter combinations.

### 2. `simulated_TM_results.csv`
- This file contains the simulated TM transmission spectra for each combination of geometric parameters in the training dataset.
- **Wavelength Range**: 2–6 µm with 0.1 µm increments.
- **Purpose**: The transmission results from this dataset are used as the ground truth for training the machine learning models.

### 3. `geometric_parameters_test.csv`
- This dataset contains a more finely tuned set of geometric parameters for testing and prediction.
- **Parameter Range**:  
  - **p_u**, **p_l**, **t_d**: Vary from 0.05 µm to 1 µm.
- **Step Size**: 0.01 µm increments, allowing for more precise parameter combinations.
- **Total Combinations**: 884,736 unique parameter combinations (using a 0.01 µm step).
- **Usage**: This dataset is used for testing and predicting TM transmission for finer combinations of parameters, and can be customized for even more granular testing.


