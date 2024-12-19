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

### 1. `data/geometric_parameters_train.csv`
- This dataset contains the geometric parameters used for training the deeplearning models. It includes combinations of three parameters.
- **Parameter Range**:  
  - **p_u**, **p_l**, **t_d**: Vary from 0.05 µm to 1 µm.
- **Step Size**: 0.05 µm increments.
- **Total Combinations**: 2,600 unique parameter combinations.

### 2. `data/simulated_TM_results.csv`
- This file contains the simulated TM transmission spectra for each combination of geometric parameters in the training dataset.
- **Wavelength Range**: 2–6 µm with 0.1 µm increments.
- **Purpose**: The transmission results from this dataset are used as the ground truth for training the machine learning models.

### 3. `data/geometric_parameters_test.csv`
- This dataset contains a more finely tuned set of geometric parameters for testing and prediction.
- **Parameter Range**:  
  - **p_u**, **p_l**, **t_d**: Vary from 0.05 µm to 1 µm.
- **Step Size**: 0.01 µm increments, allowing for more precise parameter combinations.
- **Total Combinations**: 884,736 unique parameter combinations (using a 0.01 µm step).
- **Usage**: This dataset is used for testing and predicting TM transmission for finer combinations of parameters, and can be customized for even more granular testing.



## Training

The **Learning-based Wave Method (LnWM)** is the primary model used for predicting TM transmission spectra. It was trained using the geometric parameter combinations from `geometric_parameters_train.csv` and the corresponding TM transmission values from `simulated_TM_results.csv`.

- **Training File**: `models/LnWM_TM_prediction_train.py`  

For comparison, a **Multi-layer Perceptron (MLP)** model was also trained to predict TM transmission.

- **Training File**: `models/MLP_TM_prediction_train.py`  


### Model Saved Files
The trained models are saved as `.h5` files in the `model saved` folder.

- **LnWM Model**: `model saved/LnWM_TM_predictor.h5`  
- **MLP Model**: `model saved/MLP_TM_predictor.h5`

### SHAP Value Analysis
To understand the contribution of each feature to the predicted TM transmission, SHAP (Shapley Additive Explanations) values were used. The SHAP analysis provides insights into how each geometric parameter (p_u, p_l, t_d) influences the model's predictions.

#### SHAP Value Analysis Code:
The SHAP value analysis code can be found in the file `shap_analysis.py`. This script calculates and visualizes the SHAP values for the trained LnWM model, helping to explain the feature importance and the decision-making process behind the model's predictions.

- **SHAP Analysis File**: `shap_analysis.py`






