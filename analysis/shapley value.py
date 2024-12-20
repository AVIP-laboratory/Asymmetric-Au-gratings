import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import shap

model = load_model('file directory/LnWM_TM_predictior.h5')

X = pd.read_csv('file directory/geometric_parameters_train.csv.csv',header=None).values
Y = pd.read_csv('file directory/simulated_TM_results.csv',header=None).values
X = np.expand_dims(X, 1)
X = X.reshape(X.shape[0], 1, 3, 1)
print("Data shape:", X.shape)
print("First sample shape:", X[0].shape)

# Load the data to be analyzed
domain_x = pd.read_csv("file directory/analysis_data_x.csv", header=None).values
domain_y = pd.read_csv("file directory/analysis_data_y.csv", header=None).values
domain_x = np.expand_dims(domain_x, 1)
domain_x = domain_x.reshape(domain_x.shape[0], 1, 3, 1)

# Set up the SHAP Gradient Explainer
explainer = shap.GradientExplainer(model, X)

# Compute SHAP values for the analyzed data
shap_values = explainer.shap_values(domain_x)

# Print the results
print("Number of SHAP values:", len(shap_values))
print("Shape of each SHAP value:", shap_values[0].shape)

# Squeeze the SHAP values to remove unnecessary dimensions
shap_values = np.squeeze(shap_values)
