from tensorflow.keras.models import load_model
import pandas as pd

# Load the pre-trained LnWM or MLP model for TM transmission prediction
model = load_model('file directory/LnWM_TM_predictor.h5')  # or 'MLP_TM_predictor.h5'

# Load test data: finely tuned geometric parameter combinations from a CSV file
# Example: geometric parameters with 0.01 step increments
X0 = pd.read_csv('file directory/geometric_parameters_test.csv', header=None)

# Reshape the test data to match the input format of the model
X = X0.reshape(X0.shape[0], 1, X0.shape[1], 1)

# Predict TM transmission using the loaded model
yhat = model.predict(X)

# Save the predicted TM transmission results to a CSV file
yhat = pd.DataFrame(yhat)
yhat.to_csv("file directory/predicted_TM_LnWM.csv", index=False, header=None)
