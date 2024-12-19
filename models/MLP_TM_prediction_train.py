# Import required libraries
import pandas as pd
from tensorflow.keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score

# Load input and output data from CSV files
input = pd.read_csv('file directory/geometric_parameters_train.csv', header=None)# Input data with 3 features
output = pd.read_csv('file directory/simulated_TM_results.csv', header=None)# Output data with 41 features

model = Sequential()
model.add(Dense(5, input_dim=3, kernel_initializer='he_uniform', activation='relu'))
model.add(Dense(11, kernel_initializer='he_uniform', activation='relu'))
model.add(Dense(41, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy', 'mse'])

history = model.fit(input, output, epochs=100, batch_size=5)
model.summary()

model.save('file directory/MLP_TM_predictor.h5')

Y_pred = model.predict(input)
R2_score= r2_score(output, Y_pred)
print(R2_score)
