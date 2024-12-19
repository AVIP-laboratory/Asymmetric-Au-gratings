# This code trains a deep learning model using a custom architecture.
# Input: (2600, 1, 3, 1) shaped data from a CSV file
# Output: (2600, 1, 41, 1) shaped data from a CSV file

#Import TensorFlow and other libraries
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from keras.layers import Input, Dense, Flatten, Conv2D, Conv2DTranspose, Cropping2D, Concatenate, ReLU, BatchNormalization

# Load input and output data from CSV files
input = pd.read_csv('file directory/geometric_parameters_train.csv', header=None)
output = pd.read_csv('file directory/simulated_TM_results.csv', header=None)

X = input.values
Y = output.values

X = np.expand_dims(X, 1)
Y = np.expand_dims(Y, 1)
X = np.array(X)

X = X.reshape(2600, 1, 3, 1)
Y = Y.reshape(2600, 1, 41, 1)

input_shape = (1, 3, 1)
X_input = Input(input_shape)

# Transposed convolution with cropping to refine spatial dimensions
T11 = Conv2DTranspose(16, kernel_size=(1, 1), strides=(1, 4), kernel_initializer='he_uniform', padding="valid")(X_input)
T11_c = Cropping2D(cropping=((0, 0), (1, 0)))(T11)
T12 = Conv2DTranspose(16, kernel_size=(1, 2), strides=(1, 4), kernel_initializer='he_uniform', padding="valid")(X_input)
T12_c = Cropping2D(cropping=((0, 0), (0, 1)))(T12)
T13 = Conv2DTranspose(16, kernel_size=(1, 3), strides=(1, 4), kernel_initializer='he_uniform', padding="valid")(X_input)
T13_c = Cropping2D(cropping=((0, 0), (1, 0)))(T13)
T14 = Conv2DTranspose(16, kernel_size=(1, 5), strides=(1, 4), kernel_initializer='he_uniform', padding="valid")(X_input)
T14_c = Cropping2D(cropping=((0, 0), (1, 1)))(T14)
T15 = Conv2DTranspose(16, kernel_size=(1, 5), strides=(1, 3), kernel_initializer='he_uniform', padding="valid")(X_input)
T15_c = Cropping2D(cropping=((0, 0), (0, 0)))(T15)
TIM1 = Concatenate()([T11_c, T12_c, T13_c, T14_c, T15_c])
R1 = ReLU()(TIM1)
R1 = BatchNormalization()(R1)

R21 = Conv2D(1, kernel_size=(1, 1), strides=(1, 1))(R1)
T21 = Conv2DTranspose(16, kernel_size=(1, 1), strides=(1, 4), padding="valid")(R21)
T21_c = Cropping2D(cropping=((0, 0), (2, 1)))(T21)
R22 = Conv2D(1, kernel_size=(1, 1), strides=(1, 1))(R1)
T22 = Conv2DTranspose(16, kernel_size=(1, 3), strides=(1, 4), padding="valid")(R22)
T22_c = Cropping2D(cropping=((0, 0), (2, 1)))(T22)
R23 = Conv2D(1, kernel_size=(1, 1), strides=(1, 1))(R1)
T23 = Conv2DTranspose(16, kernel_size=(1, 5), strides=(1, 4), padding="valid")(R23)
T23_c = Cropping2D(cropping=((0, 0), (2, 2)))(T23)
R24 = Conv2D(1, kernel_size=(1, 1), strides=(1, 1))(R1)
T24 = Conv2DTranspose(16, kernel_size=(1, 6), strides=(1, 4), padding="valid")(R24)
T24_c = Cropping2D(cropping=((0, 0), (3, 2)))(T24)
R25 = Conv2D(1, kernel_size=(1, 1), strides=(1, 1))(R1)
T25 = Conv2DTranspose(16, kernel_size=(1, 11), strides=(1, 3), padding="valid")(R25)
T25_c = Cropping2D(cropping=((0, 0), (0, 0)))(T25)

TIM2 = Concatenate()([T21_c, T22_c, T23_c, T24_c,T25_c])
R2 = ReLU()(TIM2)
R2 = BatchNormalization()(R2)

MM = Flatten()(R2)
MM = Dense(41)(MM)
model = Model(inputs=X_input, outputs=MM)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy','mse'])

history = model.fit(X, output, epochs=100,batch_size=5, verbose=0)
model.summary()

model.save('file directory/LnWM_TM_predictor.h5')

Y_pred = model.predict(X)
R2_score= r2_score(output, Y_pred)
print(R2_score)
