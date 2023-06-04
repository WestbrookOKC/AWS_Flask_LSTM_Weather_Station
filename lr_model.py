import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv('climate.csv')
# use the same train size with lstm
train_data = data[:28000]
test_data = data[28000:35000]

# Select input and output features
X = train_data[['T (degC)', 'p (mbar)', 'rh (%)', 'VPmax (mbar)', 'wv (m/s)', 'rho (g/m**3)']]
y = train_data["T (degC)"].shift(-12) # shift target variable forward 2 hours (12 x 10min = 2hr)

# Remove rows with NaN values, since last 12 rows do not have target value
X = X[:-12]
y = y[:-12]

X = X.values.reshape(-1, 6)

# fit data into lm
model = LinearRegression()
model.fit(X, y)

x_test = test_data[['T (degC)', 'p (mbar)', 'rh (%)', 'VPmax (mbar)', 'wv (m/s)', 'rho (g/m**3)']]
y_test = test_data["T (degC)"].shift(-12)

# Make prediction
y_pred = model.predict(x_test)

# true value
real_value = list(y_test)

# feature importance
importance = model.coef_
print(importance)

# store the model
# import pickle
# pickle.dump(model, open("model2.pkl", "wb"))

