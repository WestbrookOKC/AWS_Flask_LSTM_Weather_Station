import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from LSTM import LSTM

# initialize the parameter of lstm
input_size = 6
hidden_size = 64
output_size = 1
# load model
model_LSTM = LSTM(input_size, hidden_size, output_size)
model_LSTM.load_state_dict(torch.load('LSTM_Model.pth'))
model = pickle.load(open('model2.pkl', 'rb'))
# use last 5000 rows as testset
df = pd.read_csv('climate.csv')
df = df[-5000:]

# Choose the input features
df = df[['T (degC)', 'p (mbar)', 'rh (%)', 'VPmax (mbar)', 'wv (m/s)', 'rho (g/m**3)']]
real_value = df['T (degC)'].shift(-12)
real_value = real_value[:-12]
# linear regression prediction
lr_preds = model.predict(df[:-12])

# lstm model prediciton
test_data = np.array([df.iloc[:-12]])
x_test = torch.from_numpy(test_data).float()

with torch.no_grad():
    model_LSTM.eval()
    predicted_temperature = model_LSTM(x_test)

lstm_preds = predicted_temperature.numpy()

# Calculate the MSE for Linear Regression
lr_mse = mean_squared_error(real_value, lr_preds)

# get the mse for lstm
lstm_mse = mean_squared_error(real_value, lstm_preds)

print("Linear Regression MSE:", lr_mse)
print("LSTM MSE:", lstm_mse)


# plot the predicted value with true value
t = range(1, len(real_value) + 1)

# plot actual values
plt.plot(t, real_value, label="Actual Values")

# plot predicted values of LSTM model
plt.plot(t, lstm_preds, label="LSTM Model")

# plot predicted values of linear regression model
plt.plot(t, lr_preds, label="Linear Regression Model")

plt.xlabel("Time")
plt.ylabel("Temperature")
plt.legend()
plt.show()
