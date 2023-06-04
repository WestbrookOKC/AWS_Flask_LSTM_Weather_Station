import os

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Load the dataset and select the features
df = pd.read_csv('climate.csv')
df = df[:35000]
# Choose the important features
df = df[['Date Time','T (degC)', 'p (mbar)', 'rh (%)', 'VPmax (mbar)', 'wv (m/s)', 'rho (g/m**3)']]

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length), 1:]
        y = data[(i+seq_length), 1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 12  # 12 * 10 = 120 -> 2 hours latter
data = df.to_numpy()
x_train, y_train = create_sequences(data, seq_length)

x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
# Split the data into training and validation sets
train_size = int(len(x_train) * 0.8)
x_val, y_val = x_train[train_size:], y_train[train_size:]
x_train, y_train = x_train[:train_size], y_train[:train_size]

# Convert the data 
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
x_val = torch.from_numpy(x_val).float()
y_val = torch.from_numpy(y_val).float()


# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(1), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(1), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[-1])
        return out

# Set the hyperparameters 
input_size = 6 
hidden_size = 64 
output_size = 1 
learning_rate = 0.001
num_epochs = 20
batch_size = 64
# Check if the model file exists
if not os.path.isfile('LSTM_Model.pth'):


    model = LSTM(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    train_losses_plot = []
    val_losses_plot = []
    # train model and validate model
    for epoch in range(num_epochs):
        permutation = torch.randperm(x_train.size()[0])
        train_losses = []
        val_losses = []
        for i in range(0, x_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = x_train[indices], y_train[indices]
            optimizer.zero_grad()
            output = model(batch_x.transpose(0, 1))
            loss = criterion(output, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        with torch.no_grad():
            model.eval()
            train_output = model(x_train.transpose(0, 1))
            train_loss = criterion(train_output, y_train.unsqueeze(1))
            val_output = model(x_val.transpose(0, 1))
            val_loss = criterion(val_output, y_val.unsqueeze(1))
            val_losses.append(val_loss.item())
            print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, num_epochs,
                                                                                sum(train_losses)/len(train_losses),
                                                                                sum(val_losses)/len(val_losses)))
            train_losses_plot.append(sum(train_losses)/len(train_losses))
            val_losses_plot.append(sum(val_losses)/len(val_losses))
            model.train()

    # plot the loss curves
    plt.plot(train_losses_plot, label='Train Loss')
    plt.plot(val_losses_plot, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # Save the trained model
    torch.save(model.state_dict(), 'LSTM_Model.pth')

else:
    model = LSTM(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load('LSTM_Model.pth'))
