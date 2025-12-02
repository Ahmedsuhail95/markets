import yfinance as yf
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
STOCK_SYMBOL = 'AAPL'  # Apple Inc.
START_DATE = '2020-01-01'
END_DATE = '2024-01-01'

# Data prep params
LOOKBACK_WINDOW = 60   # How many past days to use to predict the next day
TRAIN_SPLIT = 0.8      # 80% data for training

# Model params
INPUT_DIM = 1          # Only using 'Close' price
HIDDEN_DIM = 64        # Neurons in LSTM layer
NUM_LAYERS = 2         # Stacked LSTM layers
OUTPUT_DIM = 1         # Predicting 1 value (price)
NUM_EPOCHS = 100
LEARNING_RATE = 0.01

# Device configuration (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# 2. Data Preparation
# ==========================================
def load_and_process_data(symbol, start, end, window):
    print(f"Downloading data for {symbol}...")
    data = yf.download(symbol, start=start, end=end, progress=False)
    
    # We only care about the 'Close' price
    data = data[['Close']].values.astype(float)
    
    # Normalize data (Scale between 0 and 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_normalized = scaler.fit_transform(data.reshape(-1, 1))
    
    # Create sequences
    # X: Data from (t) to (t+window)
    # y: Data at (t+window+1)
    X, y = [], []
    for i in range(len(data_normalized) - window):
        X.append(data_normalized[i:i+window])
        y.append(data_normalized[i+window])
        
    X = np.array(X)
    y = np.array(y)
    
    # Convert to PyTorch tensors
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    
    return X, y, scaler, data

# Load data
X, y, scaler, raw_data = load_and_process_data(STOCK_SYMBOL, START_DATE, END_DATE, LOOKBACK_WINDOW)

# Split into train and test sets
train_size = int(len(X) * TRAIN_SPLIT)
test_size = len(X) - train_size

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Move to device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

print(f"Training shape: {X_train.shape}")
print(f"Testing shape: {X_test.shape}")

# ==========================================
# 3. Define LSTM Model
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        
        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

model = LSTMModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM).to(device)

# ==========================================
# 4. Training Loop
# ==========================================
criterion = torch.nn.MSELoss()    # Mean Squared Error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\nStarting training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    outputs = model(X_train)
    optimizer.zero_grad()
    
    # Calculate loss
    loss = criterion(outputs, y_train)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.5f}')

# ==========================================
# 5. Prediction & Visualization
# ==========================================
model.eval()
with torch.no_grad():
    # Predict on training data to see how well it learned
    train_predict = model(X_train)
    # Predict on test data (unseen)
    test_predict = model(X_test)

# Inverse transform to get actual prices back (un-normalize)
train_predict = scaler.inverse_transform(train_predict.cpu().numpy())
y_train_actual = scaler.inverse_transform(y_train.cpu().numpy())
test_predict = scaler.inverse_transform(test_predict.cpu().numpy())
y_test_actual = scaler.inverse_transform(y_test.cpu().numpy())

# Plotting
plt.figure(figsize=(14, 6))

# Plot training data
plt.subplot(1, 2, 1)
plt.plot(y_train_actual, label='Actual Price')
plt.plot(train_predict, label='Predicted Price')
plt.title('Training Data Prediction')
plt.legend()

# Plot testing data (The real test)
plt.subplot(1, 2, 2)
plt.plot(y_test_actual, label='Actual Price')
plt.plot(test_predict, label='Predicted Price')
plt.title(f'Testing Data Prediction ({STOCK_SYMBOL})')
plt.legend()

plt.tight_layout()
plt.show()

print("\nDone! Check the plot for results.")