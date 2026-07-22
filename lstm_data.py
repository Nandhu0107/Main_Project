import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

CSV_FILE = "preprocessed_traffic_data.csv"  
TARGET_COLUMN = "congestion_level"     
TIME_WINDOW = 5                         
OUTPUT_X = "X_time_windows.npy"
OUTPUT_Y = "y_labels.npy"

df = pd.read_csv(CSV_FILE)
print("Original shape:", df.shape)

y = df[TARGET_COLUMN].values
X = df.drop(columns=[TARGET_COLUMN])

X = X.select_dtypes(include=["int64", "float64"])

print("Feature shape (before normalization):", X.shape)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("Feature normalization completed.")
print("Min value:", X_scaled.min(), "Max value:", X_scaled.max())

X_windows = []
y_windows = []

for i in range(len(X_scaled) - TIME_WINDOW):
    X_windows.append(X_scaled[i:i + TIME_WINDOW])
    y_windows.append(y[i + TIME_WINDOW])

X_windows = np.array(X_windows)
y_windows = np.array(y_windows)

print("Time-window preparation completed.")
print("X_windows shape:", X_windows.shape)
print("y_windows shape:", y_windows.shape)

np.save(OUTPUT_X, X_windows)
np.save(OUTPUT_Y, y_windows)

print("Saved files:")
print(" -", OUTPUT_X)
print(" -", OUTPUT_Y)

print("Data is now READY for LSTM/ Bi-LSTM training.")