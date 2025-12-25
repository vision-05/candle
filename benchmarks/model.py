import candle.nn as nn
from candle.activations import Dense, LeakyReLU
from candle.costs import MSE
from candle.optimisers import ADAM
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

data = pd.read_csv("Wifi_train_dataset.csv")

def split_rssi(df: pd.DataFrame) -> pd.DataFrame:

    dicts = []

    for i, r in df.iterrows():
        rssis = [int(x) for x in r['rssi'].split(',')]
        macs = [int(x) for x in r['mac_addrs_idx'].split(',')]
        row_dict = dict(zip(macs, rssis))
        dicts.append(row_dict)

    processed = pd.DataFrame(dicts)
    try:
        processed['x'] = df['x']
        processed['y'] = df['y']

    except:
        return processed

    return processed

fmt = split_rssi(data)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    #nan_dropped = df.dropna(thresh=0.5*df.shape[0], axis='columns') #remove columns with 20% or more NaNs
    nan_dropped = df.fillna(-100)
    
    return nan_dropped

fmt_clean = clean_data(fmt)
#find most important features
X = fmt_clean.drop(columns=['x','y']).values
train_cols = fmt_clean.drop(columns=['x', 'y']).columns
y = fmt_clean[['x', 'y']].values

indices = np.arange(len(X))
test_mask = (indices % 5) == 0
train_mask = ~test_mask

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

print(y_train_scaled)

model = nn.nn()
model.model(Dense(X_train.shape[1], 1024, True),
            LeakyReLU(),
            Dense(1024, 1024, True),
            LeakyReLU(),
            Dense(1024, 1024, True),
            LeakyReLU(),
            Dense(1024,1024, True),
            LeakyReLU(),
            Dense(1024,2, True),
            MSE(),
            ADAM())

model.train(X_train, y_train_scaled, X_test, y_test_scaled)
preds = model.predict(X_test)

def plot_acc(Y_true, Y_test, X_test, model):
    plt.subplot(1, 2, 2)
    plt.plot(Y_test[:,0], Y_test[:,1], 'x')
    plt.plot(Y_true[:,0], Y_true[:,1], 'o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['Predicted', 'Actual'])
    plt.show()

    mse_lr = mean_squared_error(Y_true, Y_test)
    mse_x = mean_squared_error(Y_true[:,0], Y_test[:,0])
    mse_y = mean_squared_error(Y_true[:,1], Y_test[:,1])

    # The Root Mean Squared Error (RMSE) is the error in the units of your coordinates
    rmse_lr = np.sqrt(mse_lr)

    # To get the average distance error (Euclidean Distance):
    # 1. Calculate the squared differences for x and y
    squared_diff = np.square(Y_true - Y_test)
    sum_of_squares = np.sum(squared_diff, axis=1)
    distance_error_per_sample = np.sqrt(sum_of_squares)
    avg_distance_error_lr = np.mean(distance_error_per_sample)

    m = Y_true.shape[0]
    mpe = np.sum(np.sum((np.abs(Y_true-Y_test) / np.abs(Y_true)), axis=1))/(2*m)
    mpe_x = np.sum((np.abs(Y_true[:,0]-Y_test[:,0]) / np.abs(Y_true[:,0])))/(m)
    mpe_y = np.sum((np.abs(Y_true[:,1]-Y_test[:,1]) / np.abs(Y_true[:,1])))/(m)

    print(f'm {m} mpe {mpe*100:.2f}% mpe_x {mpe_x*100:.2f}% mpe_y {mpe_y*100:.2f}%')

    print(f"LR Average Euclidean Distance Error: {avg_distance_error_lr:.2f}")
    print(f"MSE: {mse_lr} MSE_x: {mse_x} MSE_y: {mse_y}")

plot_acc(y_test, y_scaler.inverse_transform(preds), X_test, model)
