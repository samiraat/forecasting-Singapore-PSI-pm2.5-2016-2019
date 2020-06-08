import pandas as pd 

import numpy as np

import datetime

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM , Dropout, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score ,mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# read data 
df = pd.read_csv('psi_df_2016_2019.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp'] = df['timestamp'].dt.tz_localize(None)
#print(df.describe())

df.set_index('timestamp' , inplace=True)
#check if null value
#print(df.isna().sum())

#normalize data
def normalize_data(df):
    df.plot(subplots=True, figsize=(8, 8)); plt.legend(loc='best')
    plt.suptitle('hourly PM2.5 concenteration - BEFORE NORMALIZATION')
    plt.show()
    scaler = MinMaxScaler()
    df2 =scaler.fit_transform(df.values)
    df2 = pd.DataFrame(df2 , index= df.index , columns = df.columns)
    return df2




#prepare data
def load_data(data, seq_len , col):
    X_train = []
    y_train = []
    for i in range(seq_len, len(data)):
        X_train.append(data.iloc[i-seq_len : i, col ])
        y_train.append(data.iloc[i, col])
    
    
    #1 last 9007 days are going to be used in test
    train_size = int(0.7 * len(data))
    print(train_size)
    X_test = X_train[train_size: ]             
    y_test = y_train[train_size: ]
    
    #2 first 21015 days are going to be used in training
    X_train = X_train[:train_size ]           
    y_train = y_train[:train_size ]
    
    #3 convert to numpy array
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    #4 reshape data to input into RNN models
    X_train = np.reshape(X_train, (train_size, seq_len, 1))
    
    X_test = np.reshape(X_test, (X_test.shape[0], seq_len, 1))
    
    return [X_train, y_train, X_test, y_test]


def plot_predictions(test, predicted, title):
    plt.figure(figsize=(20,5))
    plt.plot(test, color='blue',label='Actual')
    plt.plot(predicted, alpha=0.7, color='orange',label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Normalized Pm2.5 concentration scale')
    plt.legend()
    plt.show()
    

df_norm = normalize_data(df)
print(df_norm)
df_norm.plot(subplots=True, figsize=(8, 8)); plt.legend(loc='best')
plt.suptitle('hourly PM2.5 concentration - AFTER NORMALIZATION')
plt.show()
plt.matshow(df.corr(method='spearman'),vmax=1,vmin=-1,cmap='PRGn')
plt.title('Correlation columns', size=15)
plt.colorbar()
plt.show()
plt.matshow(df.resample('M').mean().corr(method='spearman'),vmax=1,vmin=-1,cmap='PRGn')
plt.title('resampled over month', size=15)
plt.colorbar()
plt.margins(0.02)
plt.matshow(df.resample('D').mean().corr(method='spearman'),vmax=1,vmin=-1,cmap='PRGn')
plt.title('resampled over Day', size=15)
plt.colorbar()
plt.show()
column = {0 : 'national',1 : 'south',2 : 'north',3 : 'east',4 :'central',5 : 'west'}
for j in column.keys() :
    print("LSTM model for pm2.5 concentration on " + column[j]  + " station")
    seq_len = 23 #choose sequence length

    X_train, y_train, X_test, y_test = load_data(df_norm, seq_len , j )

    print('X_train.shape = ',X_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('X_test.shape = ', X_test.shape)
    print('y_test.shape = ',y_test.shape)



    rnn_model = Sequential()

    rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=True, input_shape=(X_train.shape[1],1)))
    rnn_model.add(Dropout(0.15))

    rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=True))
    rnn_model.add(Dropout(0.15))

    rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=False))
    rnn_model.add(Dropout(0.15))

    rnn_model.add(Dense(1))

    rnn_model.summary()

    rnn_model.compile(optimizer="adam",loss="MSE")
    rnn_model.fit(X_train, y_train, epochs=10, batch_size=1000)
    rnn_predictions = rnn_model.predict(X_test)
    rnn_score = r2_score(y_test,rnn_predictions)
    print("accuracy of RNN model = ",rnn_score)


    plot_predictions(y_test, rnn_predictions, "Predictions made by simple RNN model")
    
    lstm_model = Sequential()

    lstm_model.add(LSTM(40,activation="tanh",return_sequences=True, input_shape=(X_train.shape[1],1)))
    lstm_model.add(Dropout(0.15))

    lstm_model.add(LSTM(40,activation="tanh",return_sequences=True))
    lstm_model.add(Dropout(0.15))

    lstm_model.add(LSTM(40,activation="tanh",return_sequences=False))
    lstm_model.add(Dropout(0.15))

    lstm_model.add(Dense(1))

    lstm_model.summary()
    lstm_model.compile(optimizer="adam",loss="MSE")
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=1000)
    lstm_predictions = lstm_model.predict(X_test)

    lstm_score = r2_score(y_test, lstm_predictions)
    print("R^2 Score of LSTM model = ",lstm_score)
    plot_predictions(y_test, lstm_predictions, "Predictions made by LSTM model")
    plt.figure(figsize=(15,8))

    plt.plot(y_test, c="orange", linewidth=3, label="Original values")
    plt.plot(lstm_predictions, c="red", linewidth=3, label="LSTM predictions")
    plt.plot(rnn_predictions, alpha=0.5, c="green", linewidth=3, label="RNN predictions")
    plt.legend()
    plt.title("Predictions vs actual data", fontsize=20)
    plt.show()

