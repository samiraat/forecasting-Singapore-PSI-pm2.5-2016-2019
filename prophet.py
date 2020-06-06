import pandas as pd 

import numpy as np

import datetime

import matplotlib.pyplot as plt

from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from sklearn.metrics import mean_squared_error, r2_score ,mean_absolute_error

# read data 
df = pd.read_csv('psi_df_2016_2019.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp'] = df['timestamp'].dt.tz_localize(None)
#print(df.isnull().sum())

column = ['national' , 'south' , 'north' , 'east', 'central' , 'west']

train_prophet = pd.DataFrame()
test_prophet = pd.DataFrame()
for i in column:
    train_size_prophet = int(len(df) * 0.7)
    train_prophet['ds'] = df['timestamp'].iloc[:train_size_prophet]
    train_prophet['y'] = df[i].iloc[:train_size_prophet]
    test_prophet['ds'] = df['timestamp'].iloc[train_size_prophet:]
    test_prophet['y'] = df[i].iloc[train_size_prophet:]

    m = Prophet()
    m.fit(train_prophet)
    forecast = m.predict(test_prophet)
    #forecast.head()
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    fig1 = m.plot(forecast)
    fig2 = m.plot_components(forecast)
    plt.show(fig1)
    plt.show(fig2)
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    ax.scatter(test_prophet['ds'], test_prophet['y'], color='r')
    fig = m.plot(forecast, ax=ax)
    plt.show(fig)
    mse = mean_squared_error(y_true=test_prophet['y'],y_pred=forecast['yhat'])
    print("MSE :" ,mse)
