import pandas as pd 
import itertools 
import numpy as np

import datetime

import matplotlib.pyplot as plt

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

import pmdarima as pm

# read data 
df = pd.read_csv('psi_df_2016_2019.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'])

df['timestamp'] = df['timestamp'].dt.tz_localize(None)

#print(df.info())

df.set_index('timestamp' , inplace=True)
# is there any missing value?
#print(df.isnull().sum())


col = ['national' , 'south' , 'north' , 'east', 'central' , 'west']
station = col[5]
df_day = df.resample('D').mean()
df_day = df_day.dropna()
#print(df_day.isnull().sum())

# Resampling to monthly frequency
df_month = df.resample('M').mean()

# Resampling to annual frequency
df_year = df.resample('A-DEC').mean()

# Resampling to quarterly frequency
df_Q = df.resample('Q-DEC').mean()

#df_year.plot()
#plt.show()

def plot_bytime(df_day, df_month,df_Q,df_year):
    # PLOTS
    fig = plt.figure(figsize=[15, 7])
    plt.suptitle('Pm2.5 concentration in 5 station', fontsize=22)

    plt.subplot(221)
    plt.plot(df_day)
    plt.legend()

    plt.subplot(222)
    plt.plot(df_month)
    plt.legend()

    plt.subplot(223)
    plt.plot(df_Q)
    plt.legend()

    plt.subplot(224)
    plt.plot(df_year)
    plt.legend()

    # plt.tight_layout()
    plt.show()

#seasonal decomposition to check stationary of data
def sd(x , column):
    decomposition = seasonal_decompose(x , freq = 161)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.subplot(411)
    plt.plot(x, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    return residual
'''   
For a Time series to be stationary, its ADCF test should have:

   1. p-value to be low (according to the null hypothesis)
   2. The critical values at 1%,5%,10% confidence intervals should be as close as possible to the Test Statistics
'''
def adft(x , column ):
    print('Results of Dickey Fuller Test for '+ column + ' station')
    dftest = adfuller(x , autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    pvalue = dftest[1]
    cutoff = 0.01
    if pvalue < cutoff:
        print('p-value = %.4f. The series is likely stationary.' % pvalue)
    else:
        print('p-value = %.4f. The series is likely non-stationary.' % pvalue)
    print(dfoutput)


def roll_mean_std(x , col):
    roll_mean = x.rolling(window=161).mean()
    roll_std = x.rolling(window=161).std()
    #print(roll_mean,roll_std)
    orig = plt.plot(x, color='blue', label='Original')
    mean = plt.plot(roll_mean, color='red', label='Rolling Mean')
    std = plt.plot(roll_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation ' + col + ' station')
    plt.show()


#autocorrelation
def auto_correlation(x ,col):
    fig = plt.figure(figsize=(15,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(x, lags=40, ax=ax1) # 
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(x, lags=40, ax=ax2)# , lags=40
    plt.show()

# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    #acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 
            'corr':corr, 'minmax':minmax})

train = []
test = []
train_size = int(0.7 * len(df))
train = df[:train_size ] 
test = df[train_size:]

def arima(column):

    model = pm.auto_arima(train[column], start_p=1, start_q=1,
                            test='adf',       # use adftest to find optimal 'd'
                            max_p=3, max_q=3, # maximum p and q
                            m=6,              # frequency of series
                            d=None,           # let model determine 'd'
                            seasonal=True,   #  Seasonality
                            start_P=0, 
                            D=0, 
                            trace=True,
                            error_action='ignore',  
                            suppress_warnings=True, 
                            stepwise=True)


    model_fit = model.fit(train[column])
    y_pred = model_fit.predict(n_periods = len(test))
    print(y_pred)
    fc_series = pd.Series(y_pred, index=test[column].index)
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(train[column], label='training')
    plt.plot(test[column], label='actual')
    plt.plot(fc_series, label='forecast')
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
    fs = forecast_accuracy(y_pred , test[column])
    print(fs)


for i in col:
    print("PM2.5 Concenteration for " + i +" Station")
    sd(df[i] , i)
    adft(df[i] , i)
    roll_mean_std(df[i] , i)
    auto_correlation(df[i] , i)
    arima(i)

 