# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 13:11:06 2018

@author: ashwin.monpur
"""
#import packages



from pylab import rcParams
from statsmodels.tsa.stattools import acf, pacf
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
rcParams['figure.figsize'] = 15, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA

#Get Data from CSV

data=pd.read_csv('MFC.csv')
data['date']=pd.to_datetime(data['date'])

#dateparse=lambda dates: pd.datetime.strptime(dates, '%M/%d/%Y')
#data = pd.read_csv('MFC.csv', parse_dates=['date_new'], index_col='date_new',date_parser=dateparse)

t_series=data['nav']
t_series.index=data['date']

#print(ts)

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=55) #Cont Mean
    rolstd = pd.rolling_std(timeseries, window=55)  #Cont Std
   
    #Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC') 
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

test_stationarity(t_series)
t_series_log=np.log(t_series)

#moving_avg = pd.rolling_mean(t_series_log,55)
#ts_log_moving_avg_diff = t_series_log - moving_avg
#ts_log_moving_avg_diff.head(55)
#ts_log_moving_avg_diff.dropna(inplace=True)
#test_stationarity(ts_log_moving_avg_diff)
#test_stationarity(t_series_log)

# check for seasonality and trend

# If seasonality and trend are present : eliminationg 
# Differencing removing trend

ts_log_diff = t_series_log - t_series_log.shift()

plt.plot(ts_log_diff)

ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

# Plots of ACF and PACF

lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

y1=-1.96/np.sqrt(len(ts_log_diff))
y2=1.96/np.sqrt(len(ts_log_diff))
y0=0
#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y0)
plt.axhline(y1)
plt.axhline(y2)
plt.title('ACF')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y0)
plt.axhline(y1)
plt.axhline(y2)
plt.title('PACF')
plt.tight_layout()
   
# ARIMA Model

model= ARIMA(t_series, order=(1,2,1))
model_fit=model.fit(disp=0)
print(model_fit.summary())
fit = model_fit.forecast(steps =100)   
   










