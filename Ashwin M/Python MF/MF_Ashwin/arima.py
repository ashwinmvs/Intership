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

import seaborn; seaborn.set()
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot


#Get Data from CSV

data=pd.read_csv('MFC.csv')
data['date']=pd.to_datetime(data['date'])

#dateparse=lambda dates: pd.datetime.strptime(dates, '%M/%d/%Y')
#data = pd.read_csv('MFC.csv', parse_dates=['date_new'], index_col='date_new',date_parser=dateparse)

t_series=data['nav']
t_series.index=data['date']
np.sort(t_series)

#print(ts)

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=30) #Cont Mean
    rolstd = pd.rolling_std(timeseries, window=30)  #Cont Std
   
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

# Transforming the data

ts_logtransformed=(np.log(t_series))

plt.plot(ts_logtransformed)
ts_logtransformed.head(1289)

# Rolling average smoothing
rolling_average = ts_logtransformed.rolling(window = 7, center= False).mean()
plt.plot(ts_logtransformed, label = 'Log Transformed')
plt.plot(rolling_average, color = 'red', label = 'Rolling Average')
rolling_average.head(10) 

log_Rolling_difference = ts_logtransformed - rolling_average
#log_Rolling_difference.head(10)
#log_Rolling_difference.tail(10)

log_Rolling_difference.dropna(inplace=True)
plt.plot(log_Rolling_difference)

test_stationarity(log_Rolling_difference)

# Seasonality Adjustment

ts_diff_logtrans = ts_logtransformed-ts_logtransformed.shift(7)
plt.plot(ts_diff_logtrans)
ts_diff_logtrans.head(10)


ts_diff_logtrans.dropna(inplace=True)
plt.plot(ts_diff_logtrans)
ts_diff_logtrans.head(10)

# Decomposing

#decomposition = seasonal_decompose(ts_logtransformed)
#
#trend = decomposition.trend
#seasonal = decomposition.seasonal
#residual = decomposition.resid

#ACF and PACF plots:
lag_acf = acf(ts_diff_logtrans, nlags=30)
lag_pacf = pacf(ts_diff_logtrans, nlags=50, method='ols')

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_diff_logtrans)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_diff_logtrans)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')


#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_diff_logtrans)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_diff_logtrans)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


pyplot.figure()
pyplot.subplot(211)
plot_acf(ts_diff_logtrans, ax=pyplot.gca(),lags=40)
pyplot.subplot(212)
plot_pacf(ts_diff_logtrans, ax=pyplot.gca(), lags=50)
pyplot.show()


ts_diff_logtrans = ts_diff_logtrans.fillna(0)

# ARIMA

model = ARIMA(ts_logtransformed, order=(2, 2, 7))  

# Prediction

results_ARIMA = model.fit(trend= 'nc', disp=-1)  
plt.plot(ts_diff_logtrans)
plt.plot(results_ARIMA.fittedvalues, color='red', label = 'p =8, q =18')


RSS =results_ARIMA.fittedvalues-ts_diff_logtrans
RSS.dropna(inplace=True)
#plt.title('RSS: %.4f'% sum(RSS**2))
#plt.legend(loc='best')


#plt.plot(ts_logtransformed, label = 'log_tranfromed_data')
#plt.plot(results_ARIMA.resid, color ='green',label= 'Residuals')
#plt.title('ARIMA Model Residual plot')
#plt.legend(loc = 'best')

plt.subplot(1,1,1)
results_ARIMA.resid.plot(kind='kde')
#plt.title('Density plot of the residual error values')
#print(results_ARIMA.resid.describe())
#
#
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
plt.plot(predictions_ARIMA_diff)
print(predictions_ARIMA_diff.head())

predictions_ARIMA_diff_cumsum = (predictions_ARIMA_diff.cumsum())
plt.plot(predictions_ARIMA_diff_cumsum)
print(predictions_ARIMA_diff_cumsum.head(1289))
#
#
predictions_ARIMA_log = pd.Series(ts_logtransformed.iloc[0], index=ts_logtransformed.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head(1289)
plt.plot(predictions_ARIMA_log)
#
#

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(t_series)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-data)**2)/len(data)))











#
#lag_acf = acf(ts_diff_logtrans, nlags=20)
#lag_pacf = pacf(ts_diff_logtrans, nlags=20, method='ols')
#
#y1=-1.96/np.sqrt(len(ts_diff_logtrans))
#y2=1.96/np.sqrt(len(ts_diff_logtrans))
#y0=0
##Plot ACF: 
#plt.subplot(121) 
#plt.plot(lag_acf)
#plt.axhline(y0)
#plt.axhline(y1)
#plt.axhline(y2)
#plt.title('ACF')
#
##Plot PACF:
#plt.subplot(122)
#plt.plot(lag_pacf)
#plt.axhline(y0)
#plt.axhline(y1)
#plt.axhline(y2)
#plt.title('PACF')
#plt.tight_layout()


#ts_log_diff = t_series_log - t_series_log.shift()
#
#plt.plot(ts_log_diff)
#
#ts_log_diff.dropna(inplace=True)
#test_stationarity(ts_log_diff)
#

#   
## ARIMA Model
#
#model= ARIMA(t_series_log, order=(1,2,1))
#model_fit=model.fit(disp=0)
#
#fit = model_fit.forecast(steps =100) 
#plt.subplot(1,1,1)
#plt.plot(ts_log_diff)
#plt.plot(model_fit.fittedvalues, color='red')
#
#predictions_ARIMA_diff = pd.Series(model_fit.fittedvalues, copy=True)
##print (predictions_ARIMA_diff.head())
#
#predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
##print (predictions_ARIMA_diff_cumsum.head())
#
#predictions_ARIMA_log = pd.Series(t_series_log.ix[0], index=t_series_log.index)
#predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
#predictions_ARIMA_log.head()
#
#predictions_ARIMA = np.exp(predictions_ARIMA_log)
#plt.plot(t_series)
#plt.plot(predictions_ARIMA)
#
#print(model_fit.summary())
  






























#moving_avg = pd.rolling_mean(t_series_log,55)
#ts_log_moving_avg_diff = t_series_log - moving_avg
#ts_log_moving_avg_diff.head(55)
#ts_log_moving_avg_diff.dropna(inplace=True)
#test_stationarity(ts_log_moving_avg_diff)
#test_stationarity(t_series_log)

# check for seasonality and trend

# If seasonality and trend are present : eliminationg 
# Differencing removing trend 









#forecast=fit[0]
##stan_err=fit[1]
#con_int_data=fit[2]
#
#len_data_forecast=len(forecast)
#new_frcst=[i for i in t_series]
#for i in range(len_data_forecast):
#    new_frcst.append(forecast[i-1])
#    
#forc_val_log=np.exp(forecast)
#    
#plt.subplot(1,1,1)
#plt.plot(t_series)
  

#plt.subplot(1,1,1)    
##plt.plot(ts_log_diff)
##plt.plot(model_fit.fittedvalues, color='red')
##plt.title('RSS: %.4f'% sum((model_fit.fittedvalues-ts_log_diff)**2))
#predictions_ARIMA_diff = pd.Series(model_fit.fittedvalues, copy=True)
#print (predictions_ARIMA_diff.head())
#predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
#print (predictions_ARIMA_diff_cumsum.head())
#predictions_ARIMA_log = pd.Series(t_series_log.ix[0], index=t_series_log.index)
#predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
#predictions_ARIMA_log.head()
#predictions_ARIMA = np.exp(predictions_ARIMA_log)
#plt.plot(t_series)
#plt.plot(predictions_ARIMA)
#plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-t_series))/len(t_series)))






