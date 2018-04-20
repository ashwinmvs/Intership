# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 09:58:02 2018

@author: ashwin.monpur
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA



data=pd.read_csv('MFC_3.csv')

ts_nav=data['nav']

date_nav = pd.to_datetime(data['date']) #dtype: datetime64[ns] Convert argument to datetime.
ts_nav.index=date_nav
#plt.plot(ts_nav)

# Minimizing the variance

ts_log=np.log(ts_nav)
#plt.plot(ts_log)  #varaince of the data is minimized

# making the data stationary using 1st order differencing

ts_logdff= ts_log-ts_log.shift()
ts_logdff.dropna(inplace=True) #removing NaN values 
#plt.plot(ts_logdff)

# According to the plot, seasonality exists so 2nd order differencing is needed

ts_log2dff= ts_logdff-ts_logdff.shift()
ts_log2dff.dropna(inplace=True)
#plt.plot(ts_log2dff)

#Checking seasonal & differencing for stationarity 
# Dickey-Fuller (ADF) test

def test_stationarity(timeseries):

#Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC') 
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
test_stationarity(ts_log2dff)

#Results of Dickey-Fuller Test:
#Test Statistic                  -18.992132
#p-value                           0.000000
##Lags Used                        3.000000
#Number of Observations Used    1284.000000
#Critical Value (1%)              -3.435453
#Critical Value (5%)              -2.863794
#Critical Value (10%)             -2.567970
#dtype: float64

# test_stat < critical 1% 
# Reject null hypothesis and data is stationary with 99 % confidence interval 

# p - value is also les than 5% no further differencing required

# ploting ACF and PACF

lag_acf = acf(ts_logdff, nlags=30)
lag_pacf = pacf(ts_logdff, nlags=50, method='ols')

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_logdff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_logdff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log2dff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log2dff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot
pyplot.figure()
pyplot.subplot(211)
plot_acf(ts_logdff, ax=pyplot.gca(),lags=40)
pyplot.subplot(212)
plot_pacf(ts_logdff, ax=pyplot.gca(), lags=50)
pyplot.show()

# Model 

model = ARIMA(ts_log, order=(2,1,1))
result_ARIMA = model.fit(disp=0)
#plt.plot(result_ARIMA.fittedvalues)
#print(result_ARIMA.summary())
forecast = result_ARIMA.forecast(4)

f_prediction = forecast[0]
f_prediction_data=np.exp(f_prediction)
dates=pd.date_range('2015-08-12', periods=4)

data_test=pd.read_csv('test_MFC.csv')

test = data_test['nav']
test_date = pd.to_datetime(data_test['date'])
test.index=test_date


data=pd.read_csv('MFC_3.csv')

ts_nav=data['nav']

date_nav = pd.to_datetime(data['date']) #dtype: datetime64[ns] Convert argument to datetime.
ts_nav.index=date_nav


confid_int=forecast[2]
c_1=[]
c_2=[]
for i in range(len(confid_int)):
    c_1.append(confid_int[i][0])
    c_2.append(confid_int[i][1])

c_1_exp=np.exp(c_1)
c_2_exp=np.exp(c_2)

#
#plt.subplot(3,2,1)
#plt.figure(figsize=(20,20))
#plt.plot(test)
#plt.plot(dates,f_prediction_data, linestyle='-.',color='red')
#plt.plot(ts_nav)
#plt.plot(dates,c_1_exp, linestyle='--',color='gray')
#plt.plot(dates,c_2_exp, linestyle='--',color='gray')
#
#
#plt.subplot(3,2,2)
#plt.figure(figsize=(5,5))
#plt.plot(dates,f_prediction_data, linestyle='-.',color='red')
#plt.plot(test['2015-08-12':'2015-08-17'])






















#plt.plot(ts_log)
#plt.plot(result_ARIMA.fittedvalues)


predictions_ARIMA_diff = pd.Series(result_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())
#plt.plot(predictions_ARIMA_diff)
#
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())
#plt.plot(predictions_ARIMA_diff_cumsum)

#predictions_ARIMA_diff_cumsum_2 = predictions_ARIMA_diff_cumsum.cumsum()
#print(predictions_ARIMA_diff_cumsum_2.head())

predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()
#plt.plot(predictions_ARIMA_log)

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts_nav)
plt.plot(predictions_ARIMA)

#forecast = result_ARIMA.forecast(steps =100)
#predicted_list = np.exp(forecast[0])
#
#plt.plot(predicted_list)
 

#
#monthly_mean = ts_nav.resample('W',loffset=pd.offsets.timedelta(days=-6))
#print(monthly_mean.values)
##monthly_mean
##print(monthly_mean.head(13))
##monthly_mean.plot()















