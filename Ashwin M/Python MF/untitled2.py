# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 18:54:39 2018

@author: ashwin.monpur
"""


print(model_fit.summary())
 
forecast=fit[0]
f_p_ARIMA_diff=pd.series(forecast.fittedvalues, copy=True)

##stan_err=fit[1]
con_int_data=fit[2]
d=len(t_series)
last_date=str(t_series.index[d-1])
past_date = datetime.datetime.strptime(last_date[0:10], "%Y-%m-%d")

end_date = past_date + datetime.timedelta(days=100)

start_date= past_date + datetime.timedelta(days=1)
d_f_list=[]
for d in range(len(forecast)):
    f_dates = past_date + datetime.timedelta(days=d)
    d_f_list.append(f_dates)

plt.subplot(1,1,1)
plt.plot(ts_log_diff)
plt.plot(model_fit.fittedvalues, color='red')
plt.plot(t_series)
plt.subplot(1,2,1)

plt.plot(t_series)
plt.plot(d_f_list,forecast)