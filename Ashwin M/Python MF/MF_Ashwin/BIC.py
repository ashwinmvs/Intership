# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 10:51:48 2018

@author: ashwin.monpur
"""

# geting BIC value of the models


import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA 
import itertools

data=pd.read_csv('MFC_3.csv')

ts_nav=data['nav']

date_nav = pd.to_datetime(data['date']) #dtype: datetime64[ns] Convert argument to datetime.
ts_nav.index=date_nav
ts_log=np.log(ts_nav)

model = ARIMA(ts_log, order=(2,2,2))
result_ARIMA = model.fit(disp=0)
print(result_ARIMA.aic)

d=range(0,7)
p=q=range(0,7)

combinations = list(itertools.product(p, d, q))
dic_model={}
for params in combinations:
    try:
        t_model=ARIMA(ts_log, order=params)
        result_ARIMA=t_model.fit()
        key=params
        dic_model.setdefault(key,[result_ARIMA.aic])
    except:
        continue

print(max(dic_model))
