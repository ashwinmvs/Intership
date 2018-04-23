# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 18:21:33 2018

@author: ashwin.monpur
"""

import MySQLdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from dateutil.relativedelta import relativedelta

conn = MySQLdb.connect(host="localhost",passwd="pass@123",db="mf",user="root")
cursor = conn.cursor()
scheme_code=input("Please enter Mutual Fund Scheme ID: ")
query = pd.read_sql_query("select * from mf.daily_data_2006 where Scheme_Code="+scheme_code,conn)

nav_data=pd.to_numeric(query['Net_Asset_Value'])
query['Date']=pd.to_datetime(query['Date'])
nav_data.index=query['Date']

plt.plot(nav_data)

# return calculation 

#Return

print('\n Calculating Return')
range_return=input("Select the number to find the return: \n 1. Daily -- 1 \n 2. Monthly -- 2 \n 3. Quarterly -- 3 \n 4. Semiannually -- 4 \n 5. Yearly -- 5 \n Enter the Desired input:   ")
date_data=query['Date'].max()

if int(range_return)==1:
    daily=nav_data.asfreq('D','ffill')
    ret_d=daily.diff(1)
    fdat_d=date_data - timedelta(days=1)
    #fdat_m_f=format(scheme_code,fdat_m.strftime('%Y-%m-%d'))
    query_d = pd.read_sql_query("select Date,Net_Asset_Value from mf.daily_data_2006 where Scheme_Code={0} AND Date >= '{1}'".format(scheme_code,fdat_d.strftime('%Y-%m-%d')),conn)
    nav_data_d=pd.to_numeric(query_d['Net_Asset_Value'])
    ret_d_d=nav_data_d.diff(1)
    ret_d_d_v=ret_d_d.var()
    std_d_d=np.sqrt(ret_d_d_v)
    print(ret_d[(len(ret_d)-1)])
    print('Mutual Fund: '+scheme_code+"\n---"+query['Scheme_Name'][0]+"---")
    print(' Max Return: %f \n Min Return: %f \n volatility : %f ' % (ret_d_d.max(), ret_d_d.min(), std_d_d))
elif int(range_return)==2:
    monthly=nav_data.asfreq('M','ffill')
    ret_m=monthly.diff(1)
    fdat_m=date_data - timedelta(days=30)
    #fdat_m_f=format(scheme_code,fdat_m.strftime('%Y-%m-%d'))
    query_m = pd.read_sql_query("select Date,Net_Asset_Value from mf.daily_data_2006 where Scheme_Code={0} AND Date >= '{1}'".format(scheme_code,fdat_m.strftime('%Y-%m-%d')),conn)
    nav_data_m=pd.to_numeric(query_m['Net_Asset_Value'])
    ret_d_m=nav_data_m.diff(1)
    ret_d_m_v=ret_d_m.var()
    std_d_m=np.sqrt(ret_d_m_v)
    print(ret_m)
    print('Return of Mutual Fund: '+scheme_code+"---",query['Scheme_Name'][0])
    print(' Max Return: %f \n Min Return: %f \n volatility : %f ' % (ret_d_m.max(), ret_d_m.min(), std_d_m))
elif int(range_return)==3:
    quarterly=nav_data.asfreq('Q','ffill')
    ret_q=quarterly.diff(1)
    fdat_q=date_data - relativedelta(months=3)
    query_q = pd.read_sql_query("select Date,Net_Asset_Value from mf.daily_data_2006 where Scheme_Code={0} AND Date >= '{1}'".format(scheme_code,fdat_q.strftime('%Y-%m-%d')),conn)
    nav_data_q=pd.to_numeric(query_q['Net_Asset_Value'])
    ret_d_q=nav_data_q.diff(1)
    ret_d_q_v=ret_d_q.var()
    std_d_q=np.sqrt(ret_d_q_v)
    print(ret_q)
    print('Return of Mutual Fund: '+scheme_code+"---",query['Scheme_Name'][0])
    print(' Max Return: %f \n Min Return: %f \n volatility : %f ' % (ret_q.max(), ret_q.min(), std_d_q))
elif int(range_return)==4:
    semiannually=nav_data.asfreq('2Q','ffill')
    ret_sa=semiannually.diff(1)
    fdat_sa=date_data - relativedelta(months=6)
    query_sa = pd.read_sql_query("select Date,Net_Asset_Value from mf.daily_data_2006 where Scheme_Code={0} AND Date >= '{1}'".format(scheme_code,fdat_sa.strftime('%Y-%m-%d')),conn)
    nav_data_sa=pd.to_numeric(query_sa['Net_Asset_Value'])
    ret_d_sa=nav_data_sa.diff(1)
    ret_d_sa_v=ret_d_sa.var()
    std_d_sa=np.sqrt(ret_d_sa_v)
    print(ret_sa)
    print('Return of Mutual Fund: '+scheme_code+"---",query['Scheme_Name'][0])
    print(' Max Return: %f \n Min Return: %f \n volatility : %f ' % (ret_sa.max(), ret_sa.min(), std_d_sa))
elif int(range_return)==5:
    annually=nav_data.asfreq('A','ffill')
    ret_a=annually.diff(1)
    fdat_a=date_data - relativedelta(months=12)
    query_a = pd.read_sql_query("select Date,Net_Asset_Value from mf.daily_data_2006 where Scheme_Code={0} AND Date >= '{1}'".format(scheme_code,fdat_a.strftime('%Y-%m-%d')),conn)
    nav_data_a=pd.to_numeric(query_a['Net_Asset_Value'])
    ret_d_a=nav_data_sa.diff(1)
    ret_d_a_v=ret_d_sa.var()
    std_d_a=np.sqrt(ret_d_sa_v)
    print(ret_a)
    print('Return of Mutual Fund: '+scheme_code+"---",query['Scheme_Name'][0])
    print(' Max Return: %f \n Min Return: %f \n volatility : %f ' % (ret_a.max(), ret_a.min(), std_d_a))
else:
    print('Wrong input')
    






