# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 10:42:55 2018

@author: ashwin.monpur
"""

#118531

import pandas as pd
import MySQLdb
import numpy as np

# Importing mutual Fund Data from DB

conn=MySQLdb.connect(host="localhost",passwd="pass@123",db="mf",user="root")
MF_Data=pd.read_sql_query("select * from mf.daily_data_2017 where Scheme_Code=118531",conn)


# Gathering Required Data

nav=pd.to_numeric(MF_Data['Net_Asset_Value'])
nav.index=pd.to_datetime(MF_Data['Date'])

# Get index Data from csv
index_data = pd.read_csv('nif50ty.csv')

index_open=index_data['Open']
index_close=index_data['Close']
index_avg=(index_open+index_close)/2
index_avg.index=pd.to_datetime(index_data['Date'])

list_av_mf=[]
list_match_indx=[]
#list_mf_d_date=[]
#list_indx_d_date=[]
for i in pd.to_datetime(index_data['Date']):
    for j in pd.to_datetime(MF_Data['Date']):
        list_mf=[]
        list_indx=[]
        list_mf_d=[]
        list_indx_d=[]
        if i==j:
            list_mf.append(j)
            list_mf.append(nav[j])
            list_indx.append(i)
            list_indx.append(index_avg[i])
            
            list_av_mf.append(list_mf)
            list_match_indx.append(list_indx)
#            list_mf_d_date.append(list_mf_d)
#            list_indx_d_date.append(list_indx_d)
#            print('index: {0} {1}'.format(i,index_ret[i]))
#            print('MF   : {0} {1}'.format(j,ret_d[j]))
        else:
            continue


nav_spd_data=pd.DataFrame(list_av_mf)
indx_spd_data=pd.DataFrame(list_match_indx)

nav_spd_data.columns=['Date','NAV']
indx_spd_data.columns=['Date','Average']
nav_spd_data.index=nav_spd_data['Date']
indx_spd_data.index=indx_spd_data['Date']

data_nav=nav_spd_data['NAV']
data_indx=indx_spd_data['Average']

# MF Return,var,std
ret_mf=pd.to_numeric(data_nav.diff(1))
ret_mf.dropna(inplace=True)
mean_mf=ret_mf.mean()
var_mf=ret_mf.var()
std_mf=np.sqrt(var_mf)

# Index return,var,std
ret_indx=pd.to_numeric(data_indx.diff(1))
ret_indx.dropna(inplace=True)
mean_indx=ret_indx.mean()
var_indx=ret_indx.var()
std_indx=np.sqrt(var_indx)

# R square

corr=np.corrcoef(ret_indx.tolist(),ret_mf.tolist())
R_square=np.square(corr[0][1])

# Beta

beta= (std_mf*R_square)/std_indx

# Alpha
risk_free_rate=0.069
alpha = (mean_mf-risk_free_rate)-beta*(mean_indx-risk_free_rate)

# Active Risk

a_r=np.std((ret_mf-ret_indx)/ret_indx)

# Sharpe ratio

sharp_ratio=(mean_mf-risk_free_rate)/std_mf

# Treynor ratio

treynor_r=(mean_mf-risk_free_rate)/beta

# Information Ratio

i_r=(mean_mf-mean_indx)/a_r







