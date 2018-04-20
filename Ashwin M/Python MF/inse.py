# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 18:21:33 2018

@author: ashwin.monpur
"""

import MySQLdb as my
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from sqlalchemy import create_engine

conn = my.connect(host="localhost",passwd="pass@123",db="mf",user="root")
cursor = conn.cursor()
scheme_code=(pd.read_sql_query("select distinct Scheme_Code from mf.daily_data_2006",conn))

scheme_code_list=scheme_code.values.T.flatten()
scheme_code_list[2]


return_mf_list=[]

risk_mf_list=[]

for s in range(1,len(scheme_code_list)): #len(scheme_code_list)):
   
    print(scheme_code_list[s])
    
    return_sm_list=[]
    risk_sm_list=[]
    
    query = pd.read_sql_query("select * from mf.daily_data_2006 where Scheme_Code="+scheme_code_list[s],conn)
    
    return_sm_list.append(scheme_code_list[s])
    return_sm_list.append(str(query['Scheme_Name'][0]))    
    
    risk_sm_list.append(scheme_code_list[s])
    risk_sm_list.append(str(query['Scheme_Name'][0]))
    
    nav_data=pd.to_numeric(query['Net_Asset_Value'])
    query['Date']=pd.to_datetime(query['Date'])
    nav_data.index=query['Date']    
    date_data=query['Date'].max()
    #Daily Return
    
    
#    daily=nav_data.asfreq('D','ffill')
#    
#    ret_d=daily.diff(1)
#    
#    fdat_d=date_data - timedelta(days=1)    
#    
#    query_d = pd.read_sql_query("select Date,Net_Asset_Value from mf.daily_data_2006 where Scheme_Code={0} AND Date >= '{1}'".format(scheme_code_list[s],fdat_d.strftime('%Y-%m-%d')),conn)
#    
#    nav_data_d=pd.to_numeric(query_d['Net_Asset_Value'])
#    
#    ret_d_d=nav_data_d.diff(1)
#    ret_d_d_v=ret_d_d.var()
#    std_d_d=np.sqrt(ret_d_d_v)
#    
#    return_sm_list.append(round(ret_d[(len(ret_d)-1)],4))
#    return_sm_list.append(fdat_d.strftime('%Y-%m-%d'))
#    
#    risk_sm_list.append(round(std_d_d,4))
#    risk_sm_list.append(fdat_d.strftime('%Y-%m-%d'))
    
    # Monthly Return
    
    monthly=nav_data.asfreq('M','ffill')
    ret_m=monthly.diff(1)
    fdat_m=date_data - timedelta(days=30)

    query_m = pd.read_sql_query("select Date,Net_Asset_Value from mf.daily_data_2006 where Scheme_Code={0} AND Date >= '{1}'".format(scheme_code_list[s],fdat_m.strftime('%Y-%m-%d')),conn)
    nav_data_m=pd.to_numeric(query_m['Net_Asset_Value'])
    ret_d_m=nav_data_m.diff(1)
    ret_d_m_v=ret_d_m.var()
    std_d_m=np.sqrt(ret_d_m_v)
    
    return_sm_list.append(round(ret_m[(len(ret_m)-1)],4))
    return_sm_list.append(fdat_m.strftime('%Y-%m-%d'))
    
    risk_sm_list.append(round(std_d_m,4))
    risk_sm_list.append(fdat_m.strftime('%Y-%m-%d'))
    
    # Quarterly Return
    
    quarterly=nav_data.asfreq('Q','ffill')
    ret_q=quarterly.diff(1)
    fdat_q=date_data - relativedelta(months=3)
    query_q = pd.read_sql_query("select Date,Net_Asset_Value from mf.daily_data_2006 where Scheme_Code={0} AND Date >= '{1}'".format(scheme_code_list[s],fdat_q.strftime('%Y-%m-%d')),conn)
    nav_data_q=pd.to_numeric(query_q['Net_Asset_Value'])
    ret_d_q=nav_data_q.diff(1)
    ret_d_q_v=ret_d_q.var()
    std_d_q=np.sqrt(ret_d_q_v)
    
    return_sm_list.append(round(ret_q[(len(ret_q)-1)],4))
    return_sm_list.append(fdat_q.strftime('%Y-%m-%d'))
    
    risk_sm_list.append(round(std_d_q,4))
    risk_sm_list.append(fdat_q.strftime('%Y-%m-%d'))
    
    # SemiAnnual Return
    
    semiannually=nav_data.asfreq('2Q','ffill')
    ret_sa=semiannually.diff(1)
    fdat_sa=date_data - relativedelta(months=6)
    query_sa = pd.read_sql_query("select Date,Net_Asset_Value from mf.daily_data_2006 where Scheme_Code={0} AND Date >= '{1}'".format(scheme_code_list[s],fdat_sa.strftime('%Y-%m-%d')),conn)
    nav_data_sa=pd.to_numeric(query_sa['Net_Asset_Value'])
    ret_d_sa=nav_data_sa.diff(1)
    ret_d_sa_v=ret_d_sa.var()
    std_d_sa=np.sqrt(ret_d_sa_v)
    
    return_sm_list.append(round(ret_sa[(len(ret_sa)-1)],4))
    return_sm_list.append(fdat_sa.strftime('%Y-%m-%d'))
    
    risk_sm_list.append(round(std_d_sa,4))
    risk_sm_list.append(fdat_sa.strftime('%Y-%m-%d'))
    
    # Annual Return
    
    annually=nav_data.asfreq('A','ffill')
    ret_a=annually.diff(1)
    fdat_a=date_data - relativedelta(months=12)
    query_a = pd.read_sql_query("select Date,Net_Asset_Value from mf.daily_data_2006 where Scheme_Code={0} AND Date >= '{1}'".format(scheme_code_list[s],fdat_a.strftime('%Y-%m-%d')),conn)
    nav_data_a=pd.to_numeric(query_a['Net_Asset_Value'])
    ret_d_a=nav_data_sa.diff(1)
    ret_d_a_v=ret_d_sa.var()
    std_d_a=np.sqrt(ret_d_sa_v)
    
    return_sm_list.append(round(ret_a[(len(ret_a)-1)],4))
    return_sm_list.append(fdat_a.strftime('%Y-%m-%d'))
    
    risk_sm_list.append(round(std_d_a,4))
    risk_sm_list.append(fdat_a.strftime('%Y-%m-%d'))
    
    return_sm_list.append(date_data.strftime('%Y-%m-%d'))
    risk_sm_list.append(date_data.strftime('%Y-%m-%d'))
    
    return_mf_list.append(return_sm_list)
    risk_mf_list.append(risk_sm_list)
    
    
    
    
data_return=pd.DataFrame(return_mf_list) 
data_risk=pd.DataFrame(risk_mf_list) 
data_return.columns=['Scheme_Code','Scheme_Name','Return_Monthly','Monthly_Date','Return_Quarterly','Quarterly_Date','Return_SemiAnnual','SemiAnnual_Date','Return_Annual','Annual_Date','Recent_Date']  
data_risk.columns=['Scheme_Code','Scheme_Name','Risk_Monthly','Monthly_Date','Risk_Quarterly','Quarterly_Date','Risk_SemiAnnual','SemiAnnual_Date','Risk_Annual','Annual_Date','Recent_Date']  

engine = create_engine('mysql://root:pass@123@localhost:3306/mf', echo = False)
data_return.to_sql(name = 'test_return', con = engine, if_exists = 'replace', index = True)
data_risk.to_sql(name = 'test_risk', con = engine, if_exists = 'replace', index = True)    


#data_return.to_sql(name='test',con=conn,if_exists='replace', index=True,)