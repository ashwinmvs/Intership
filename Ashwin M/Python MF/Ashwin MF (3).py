# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:07:44 2018

@author: rajiv18711
"""

# MF Risk
import math
import openpyxl
import numpy as np


# Get Mutual fund data 
mu_fund = openpyxl.load_workbook('MF.xlsx')
MF_sheet=mu_fund.get_sheet_by_name('MFFranklin')
list_mf_value=[i.value for i in MF_sheet['N']]
count=len(list_mf_value)

# calculating Return 
s=0
ret=[]
for j in range(2,count):
    valr=((float(list_mf_value[j])-float(list_mf_value[j-1]))/float(list_mf_value[j-1]))*100
    x=ret.append(valr)
    
#calculating variance and standard deviation
sum_ret=sum(ret)
mean=sum_ret/len(ret)

var_ret=0
for k in range(1,len(ret)):
    var_ret=var_ret+(float(ret[k])-mean)**2

var=var_ret/(len(ret)-1)
sd_ret=math.sqrt(var)

# Get Index Data
indx_fund = openpyxl.load_workbook('MF.xlsx')
indx_sheet=indx_fund.get_sheet_by_name('IMF')
list_indx_open=[b.value for b in indx_sheet['B']]
list_indx_close=[c.value for c in indx_sheet['C']]
length_open=len(list_indx_open)
length_close=len(list_indx_close)
avg_indx_list=[]
ret_indx_list=[]
for l in range(1,length_open):
    avg_indx=(float(list_indx_open[l]+float(list_indx_close[l])))/2
    avg_indx_list.append(avg_indx)

for m in range(1,len(avg_indx_list)):
    ret_indx=((avg_indx_list[m]-avg_indx_list[m-1])/avg_indx_list[m-1])*100
    ret_indx_list.append(ret_indx)
sum_indx_ret=sum(ret_indx_list)
mean_indx=sum_indx_ret/len(ret_indx_list)
print(mean_indx)

#calculate variance and Standard deviation of index

var_indx_ret=0
for n in range(1,len(ret_indx_list)):
    var_indx_ret=var_indx_ret+(float(ret_indx_list[n])-mean_indx)**2
    
var_indx=var_indx_ret/(len(ret_indx_list)-1)
sd_indx=math.sqrt(var_indx)

cova=np.cov(ret,ret_indx_list)
correlat= cova/(sd_indx*sd_ret)

print(np.corrcoef(ret,ret_indx_list))







    






    
    



 
    
    
    


# =============================================================================
# for k in range(count):
#     lis1=list_mf_value[k]
#     print(lis1)
#     
# for j in range(count):
#     lis2=list_mf_value[j]
#     print(lis2)
# print(k)
# print(j)        
# =============================================================================
    
#j=0
#for nav in MF_sheet['X']:
      #print(i.value)
      #j=j+1
      #ret=nav.value
      #MF_sheet.col_slicer(colx=X, start_rowx=2,end_rowx=j)
      #print(ret)
#data=MF_sheet.col_slicer(colx='X', start_rowx=2,end_rowx=j)      
#print ('number of data points in sheet %d' %j)

