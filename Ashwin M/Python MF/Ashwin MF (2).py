# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:07:44 2018

@author: rajiv18711
"""

# MF Risk

import openpyxl
mu_fund = openpyxl.load_workbook('MF.xlsx')
MF_sheet=mu_fund.get_sheet_by_name('MFFranklin')
list_mf_value=[i.value for i in MF_sheet['X']]
count=len(list_mf_value)
# calculating Return 
s=0
ret=[]
for j in range(2,count):
    valr=((float(list_mf_value[j])-float(list_mf_value[j-1]))/float(list_mf_value[j-1]))
    x=ret.append(valr)
    
 
#calculating variance
sum_ret=sum(ret)
print(sum_ret)
mean=sum_ret/len(ret)
print(mean)
for k in range(1,j):
    s_var=sum(ret)
    



 
    
    
    


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

