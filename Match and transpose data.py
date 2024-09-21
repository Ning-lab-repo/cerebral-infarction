#匹配两个文件夹数据
import pandas as pd
import os
os.getcwd()
df1 = pd.read_csv(r"data.csv",engine = 'python',encoding='gb18030')
df2 = pd.read_csv(r"data.csv",engine = 'python')
data3=pd.merge(df1,df2,on=["id"])
data3.to_csv(r'data.csv',encoding='utf_8_sig')

#删除重复数据
import pandas as pd
frame=pd.read_csv(r'data.csv',engine='python',encoding='gb18030')
data = frame.drop_duplicates(subset=["id"], keep='first', inplace=False)  #id有重复的只保留第一个
data.to_csv(r'data.csv',encoding='utf_8_sig')


#多行转换为多列
import pandas as pd
import numpy as np
data=pd.read_csv(r'data.csv',low_memory=False,encoding='gb18030')
table=pd.pivot_table(data,index=[u'id'],columns=[u'检查代码'],values=[u'检查结果'])
table.to_csv(r'data.csv',encoding='utf_8_sig')

