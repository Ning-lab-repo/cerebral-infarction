import pandas as pd
data=pd.read_csv(r'data.csv',low_memory=False,encoding='gb18030')
#删除行缺失值超过20%
def del_rows(data):
    t = int(0.8*data.shape[1])
    data = data.dropna(thresh=t)#保留至少有 t 个非空的行
    return data
data1=del_rows(data)
data1.to_csv(r'data.csv',encoding='utf_8_sig')

#删除列缺失值超过20%的
def remcolumns(data):
    t = int(0.8*data.shape[0])
    data = data.dropna(thresh=t,axis=1)#保留至少有 t 个非空的列
    return data
data1=remcolumns(data)
data1.to_csv(r'data.csv',encoding='utf_8_sig')

#用KNN填充缺失值
import pandas as pd
from sklearn.impute import KNNImputer
data=pd.read_csv(r'data.csv',engine='python',encoding='gb18030')
# 使用KNNImputer
KI= KNNImputer(n_neighbors=3)
df=KI.fit_transform(data)
df1=pd.DataFrame(df)
df1.to_csv(r'data.csv',encoding='utf_8_sig')
