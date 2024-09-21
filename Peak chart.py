import joypy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
data = pd.read_csv(r'data.csv',engine='python',encoding='gb18030',delimiter=",")
#根据"Name"分组，每个Name是一行"脊"，其中有多个，默认y轴一致
fig, axes = joypy.joyplot(data, by="y", column='GLU',ylabelsize=16)
plt.rcParams.update({'font.size': 16})
plt.rcParams['font.family'] = 'sans-serif'  # 设置字体家族
plt.rcParams['font.sans-serif'] = ['Arial']
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('GLU (mmol/L)', fontsize=16)
plt.subplots_adjust(bottom=0.2)
plt.show()