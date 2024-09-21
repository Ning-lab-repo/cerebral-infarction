import matplotlib.pyplot as plt
import pandas as pd
# 数据
data=pd.read_csv(r'data.csv',engine='python',encoding='gb18030')
categories = data.疾病
values1 = data.男
values2 = data.女
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif']='Arial'
fig, ax = plt.subplots()
# 绘制第一个柱状图
bar_width = 0.6
ax.barh(categories, values1, height=bar_width,color='lightblue', label='Male')
# 绘制第二个柱状图，基于第一个柱状图的值上叠加
ax.barh(categories, values2, height=bar_width,left=values1, color='gainsboro', label='Female')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('Number of samples',fontsize=20,pad=30,loc='center')
plt.legend(loc='upper center', ncol=2,fontsize=16,bbox_to_anchor=(0.5, 1.10),frameon=False)
plt.subplots_adjust(left=0.2)
plt.show()
