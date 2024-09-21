import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd

shap.initjs()
data = pd.read_csv(r'data.csv',engine='python',encoding='gb18030',delimiter=",")
X=data.drop(['id','y'], axis=1)
Y=data['y']
# Add noisy features to make the problem harder

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=0)
ratio = float(np.sum(Y == 0)) / np.sum(Y == 1)
xg = XGBClassifier(scale_pos_weight=ratio,learning_rate=, max_depth=, n_estimators=)
xg=xg.fit(X_train,y_train)
plt.rcParams.update({'font.size': 16, 'font.family': 'Arial'})
shap_values = shap.TreeExplainer(xg).shap_values(X_test)
shap.summary_plot(shap_values,X_test,max_display=10)
shap.summary_plot(shap_values, X_test,plot_type="bar",max_display=10)#画条形图
plt.show()