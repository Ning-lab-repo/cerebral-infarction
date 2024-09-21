import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc  ###计算roc和auc
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import numpy as np
data = pd.read_csv(r'data.csv',engine='python',encoding='gb18030',delimiter=",")
X=data.drop(['住院号','y'], axis=1)
Y=data['y']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=0)
transfer=StandardScaler()
X_train=transfer.fit_transform(X_train)
X_test=transfer.transform(X_test)
ratio = float(np.sum(Y == 0)) / np.sum(Y == 1)
classifier = XGBClassifier(scale_pos_weight=ratio,learning_rate=, max_depth=, n_estimators=) #最佳参数
y_score=classifier.fit(X_train, y_train).predict_proba(X_test)
fpr,tpr,threshold = roc_curve(y_test, y_score[:,1]) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
plt.figure()
plt.rcParams['font.sans-serif'] = 'Arial'
plt.plot(fpr, tpr, color='darkorange',
          label='ROC curve (AUC = %0.4f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.legend(loc='lower right', fontsize=12)
plt.title('ROC',fontsize=20,pad=10,loc='center')
plt.subplots_adjust(bottom=0.2)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.ylabel('True positive rate', fontsize=16)
plt.xlabel('False positive rate', fontsize=16)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.show()
