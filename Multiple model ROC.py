import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
data = pd.read_csv(r'data.csv',engine='python',encoding='gb18030',delimiter=",")
X=data.drop(['id','y'], axis=1)
Y=data['y']
# Add noisy features to make the problem harder

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=0)
transfer=StandardScaler()
X_train=transfer.fit_transform(X_train)
X_test=transfer.transform(X_test)
sv=SVC(class_weight='balanced',C=,kernel=)
sv.fit(X_train,y_train)
probas_1=sv.decision_function(X_test)
lg=LogisticRegression(class_weight='balanced',C=,penalty=,max_iter=)
lg.fit(X_train,y_train)
probas_2=lg.predict_proba(X_test)[:, 1]
rf=RandomForestClassifier(class_weight='balanced',max_depth=,min_samples_leaf=)#
rf.fit(X_train,y_train)
probas_3=rf.predict_proba(X_test)[:, 1]
ratio = float(np.sum(Y == 0)) / np.sum(Y == 1)
xg = XGBClassifier(scale_pos_weight=ratio,learning_rate=, max_depth=, n_estimators=)#
xg.fit(X_train,y_train)
probas_4=xg.predict_proba(X_test)[:, 1]
model = keras.models.Sequential()
model.add(keras.layers.Dense(100,activation="selu"))  # 输出为100的全连接层
#model.add(keras.layers.Dense(50,activation="selu"))
model.add(keras.layers.Dense(1, activation = "sigmoid"))
model.compile(loss = "binary_crossentropy",optimizer = keras.optimizers.SGD(0.05),    # 优化函数为随机梯度下降 ，学习率为0.01
             metrics = ["accuracy"])                     # 优化指标为准确度
model.fit(X_train, y_train,epochs=10)  # 验证集
y_score=model.predict(X_test)
probas_5=model.predict(X_test).flatten()
fpr1, tpr1, _ = roc_curve(y_test, probas_1)
roc_auc1 = auc(fpr1, tpr1)
fpr2, tpr2, _ = roc_curve(y_test, probas_2)
roc_auc2 = auc(fpr2, tpr2)
fpr3, tpr3, _ = roc_curve(y_test, probas_3)
roc_auc3 = auc(fpr3, tpr3)
fpr4, tpr4, _ = roc_curve(y_test, probas_4)
roc_auc4 = auc(fpr4, tpr4)
fpr5, tpr5, _ = roc_curve(y_test, probas_5)
roc_auc5 = auc(fpr5, tpr5)
# 绘制 ROC 曲线
plt.rcParams['font.sans-serif'] = 'Arial'
plt.plot(fpr4, tpr4, color='crimson',  label='XGBoost (AUC = %0.4f)' % roc_auc4)
plt.plot(fpr1, tpr1, color='orange', label='SVM (AUC = %0.4f)' % roc_auc1)
plt.plot(fpr3, tpr3, color='steelblue',  label='RF (AUC = %0.4f)' % roc_auc3)
plt.plot(fpr2, tpr2, color='mediumseagreen', label='LR (AUC = %0.4f)' % roc_auc2)
plt.plot(fpr5, tpr5, color='gold',label='DNN (AUC = %0.4f)' % roc_auc5)
plt.legend(loc='lower right' ,fontsize=12)
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