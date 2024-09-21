import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import roc_curve, roc_auc_score,auc  ###计算roc和auc
from sklearn.model_selection import train_test_split,GridSearchCV
import pandas as pd
from pandas import Series
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
import xgboost as xgb
import keras
# Import some data to play with
from sklearn.pipeline import Pipeline
data = pd.read_csv(r'data.csv',engine='python',encoding='gb18030',delimiter=",")
X=data.drop(['id','y'], axis=1)
Y=data['y']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=0)
transfer=StandardScaler()
X_train=transfer.fit_transform(X_train)
X_test=transfer.transform(X_test)

#搜索XGBoost的最佳参数
learning_rate = [0.01,0.1, 0.2,0.3,0.4]
n_estimator=[50,100, 200, 300,400,500]
max_depth = [1,3,5,6,7,9]
parameters = { 'learning_rate': learning_rate,
              'n_estimators': n_estimator,
              'max_depth': max_depth}
ratio = float(np.sum(Y == 0)) / np.sum(Y == 1)
model = XGBClassifier(scale_pos_weight=ratio)
# 进行网格搜索
clf = GridSearchCV(model, parameters)
clf = clf.fit(X_train, y_train)
# 网格搜索后的最好参数为
print(clf.best_params_)
y_score = clf.predict_proba(X_test)
print(roc_auc_score(y_test, y_score[:,1]))

#搜索SVM的最佳参数
param_dict={'C':[0.1,1,10],'kernel':['rbf','linear']}
grid = SVC(class_weight='balanced')
grid = GridSearchCV(grid, param_grid=param_dict)
grid.fit(X_train, y_train)
print(-grid.best_score_)
print(grid.best_params_)
y_score = grid.decision_function(X_test)
print(roc_auc_score(y_test, y_score))

#搜索LogisticRegression最佳参数
penaltys = ['l1', 'l2']
Cs =[0.1,0.5,1,2,4]
max_iters=[5,20,50,100,500,1000]
tuned_parameters = dict(penalty=penaltys, C=Cs,max_iter=max_iters)
grid = LogisticRegression(class_weight='balanced')
grid = GridSearchCV(grid, tuned_parameters)
grid.fit(X_train, y_train)
print(-grid.best_score_)
print(grid.best_params_)
y_score = grid.decision_function(X_test)
print(roc_auc_score(y_test, y_score))

#搜索RandomForest的最佳参数
param_dict={'max_depth':[1,3,5,10],'min_samples_leaf':[1,3,5,10]}
grid = RandomForestClassifier(class_weight='balanced')
grid = GridSearchCV(grid, param_grid=param_dict)
grid.fit(X_train, y_train)
print(-grid.best_score_)
print(grid.best_params_)
y_score = grid.predict_proba(X_test)
print(roc_auc_score(y_test, y_score[:,1]))
