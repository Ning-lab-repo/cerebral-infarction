from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
data = pd.read_csv(r'data.csv',engine='python',encoding='gb18030',delimiter=",")
X=data.drop(['id','y'], axis=1)
Y=data['y']
# Add noisy features to make the problem harder

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=0)
transfer=StandardScaler()
X_train=transfer.fit_transform(X_train)
X_test=transfer.transform(X_test)
ratio = float(np.sum(Y == 0)) / np.sum(Y == 1)
xg = XGBClassifier(scale_pos_weight=ratio,learning_rate=, max_depth=, n_estimators=)#
xg.fit(X_train,y_train)
guess=xg.predict_proba(X_test)[:, 1]
y_score=xg.predict(X_test)
guess=guess>0.5
cm = confusion_matrix(y_test, guess)
cm_sum = np.sum(cm, axis=1, keepdims=True)
cm_percentage = cm / cm_sum.astype(float) * 100
plt.rcParams['font.sans-serif'] = 'Arial'
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    此函数打印并绘制混淆矩阵。
    可以通过设置 `normalize=True` 来打印百分比。
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes,fontsize=16)
    plt.yticks(tick_marks, classes,fontsize=16)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True',fontsize=16)
    plt.xlabel('Predict',fontsize=16)
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.tight_layout()
plot_confusion_matrix(cm_percentage, classes=['class1','class2' ], normalize=True)
plt.title('Confusion matrix',fontsize=20,pad=10,loc='center')
plt.subplots_adjust(bottom=0.2,top=0.9)
plt.show()
