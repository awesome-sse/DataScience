import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import cross_val_score
titanic_data = pd.read_csv('train.csv')
X = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis = 1)
y = titanic_data.Survived

X = pd.get_dummies(X)
X = X.fillna({'Age':X.Age.median()})
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)
from sklearn.model_selection import GridSearchCV
clf = tree.DecisionTreeClassifier()
parametrs = {'criterion': ['gini','entropy'], 'max_depth':range(1,30),'min_samples_split': range(10, 200, 25), 
              'min_samples_leaf': range(5, 50, 5)}
grid_search_cv_clf = GridSearchCV(clf, parametrs, cv = 5)
grid_search_cv_clf.fit(X_train, y_train)
best_clf = grid_search_cv_clf.best_estimator_
from sklearn.metrics import precision_score, recall_score
y_pred = best_clf.predict(X_test)
y_predicted_prob = best_clf.predict_proba(X_test)
y_pred = np.where(y_predicted_prob[:, 1] > 0.2, 1, 0)
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_predicted_prob[:,1])
roc_auc= auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
