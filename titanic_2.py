import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
titanic_data = pd.read_csv('train.csv')
X = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis = 1)
y = titanic_data.Survived
X = pd.get_dummies(X)
X = X.fillna({'Age':X.Age.median()})
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.33)
clf_rf = RandomForestClassifier()
params = {'n_estimators':[10,20,30], 'max_depth': [2,5,7,10]}
grid_search_cv_clf = GridSearchCV(clf_rf, params, cv=5)
grid_search_cv_clf.fit(X_train, y_train)
best_clf = grid_search_cv_clf.best_estimator_
feature_importances = best_clf.feature_importances_
print(feature_importances)
feature_importances_df = pd.DataFrame({'features': list(X_train),'feature_importances':feature_importances})
print(feature_importances_df.sort_values('feature_importances',ascending= False))
