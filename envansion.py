import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
data = pd.read_csv('invasion.csv')
X_train = data.drop(['class'], axis = 1)
y_train = data['class']
clf_rf = RandomForestClassifier()
params = {'n_estimators': range(10,51,10), 'max_depth':range(1,13,2), 'min_samples_split':range(2,10,2),'min_samples_leaf':range(1,8)}
search = GridSearchCV(clf_rf, params, n_jobs = -1)
search.fit(X_train, y_train)
best_clf = search.best_estimator_
best_feat = best_clf.feature_importances_
importances_df = pd.DataFrame({'Feature' : list(X_train), 'imp':best_feat})
print(importances_df.sort_values('imp',ascending = False))
best_clf.fit(X_train, y_train)
pred = pd.read_csv('operative_information.csv')
y_pred = best_clf.predict(pred)

