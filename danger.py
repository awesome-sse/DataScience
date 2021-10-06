import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
danger = pd.read_csv('space_can_be_a_dangerous_place.csv')
X = danger.drop(['dangerous'], axis = 1)
y = danger.dangerous
clf_rf = RandomForestClassifier()
params = {'n_estimators': [10,20,30],'max_depth' : [1,3,5,7,9,11], 'min_samples_split': [ 1, 3, 5], 'min_samples_leaf': [1,3,5]}
search = GridSearchCV(clf_rf, params, cv = 5, n_jobs = -1)
search.fit(X,y)
best_clf = search.best_estimator_
best_clf.fit(X,y)
best_feat = best_clf.feature_importances_
frame = pd.DataFrame({'feat' : list(X), 'imp':best_feat})
print(frame.sort_values('imp',ascending = False))


