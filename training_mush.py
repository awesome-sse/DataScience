print('Import libraries')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
print('Download data...')
mush_data = pd.read_csv('training_mush.csv')
X = mush_data.drop(['class'], axis = 1)
y = mush_data['class']
print('State tree...')
clf_rf = RandomForestClassifier(random_state = 0)
params = {'n_estimators': range(10,51,10), 'max_depth':range(1,13,2), 'min_samples_split':range(2,10,2),'min_samples_leaf':range(1,8)}
search = GridSearchCV(clf_rf, params, cv=3, n_jobs=-1)
search.fit(X,y)
best_clf = search.best_estimator_
print('Done')
print(best_clf)
best_features = best_clf.feature_importances_
importances_df = pd.DataFrame({'Feature' : list(X), 'imp':best_features})
print(importances_df.sort_values('imp',ascending = False))
test = pd.read_csv('testing_mush.csv')
y_test = best_clf.predict(test)
print(y_test)
count = y_test.sum()
print(count)
y_true = pd.read_csv('testing_y_mush.csv')
sns.heatmap(confusion_matrix(y_true, y_test), annot=True,annot_kws={"size": 16}, cmap = 'Blues')
plt.show()
