import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
#чтение
print('Чтение данных')
data = pd.read_csv('trainData.csv')
print(data)
X = data.drop(['Survived','NameV'], axis = 1)
y = data['Survived']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

print('Обучение модели')

model = CatBoostClassifier(iterations=500, 
                           task_type="GPU"
                           )
model.fit(X_train,
          y_train,
          verbose=False)


y_pred = model.predict(X_test)
print(f1_score(y_test, y_pred, average=None))
print(accuracy_score(y_test, y_pred))
'''grid = grid_search(iterations=1000, 
            task_type="GPU",
            devices='0:1',
            X = X_train,
            y = y_train,
            cv=3,
            partition_random_seed=0,
            calc_cv_statistics=True,
            search_by_train_test_split=True,
            refit=True,
            shuffle=True,
            stratified=None,
            train_size=0.8,
            verbose=True,
            plot=False)
print(grid)
y_pred = grid.predict(X_test)
print(accuracy_score(y_test, y_pred))'''

print('!!!:', model.feature_importances_)
