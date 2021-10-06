import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
dogs_data = pd.read_csv('dogs_n_cats.csv')
X= dogs_data.drop(['Вид'],axis=1)
y= dogs_data['Вид']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)
clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)
clf.fit(X_train,y_train)
df_ts = pd.read_json('dataset_209691_15.json')
print(df_ts)
result = clf.predict(df_ts)
print(pd.Series(result)[result == 'собачка'].count())
