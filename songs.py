import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
songs = pd.read_csv('songs.csv')
X = songs.drop(['song','artist','genre','lyrics'],axis=1)
y = songs.artist
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.33)
max_predict = 0
max_depth = -1
for depth in range(1,100):
    clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = depth)
    clf.fit(X_train,y_train)
    predictions = clf.predict(X_test)
    predict = precision_score(y_test, predictions, average = 'micro')
    if predict > max_predict:
        max_predict = predict
        max_depth = depth
    print(predict)
predict = max_predict
print(predict)
print(max_depth)
