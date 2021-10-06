import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
iris_data = pd.read_csv('train_iris.csv')
np.random.seed(0)
iris_test_data = pd.read_csv('test_iris.csv')
X_train = iris_data.drop(['Unnamed: 0', 'species'], axis = 1)
y_train = iris_data.species
X_test = iris_test_data.drop(['Unnamed: 0', 'species'], axis = 1)
y_test = iris_test_data.species
max_depth_values = range(1,100)
scores_data = pd.DataFrame()
for max_depth in max_depth_values :
    clf = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth = max_depth)
    clf.fit(X_train,y_train)
    train_score = clf.score(X_train,y_train)
    test_score = clf.score(X_test,y_test)
    temp_score_data = pd.DataFrame({'max_depth': [max_depth],'train_score':[train_score],'test_score':[test_score]})
    scores_data = scores_data.append(temp_score_data)
scores_data_long = pd.melt(scores_data, id_vars=['max_depth'],value_vars=['train_score','test_score'],var_name='set_type',value_name='score')
sns.lineplot(x='max_depth',y='score',hue='set_type',data=scores_data_long)
plt.show()
