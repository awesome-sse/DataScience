import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
data = pd.read_csv('train_data_tree.csv')
X_train = data.drop(['num'], axis = 1)
y_train = data.num
clf = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth = 2)
clf.fit(X_train, y_train)
print(tree.plot_tree(clf, filled=True))
plt.show()
print(X_train)
