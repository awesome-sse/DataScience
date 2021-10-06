import pandas as pd
import numpy as np
iris = pd.read_csv('iris.csv')
import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(iris )
plt.show()
