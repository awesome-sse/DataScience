import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sd = pd.read_csv('task.csv', sep = ' ')
sd.plot.scatter(x = 'x',y='y')
plt.show()
