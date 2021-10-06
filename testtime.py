import pandas as pd
import time
import numpy as np
file = pd.read_csv('iris.csv')
before3 = time.time()
print(file.apply('mean'))
after3 = time.time()
print(' Время  file.apply(mean) =',after3-before3)
before4 = time.time()
print(file.apply(np.mean))
after4 = time.time()
print(' Время  file.apply(np.mean) =',after4-before4)
before1 = time.time()
print(file.describe().loc['mean'])
after1 = time.time()
print(' Время df.describe().loc[mean] =',after1-before1)
before2 = time.time()
print(file.mean(axis=0))
after2 = time.time()
print(' Время  df.mean(axis=0) =',after2-before2)


