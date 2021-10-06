import pandas as pd
import numpy as np
sp = pd.read_csv('StudentsPerformance.csv')


sp['total score'] = sp['math score'] + sp['reading score'] + sp['writing score']
sp = sp.assign(total_score_log = np.log(sp['total score']))
print(sp.drop(['total score'], axis = 1))
 
