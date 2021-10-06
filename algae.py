import pandas as pd
import numpy as np
algae = pd.read_csv('algae.csv')
print(algae.groupby('group').describe())
print(algae.sort_values('group'))
