import pandas as pd
import numpy as np
dota = pd.read_csv('dota_hero_stats.csv')
import matplotlib.pyplot as plt
import seaborn as sns
dota['n_roles'] = dota.roles.str.count(',')+1
dota.hist()
plt.show()
