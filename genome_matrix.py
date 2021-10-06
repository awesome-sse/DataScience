import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
g = pd.read_csv('genome_matrix.csv' )
sns.heatmap(g.corr(),annot = True, fmt='.1g', vmin=-1, vmax=1, center= 0, cmap="viridis")
g.xaxis.set_ticks_position('top')
g.xaxis.set_tick_params(rotation=90)
plt.show()
