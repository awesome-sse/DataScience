import pandas as pd
my_stat = pd.read_csv('my_stat_1.csv')

mean_session_value_data = my_stat.groupby('group',as_index=False ).agg({'session_value': 'mean'}) 
mean_session_value_data.rename(columns = {'session_value':'mean_session_value'})
print(mean_session_value_data)
