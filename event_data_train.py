import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
event_data = pd.read_csv('event_data_train.csv')
event_data['date'] = pd.to_datetime(event_data.timestamp, unit = 's')
event_data['day'] = event_data.date.dt.date
submissions_data = pd.read_csv('submissions_data_train.csv')
##print(event_data.groupby('day').user_id.nunique().plot())
##print(event_data[event_data.action == 'passed'].groupby('user_id',as_index = False).agg({'step_id':'count'}).rename(columns = {'step_id':'passed_steps'}).passed_steps.min())
users_events_data = event_data.pivot_table(index = 'user_id', columns = 'action', values = 'step_id', aggfunc = 'count', fill_value = 0).reset_index()

submissions_data['date'] = pd.to_datetime(submissions_data.timestamp, unit = 's')
submissions_data['day'] = submissions_data.date.dt.date
users_scores = submissions_data.pivot_table(index = 'user_id', columns = 'submission_status', values = 'step_id', aggfunc = 'count', fill_value = 0).reset_index()
##submissions = submissions_data.drop(columns=['date', 'day'],axis=1)
##submissions = submissions[submissions.submission_status == 'wrong']
##submissions = submissions.groupby('step_id').agg({'submission_status':'count'}).sort_values('submission_status', ascending=False).head()
##print(submissions)


gap_data = event_data[['user_id' , 'day', 'timestamp']].drop_duplicates(subset = ['user_id', 'day']).groupby('user_id').timestamp.apply(list).apply(np.diff).values

gap_data = pd.Series(np.concatenate(gap_data, axis = 0))
gap_data = gap_data / (24*60*60)

users_data = event_data.groupby('user_id',as_index = False).agg({'timestamp' : 'max'}).rename(columns = {'timestamp':'last_timestamp'})

 
now = 1526772811
drop_out_threshold = 2592000
users_data['is_gone_user'] = (now - users_data.last_timestamp) > drop_out_threshold
users_data = users_data.merge(users_scores,on = 'user_id', how = 'outer')
users_data = users_data.fillna(0)
users_data = users_data.merge(users_events_data, how = 'outer')
users_days = event_data.groupby('user_id').day.nunique().to_frame().reset_index()
users_data = users_data.merge(users_days, how = 'outer')
users_data['passed_corse'] = users_data.passed > 170
user_min_time = event_data.groupby('user_id',as_index = False).agg({'timestamp':'min'}).rename({'timestamp':'min_timestamp'},axis = 1)
users_data = users_data.merge(user_min_time, how = 'outer')
event_data_train = pd.DataFrame()
##for user_id in users_data.user_id:
##    min_user_time = users_data[users_data.user_id == user_id].min_timestamp.item()
##    time_threshold = min_user_time + 3 * 24 * 60 * 60
##    user_events_data = event_data[(event_data.user_id == user_id) & (event_data.timestamp < time_threshold)]
##    event_data_train = event_data_train.append(user_events_data)
##print(event_data_train)
event_data['user_time'] = event_data.user_id.map(str) + '_' + event_data.timestamp.map(str)
learning_time_threshold = 3 * 24 * 60 * 60
users_learning_time_threshold = user_min_time.user_id.map(str) + '_' + (user_min_time.min_timestamp + learning_time_threshold).map(str)
user_min_time['users_learning_time_threshold'] = users_learning_time_threshold
event_data = event_data.merge(user_min_time[['user_id','users_learning_time_threshold']], how = 'outer')
event_data_train = event_data[event_data.user_time <= event_data.users_learning_time_threshold]
submissions_data['users_time'] = submissions_data.user_id.map(str) + '_' + submissions_data.timestamp.map(str)
submissions_data = submissions_data.merge(user_min_time[['user_id', 'users_learning_time_threshold']], how='outer')
submissions_data_train = submissions_data[submissions_data.users_time <= submissions_data.users_learning_time_threshold]

X = submissions_data_train.groupby('user_id').day.nunique().to_frame().reset_index().rename(columns = {'day':'days'})
steps_tried = submissions_data_train.groupby('user_id').step_id.nunique().to_frame().reset_index().rename(columns = {'step_id':'steps_tried'})
X = X.merge(steps_tried, on='user_id', how='outer')
X = X.merge(submissions_data_train.pivot_table(index = 'user_id', columns = 'submission_status', values = 'step_id', aggfunc = 'count', fill_value = 0).reset_index())
X['correct_ratio'] = X.correct / (X.correct + X.wrong)
X= X.merge(event_data_train.pivot_table(index = 'user_id', columns = 'action', values = 'step_id', aggfunc = 'count', fill_value = 0).reset_index()[['user_id','viewed']],how='outer')
X = X.fillna(0)
X = X.merge(users_data[['user_id','passed_corse','is_gone_user']], how = 'outer')
X = X[~((X.is_gone_user == False) & (X.passed_corse == False))]
y = X.passed_corse.map(int)
X = X.drop(['passed_corse','is_gone_user'],axis=1)
X = X.set_index(X.user_id)
X = X.drop(['user_id'], axis = 1)
print(X)
print(y)



