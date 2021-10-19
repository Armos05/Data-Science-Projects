import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

train = pd.read_csv("kaggle_bike_sharing_train.csv")
test = pd.read_csv("kaggle_bike_sharing_test.csv")

train_date = pd.DatetimeIndex(train['datetime'])
train['year'] = train_date.year
train['month'] = train_date.month
train['hour'] = train_date.hour
train['dayofweek'] = train_date.dayofweek

test_date = pd.DatetimeIndex(test['datetime'])
test['year'] = test_date.year
test['month'] = test_date.month
test['hour'] = test_date.hour
test['dayofweek'] = test_date.dayofweek

features = ['season', 'holiday', 'workingday', 'weather',
            'temp', 'atemp', 'humidity', 'windspeed',
            'year', 'hour','month', 'dayofweek', 'hour_workingday_casual', 'count_season']
reg = GradientBoostingRegressor(n_estimators=1000, min_samples_leaf=6, random_state=0)
reg.fit(train[features], train['casual_log'])
pred_casual = reg.predict(test[features])
pred_casual = np.exp(pred_casual) - 1
pred_casual[pred_casual < 0] = 0

#Perdiction for Registration Demand
features = ['season', 'holiday', 'workingday', 'weather',
            'temp', 'atemp', 'humidity', 'windspeed',
            'year', 'hour', 'month', 'dayofweek', 'hour_workingday_registered', 'count_season']
reg = GradientBoostingRegressor(n_estimators=1000, min_samples_leaf=6, random_state=0)
reg.fit(train[features], train['registered_log'])
pred_registered = reg.predict(test[features])
pred_registered = np.exp(pred_registered) - 1
pred_registered[pred_registered < 0] = 0

pred1 = pred_casual + pred_registered

submission = pd.DataFrame({'datetime':test.datetime, 'count':pred1},
                          columns = ['datetime', 'count'])
submission.to_csv("submission_1.csv", index=False)
