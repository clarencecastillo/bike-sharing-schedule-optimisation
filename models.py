import pandas as pd
import numpy as np
from datetime import datetime
import calendar
from utils import *
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE

# data import
# journeys and stations optional
# journeys = pd.read_csv('data/journeys.csv')
# stations = pd.read_csv('data/stations.csv')
station_groups = pd.read_csv('data/station_groups.csv')

# check colnames + see unwanted columns
station_groups.columns
'''
 Index(['Unnamed: 0', 'Station ID', 'Time', 'Out', 'In', 'In Lag 1',
       'In Lag Day 1', 'In Lag 2', 'In Lag Day 2', 'In Lag 3', 'In Lag Day 3',
       'In Lag 4', 'In Lag Day 4', 'In Lag 5', 'In Lag Day 5', 'In Lag 6',
       'In Lag Day 6', 'In Lag 7', 'In Lag Day 7', 'In Lag 8', 'In Lag Day 8',
       'In Lag 9', 'In Lag Day 9', 'In Lag 10', 'In Lag Day 10', 'Out Lag 1',
       'Out Lag Day 1', 'Out Lag 2', 'Out Lag Day 2', 'Out Lag 3',
       'Out Lag Day 3', 'Out Lag 4', 'Out Lag Day 4', 'Out Lag 5',
       'Out Lag Day 5', 'Out Lag 6', 'Out Lag Day 6', 'Out Lag 7',
       'Out Lag Day 7', 'Out Lag 8', 'Out Lag Day 8', 'Out Lag 9',
       'Out Lag Day 9', 'Out Lag 10', 'Out Lag Day 10', 'Weekday', 'Hour',
       'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
       'Wednesday'],
      dtype='object')
'''
# station_groups.describe()
unwanted_cols = ['Hour', 'Weekday', 'Unnamed: 0', 'Station ID', 'Time']

# 1: xgb modelling

# train-test split
train_size = 0.7
target = 'Out'
seed = 100

# partition by time
timestamps = station_groups['Time'].unique()
train_cutoff = timestamps[int(len(timestamps) * train_size)]

# convert time to months; we drop day since already there
station_groups['Time'] = pd.to_datetime(station_groups['Time'])
station_groups[['Time']].sort_values(by = 'Time')
train_set = station_groups[station_groups['Time'] <= train_cutoff]
test_set = station_groups[station_groups['Time'] > train_cutoff]
x_train = train_set.drop(columns=[target] + unwanted_cols)
y_train = train_set[[target]]
x_test = test_set.drop(columns=[target] + unwanted_cols)
y_test = test_set[[target]]
# data plotting
# predict 'out'
# plot relations between days
sns.stripplot(x = "Weekday", y = "Out", data = station_groups, jitter = True)

# creation of xgb model - will use regressor
xgb_base = XGBRegressor()
xgb_base.fit(x_train, y_train)
'''
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
'''
# predict on results
base_predictions = xgb_base.predict(x_test)
np.sqrt(mean_squared_error(y_test, base_predictions)) # 1.7213931781066434
r2_score(y_test, base_predictions) # 0.7034462247285119
xgb_base.get_params
# 1.1: rfe
# TO DO: cross-validation of model (time-series)/hyper-parameter tuning/rfe
x_train.shape # (518883, 48) - 48 features
xgb_rfe = XGBRegressor()
selector = RFE(xgb_rfe)
y_train.shape
selector.fit(x_train, y_train.values.ravel())
print("done")
rfe_predictions = selector.predict(x_test)
score = r2_score(y_test, rfe_predictions)
# we can also do a search through the possible values of features from 5-24; takes quite long maybe jsut stick with half
'''
rfe_ls = []
for i in range(5, 25):
    selector = RFE(xgb_rfe)
    selector.fit(x_train, y_train.values.ravel())
    rfe_ls.append((selector, r2_score(y_test, rfe_predictions)))
print(x[1] for x in rfe_ls)
'''

# 1.2: gridsearch for optimal parameters
# should gridsearch on rfe/non-rfe and compare?
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0, 0.5, 1, 1.5, 2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
xgb_gs = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
gs = GridSearchCV(xgb_gs, param_grid = params, scoring = ['r2', 'neg_mean_squared_error'])
grid.best_estimator_
# fit and predict as per usual stuff above
