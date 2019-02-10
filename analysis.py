import pandas as pd
import numpy as np
from datetime import datetime
import calendar
from utils import *

### DATA EXPLORATION

journeys = pd.read_csv('data/journeys.csv')
stations = pd.read_csv('data/stations.csv')

#### 1. Standard exploration
journeys.columns
stations.columns
journeys.head(3)
journeys.describe()
stations.describe()

'''
> Journey ID is used as surrogate identifier of trips. Consider dropping column.

> All entries take place in the year of 2017. Consider taking out Start Year and
End Year since all rows have the same value.

> Might not yield much use keeping the Start and End timings atomic i.e. separate
columns for Hour and Minute. Consider transforming into time series date object.

> Consider calculating distance for each journey using the coordinates of Start
and End stations.

> It is observed that the minimum value for Journey Duration is 0. Can infer that
there exists entries that have been prematurely cancelled i.e. trips that were
immediately cancelled upon booking. Verify if such trips have the same Start and
End timings and consider dropping those rows.
'''

#### 2.1 Transform time type into string for easy concat
time_columns = ['Start Date', 'Start Month', 'Start Year', 'Start Hour',
'Start Minute', 'End Date', 'End Month', 'End Year', 'End Hour', 'End Minute']
journeys[time_columns] = journeys[time_columns].astype(str)
journeys.dtypes

#### 2.2 Transform time columns into time series datetime
pad_zero = lambda x: '0' + x if len(x) == 1 else x
for prefix in ['Start', 'End']:
    date = journeys[prefix + ' Date'] + '/' + journeys[prefix + ' Month'] + '/' + journeys[prefix + ' Year']
    time = journeys[prefix + ' Hour'].apply(pad_zero) + journeys[prefix + ' Minute'].apply(pad_zero)
    time_str = date + ' ' + time
    journeys[prefix + ' Time'] = pd.to_datetime(time_str, format='%d/%m/%y %H%M')

journeys.dtypes
journeys[prefix + ' Time'].head(3)

#### 3. Drop unwanted columns
unwanted_cols = time_columns + ['Journey ID']
journeys = journeys.drop(columns = unwanted_cols)
journeys.columns

#### 4. Calculate distance (in km)
stations_coordinates = stations[['Station ID', 'Latitude', 'Longitude']]
for prefix in ['Start', 'End']:
    journeys = pd.merge(journeys, stations_coordinates, left_on=prefix + ' Station ID', right_on='Station ID')
    journeys = journeys.drop('Station ID', axis=1).rename(index=str, columns={"Longitude": prefix + " Longitude", "Latitude": prefix + " Latitude"})
journeys['Distance'] = journeys.apply(lambda x: haversine(x['Start Longitude'], x['Start Latitude'], x['End Longitude'], x['End Latitude']), axis=1)
journeys = journeys.drop(columns = ['Start Longitude', 'Start Latitude', 'End Longitude', 'End Latitude'])
journeys.head(3)
journeys['Distance'].describe()

#### 5.1 Drop rows which have zero duration
is_zero_duration = journeys['Journey Duration'] == 0
journeys[is_zero_duration].count() # 1595
journeys = journeys[~is_zero_duration]

#### 5.2 Drop rows which have the same start and end time (invalid)
is_same_start_end_time = journeys['End Time'] == journeys['Start Time']
journeys[is_same_start_end_time].count() # 2612
journeys = journeys[~is_same_start_end_time]

#### 6.1 Group journeys by Station ID and by 60 minute intervals using count

def get_station_time_groups(prefix, granularity):
    grouper = pd.Grouper(key=prefix+' Time', freq=str(granularity) + 'Min', label='right')
    groups = journeys.groupby([prefix+' Station ID', grouper]).size()
    groups = groups.unstack(fill_value=0).stack() # fill nonexistent counts as 0
    return groups.reset_index()

granularity = 60 # minutes
station_groups = get_station_time_groups('Start', granularity)
station_groups = station_groups.rename(columns={0: 'Out', 'Start Station ID': 'Station ID', 'Start Time': 'Time'})
station_groups['In'] = get_station_time_groups('End', granularity)[0]

# sample plot
station_groups[station_groups['Station ID'] == 1].plot(x='Time', y=['Out'], kind='line', figsize=(60, 10))

#### 6. Feature Engineering

#### 6.1 Generate Lag
directions = ['In', 'Out']
periods = range(1, 11)
# periods = range(1, 3)

# generate lag columns
station_ids = station_groups['Station ID'].unique()
for stationId in station_ids:
    station_id_match = station_groups['Station ID'] == stationId
    for direction, t_lag in [(direction, t_lag) for direction in directions for t_lag in periods]:
        minute_lag_label = direction + ' Lag ' + str(t_lag)
        station_groups.loc[station_id_match, minute_lag_label] = station_groups[station_id_match][[direction]].shift(t_lag).values
        day_lag_label = direction + ' Lag Day ' + str(t_lag)
        station_groups.loc[station_id_match, day_lag_label] = station_groups[station_id_match][[direction]].shift(int(24 * (60/granularity) * t_lag)).values

# transform generated columns to int and drop shifted rows with na
lag_columns = np.concatenate([[dir + ' Lag ' + str(t), dir + ' Lag Day ' + str(t)] for t in periods for dir in directions]).ravel().tolist()
station_groups = station_groups.dropna()
station_groups[lag_columns] = station_groups[lag_columns].astype(int)

# get weekdays and hour
station_groups['Weekday'] = station_groups['Time'].apply(lambda x: calendar.day_name[x.weekday()])
station_groups['Hour'] = station_groups['Time'].apply(lambda x: x.hour)

# one hot encoding
station_groups = station_groups.join(pd.get_dummies(station_groups['Weekday']))

boxplot(station_groups, 'Weekday', 'Out')

# MODELING
# TODO: move to models.py

# 1. Multiple Linear regression on 'Out'
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

train_size = 0.7
target = 'Out'
seed = 100
unwanted_cols = ['Weekday', 'Time', 'Station ID']

# partition by time
timestamps = station_groups['Time'].unique()
train_cutoff = timestamps[int(len(timestamps) * train_size)]

train_set = station_groups[station_groups['Time'] <= train_cutoff]
test_set = station_groups[station_groups['Time'] > train_cutoff]

x_train = train_set.drop(columns=[target] + unwanted_cols)
y_train = train_set[[target]]
x_test = test_set.drop(columns=[target] + unwanted_cols)
y_test = test_set[[target]]

lr = LinearRegression()
lr.fit(x_train, y_train)
lr_train_predictions = lr.predict(x_train)
lr_predictions = lr.predict(x_test)

# The mean squared error
np.sqrt(mean_squared_error(y_test, lr_predictions))
# Explained variance score: 1 is perfect prediction
r2_score(y_test, lr_predictions)

# 2. Random Forest with Tuning

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
rf.fit(x_train, y_train.values.ravel())
rf_predictions = rf.predict(x_test)

np.sqrt(mean_squared_error(y_test, rf_predictions))
r2_score(y_test, rf_predictions)

# 3. Grid Search
from sklearn.model_selection import RandomizedSearchCV

grid = { 'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
         'max_features': ['auto', 'sqrt'],
         'max_depth': [int(x) for x in np.linspace(50, 150, num = 11)] + [None],
         'min_samples_split': [10, 15, 20],
         'min_samples_leaf': [1, 2, 4],
         'bootstrap': [True, False] }

rf_t = RandomForestRegressor()
rf_t_search = RandomizedSearchCV(estimator=rf_t, param_distributions=grid, n_iter=10, cv=3, verbose=2, random_state=seed, n_jobs=-1)

# Fit the random search model
rf_t_search.fit(x_train, y_train.values.ravel())
# rf_best_params = rf_t_search.__best_params

rf_best_params = {
    n_estimators: 1000,
    min_samples_split: 20,
    min_samples_leaf: 2,
    max_features: 'auto',
    max_depth: 140,
    bootstrap: True,
    n_jobs: -1
}

t = RandomForestRegressor(**rf_best_params)
t.fit(x_train, y_train.values.ravel())
t_predictions = t.predict(x_test)

np.sqrt(mean_squared_error(y_test, t_predictions))
r2_score(y_test, t_predictions)

n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
max_features = ['auto']
max_depth = [int(x) for x in np.linspace(95, 105, num = 11)]
min_samples_split = [9, 10, 15, 20]
min_samples_leaf = [1, 2, 3]
bootstrap = [True]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf_t = RandomForestRegressor()
rf_t_search = RandomizedSearchCV(estimator=rf_t, param_distributions=random_grid, n_iter=10, cv=3, verbose=2, random_state=seed, n_jobs=-1)
# Fit the random search model
rf_t_search.fit(x_train, y_train.values.ravel())

# 4. xgboost



# 5. LSTM

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

lstm_x_train = x_train.values.reshape((x_train.shape[0], 1, x_train.shape[1]))
lstm_x_test = x_test.values.reshape((x_test.shape[0], 1, x_test.shape[1]))

lstm = Sequential()
lstm.add(LSTM(100, input_shape=(lstm_x_train.shape[1], lstm_x_train.shape[2])))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')
lstm.fit(lstm_x_train, y_train, epochs=100, batch_size=512, validation_data=(lstm_x_test, y_test), verbose=2, shuffle=False)

# calculate RMSE
np.sqrt(mean_squared_error(y_test, lstm.predict(lstm_x_test)))
