import pandas as pd
import numpy as np
from datetime import datetime
from utils import haversine
import matplotlib.pylab as plt

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

plt.plot(journeys[['Start Time']].resample('H', on='Start Time').count())
plt.xticks(rotation=90)
plt.savefig('test.png', dpi=1080)
