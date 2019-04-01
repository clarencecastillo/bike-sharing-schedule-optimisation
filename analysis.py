import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import calendar
from utils import *
from scipy import stats

SEED = 2019
np.random.seed(SEED)
sns.set()

### DATA EXPLORATION

# 1. Journeys Data

journeys_df = pd.read_csv('data/journeys.csv')

# all within month of August or September
pd.concat([journeys_df['Start Month'], journeys_df['End Month']]).unique()

# all within year of 2017
pd.concat([journeys_df['Start Year'], journeys_df['End Year']]).unique()

# distribution of trip start hour
fig = plt.figure(figsize=(14, 10))
sns.distplot(journeys_df['Start Hour'], color='skyblue', label='Trip Start')
sns.distplot(journeys_df['End Hour'], color='red', label='Trip End')
plt.title('Trip Start/End Hourly Distribution', size=30)
plt.xlabel('Hour', size=24)
plt.ylabel('Trips %', size=24)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=24)
fig.savefig('images/hour_dist')

# drop rows which have zero duration
is_zero_duration = journeys_df['Journey Duration'] == 0
journeys_df[is_zero_duration].shape[0] # 1609
journeys_df = journeys_df[~is_zero_duration]

# drop rows with outlier journey duration
journeys_df[np.abs(stats.zscore(journeys_df['Journey Duration'])) >= 0.5].shape[0] #45247
journeys_df = journeys_df[np.abs(stats.zscore(journeys_df['Journey Duration'])) < 0.5]

# reduce time columns into a single datetime column
time_columns = ['Start Date', 'Start Month', 'Start Year', 'Start Hour',
'Start Minute', 'End Date', 'End Month', 'End Year', 'End Hour', 'End Minute']
journeys_df[time_columns] = journeys_df[time_columns].astype(str)
pad_zero = lambda x: '0' + x if len(x) == 1 else x
for prefix in ['Start', 'End']:
    date = journeys_df[prefix + ' Date'] + '/' + journeys_df[prefix + ' Month'] + '/' + journeys_df[prefix + ' Year']
    time = journeys_df[prefix + ' Hour'].apply(pad_zero) + journeys_df[prefix + ' Minute'].apply(pad_zero)
    time_str = date + ' ' + time
    journeys_df[prefix + ' Time'] = pd.to_datetime(time_str, format='%d/%m/%y %H%M')

# drop rows which have the same start and end time (invalid)
is_same_start_end_time = journeys_df['End Time'] == journeys_df['Start Time']
journeys_df[is_same_start_end_time].shape[0] # 2638
journeys_df = journeys_df[~is_same_start_end_time]

# group journeys by Station ID and by set window intervals
def get_station_time_groups(prefix, granularity):
    grouper = pd.Grouper(key=prefix+' Time', freq=str(granularity) + 'Min', label='right')
    groups = journeys_df.groupby([prefix+' Station ID', grouper]).size()
    groups = groups.unstack(fill_value=0).stack() # fill nonexistent counts as 0
    return groups.reset_index()

granularity = 60 * 6 # minutes
journeys_count_df = get_station_time_groups('Start', granularity)
journeys_count_df = journeys_count_df.rename(columns={0: 'Out', 'Start Station ID': 'Station ID', 'Start Time': 'Time'})
journeys_count_df['In'] = get_station_time_groups('End', granularity)[0]
journeys_count_df['Delta'] = journeys_count_df['In'] - journeys_count_df['Out']

journeys_count_df.head(10)

# station 1 incoming and outgoing trips
station_1_sept_journeys = journeys_count_df[(journeys_count_df['Station ID'] == 1) & (journeys_count_df['Time'].apply(lambda t: t.month) == 8)]
fig = plt.figure(figsize=(24, 10))
plt.title('[Station 1] Number of Incoming and Outgoing Trips for September', size=30)
sns.lineplot(station_1_sept_journeys['Time'], station_1_sept_journeys['Out'], color='red', label='Trip End')
sns.lineplot(station_1_sept_journeys['Time'], station_1_sept_journeys['In'], color='skyblue', label='Trip Start')
plt.xlabel('Time', size=24)
plt.ylabel('Trips', size=24)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=24)
fig.savefig('images/station_1_sept_journeys')

#### 6.1 Generate Footprint Data
bike_count_cols = ['In', 'Out', 'Delta']
t_periods = 20
d_periods = 10

# generate period lagged and moving average columns
station_ids = journeys_count_df['Station ID'].unique()
for station_id in station_ids:
    station_id_match = journeys_count_df['Station ID'] == station_id
    for col, t_lag in [(col, t_lag) for col in bike_count_cols for t_lag in range(1, t_periods + 1)]:

        minute_lag_label = '{} (t-{})'.format(col, t_lag)
        journeys_count_df.loc[station_id_match, minute_lag_label] = journeys_count_df[station_id_match][[col]].shift(t_lag).values

        ma_lag_label = '{} Moving Average (t-{})'.format(col, t_lag)
        journeys_count_df.loc[station_id_match, ma_lag_label] = journeys_count_df[station_id_match][[col]].shift(periods=1).rolling(window=t_lag).mean().values

# generate day lagged columns
station_ids = journeys_count_df['Station ID'].unique()
for station_id in station_ids:
    station_id_match = journeys_count_df['Station ID'] == station_id
    for col, t_lag in [(col, t_lag) for col in bike_count_cols for t_lag in range(1, d_periods + 1)]:

        day_lag_label = '{} (d-{})'.format(col, t_lag)
        journeys_count_df.loc[station_id_match, day_lag_label] = journeys_count_df[station_id_match][[col]].shift(int(24 * (60/granularity) * t_lag)).values

# drop shifted rows with na
lag_columns = journeys_count_df.columns[5:].tolist()
journeys_count_df = journeys_count_df.dropna().reset_index(drop=True)

station_1_sept_journeys = journeys_count_df[(journeys_count_df['Station ID'] == 1) & (journeys_count_df['Time'].apply(lambda t: t.month) == 8)]
fig = plt.figure(figsize=(24, 10))
plt.title('[Station 1] Number of Outgoing Trips for September (Lagged)', size=30)
sns.lineplot(station_1_sept_journeys['Time'], station_1_sept_journeys['Out'], label='Outgoing Trips')
sns.lineplot(station_1_sept_journeys['Time'], station_1_sept_journeys['Out Moving Average (t-10)'], label='Outgoing Trips Moving Average (t-10)')
sns.lineplot(station_1_sept_journeys['Time'], station_1_sept_journeys['Out Moving Average (t-20)'], label='Outgoing Trips Moving Average (t-20)')
plt.xlabel('Time', size=24)
plt.ylabel('Trips', size=24)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=24)
fig.savefig('images/station_1_sept_journeys_lagged')

# get weekdays and hour
journeys_count_df['Weekday'] = journeys_count_df['Time'].apply(lambda x: calendar.day_name[x.weekday()])
journeys_count_df['Hour'] = journeys_count_df['Time'].apply(lambda x: x.hour)

# one hot encoding
journeys_count_df = journeys_count_df.join(pd.get_dummies(journeys_count_df['Weekday']))

# get station hourly mean
for col in bike_count_cols:
    period_mean_label = '{} Station Period Mean'.format(col)
    period_mean = journeys_count_df.groupby(['Station ID', 'Hour'])[col].mean().reset_index().rename(columns = {col: period_mean_label})
    journeys_count_df = journeys_count_df.merge(period_mean)

stations_df = pd.read_csv('data/stations.csv')

import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points

# reverse station coordinates to its corresponding borough
crs = {'init': 'epsg:4326'}
geo_map = gpd.read_file('data/london_wards_2014/London_Ward_CityMerged.shp')
geo_map = geo_map.to_crs(crs)
geo_map = geo_map[['BOROUGH', 'LB_GSS_CD', 'geometry']].dissolve(by='BOROUGH').reset_index()
stations_geometry = [Point(x,y) for x, y in zip(stations_df['Longitude'], stations_df['Latitude'])]
stations_df = gpd.GeoDataFrame(stations_df, crs=crs, geometry=stations_geometry)

unwated_geo_cols = ['Latitude', 'Longitude', 'Station Name', 'geometry']
stations_df['Borough'] = stations_df['geometry'].apply(lambda p: geo_map['LB_GSS_CD'][geo_map['geometry'].apply(lambda b: b.contains(p)).idxmax(1)])

geo_ax = geo_map.plot(color='lightgray', figsize=(20, 20))
geo_ax.set_title("London Boroughs", size=30);
stations_df.plot(marker='*', color='red', markersize=5, ax=geo_ax);
min_x, min_y, max_x, max_y = stations_df.total_bounds
x_margin = (max_x - min_x) * 0.1
y_margin = (max_y - min_y) * 0.1
geo_ax.set_xlim(min_x - x_margin, max_x + x_margin)
geo_ax.set_ylim(min_y - y_margin, max_y + y_margin)
plt.savefig('images/london_map')

boroughs_df = pd.read_csv('data/london_boroughs.csv', na_values=['.'])

# parse currency format
boroughs_df['Household Median Income Estimates'] = boroughs_df['Household Median Income Estimates'].replace('[\£,]', '', regex=True).astype(float)

# set boroughs df na values as mean
boroughs_df.isna().sum()
boroughs_df = boroughs_df.fillna(boroughs_df.mean())
boroughs_df = boroughs_df.rename(columns={c:'Borough {}'.format(c) for c in boroughs_df.columns})

cols_to_plot = [
    ['Cars', 'Cycling Adults at Least Once per Month', 'Public Transport Accessibility Average Score', 'Workplace Number of Jobs'],
    ['Population Estimate', 'Gross Annual Pay', 'Working-Age People With Degree', 'Active Businesses']
]
fig, ax = plt.subplots(2,4, figsize=(24, 10))
for x, row in enumerate(cols_to_plot):
    for y, col in enumerate(cols_to_plot[x]):
        sns.distplot(boroughs_df['Borough ' + col], ax=ax[x][y], kde=False)
fig.savefig('images/boroughs_dist')

# merge borugh and station data
merged_df = stations_df.merge(boroughs_df, how='inner', left_on='Borough', right_on='Borough Code')
merged_df = merged_df.drop(columns=['Latitude', 'Longitude', 'Borough', 'geometry', 'Borough Name', 'Station Name', 'Borough Code'])
merged_df = merged_df.merge(journeys_count_df, on='Station ID')

merged_df.to_csv('data/clean.csv', index=False)
