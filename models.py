import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

data_df = pd.read_csv('data/clean.csv')
validation_size = 0.2
seed = 2019

def evaluate_classification_model(name, classifier):
    pred = classifier.predict(x_test)
    eval_classification_scores['name'].append(name)
    eval_classification_scores['rmse'].append(np.sqrt(mean_squared_error(y_test, pred)))
    eval_classification_scores['accuracy'].append(accuracy_score(y_test, pred))
    return eval_classification_scores['accuracy'][-1], eval_classification_scores['rmse'][-1]

def plot_feature_importance(name, importance, save=False):
    x, y = (list(x) for x in zip(*sorted(zip(importance, x_cols), key=lambda x: np.abs(x[0]), reverse=True)))
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.barplot(x, y, ax=ax)
    plt.show()
    if save:
        fig.savefig('images/{}'.format(name), dpi=fig.dpi)

dates = np.sort(data_df['Time'].unique())
split_date = dates[int(dates.shape[0] * (1 - validation_size))]
train_split = data_df['Time'] <= split_date

# fig = plt.figure(figsize=(14, 10))
# sns.lineplot(x_train['Time'], y_train)
# sns.lineplot(x_test['Time'], y_test)
# plt.title('Train/Test Data Split', size=30)
# plt.xlabel('Time', size=24)
# plt.ylabel('Outgoing Trips', size=24)
# plt.xticks([], [])
# plt.yticks(size=20)
# fig.savefig('images/data_split')

def split_dataset(x_cols, target_col):
    x_df = data_df[x_cols]
    y_df = data_df[target_col]
    x_train, x_test = x_df.loc[train_split], x_df.loc[~(train_split)]
    y_train, y_test = y_df.loc[train_split], y_df.loc[~(train_split)]

    scaler = StandardScaler().fit(x_df)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test

# ==============================================================================
# Classifying oversupplied/undersupplied/ok bike stations
# ==============================================================================
target = 'Supply'

eval_classification_scores = {
    'name': [],
    'rmse': [],
    'accuracy': []
}

# engineer target variable supply
data_df['Supply'] = data_df['Capacity'] * 0.5 + data_df['Delta']
data_df['Supply'] = np.where(data_df['Supply'] < 0, 1, np.where(data_df['Supply'] > data_df['Capacity'], 2, 3))
# 1 - undersupplied
# 2 - oversupplied
# 3 - normal

x_cols = [c for c in data_df.columns if c not in ['Station ID', 'Weekday', 'Supply', 'Out', 'In', 'Delta', 'Time']]
x_train, x_test, y_train, y_test = split_dataset(x_cols, target)

from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression(random_state=seed, multi_class='auto')
model_lr = model_lr.fit(x_train, y_train)
evaluate_classification_model('LR', model_lr)

from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier(random_state=seed)
model_dt = model_dt.fit(x_train, y_train)
evaluate_classification_model('DC', model_dt)

from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=50, random_state=seed)
model_rf = model_rf.fit(x_train, y_train)
evaluate_classification_model('RF', model_rf)

# XGBoost Classification
from xgboost import XGBClassifier, plot_importance
model_xgb = XGBClassifier(random_state=seed)
model_xgb = model_xgb.fit(x_train, y_train)
evaluate_classification_model('XGB', model_xgb)
model_xgb.get_booster().feature_names = x_cols

ax = plot_importance(model_xgb.get_booster(), max_num_features=20, title='XGB Classifier Feature Importance')
fig = ax.figure
plt.tight_layout()
fig.savefig('images/xgb_importance', figsize=(14, 10))

# ==============================================================================
# Predicting demand
# ==============================================================================
target = 'Out'

x_cols = [c for c in data_df.columns if c not in ['Station ID', 'Weekday', 'Out', 'In', 'Delta', 'Time', 'Supply']]
x_train, x_test, y_train, y_test = split_dataset(x_cols, target)

# partition by time
eval_regression_scores = {
    'name': [],
    'rmse': [],
    'r2': [],
    'exp_var': []
}

def evaluate_regression_model(name, regressor):
    pred = regressor.predict(x_test)
    eval_regression_scores['name'].append(name)
    eval_regression_scores['rmse'].append(np.sqrt(mean_squared_error(y_test, pred)))
    eval_regression_scores['r2'].append(r2_score(y_test, pred))
    eval_regression_scores['exp_var'].append(explained_variance_score(y_test, pred))
    return eval_regression_scores['rmse'][-1], eval_regression_scores['r2'][-1], eval_regression_scores['exp_var'][-1]

from sklearn.linear_model import LinearRegression
model_lr = LinearRegression()
model_lr.fit(x_train, y_train)
evaluate_regression_model('LR', model_lr)

from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor(n_estimators=10, random_state=seed, n_jobs=-1)
model_rf.fit(x_train, y_train)
evaluate_regression_model('RF', model_rf)

from xgboost import XGBRegressor
model_xgb = XGBRegressor(random_state=seed)
model_xgb = model_xgb.fit(x_train, y_train)
evaluate_regression_model('XGB', model_xgb)

from pyearth import Earth
model_mars = Earth()
model_mars = model_mars.fit(x_train, y_train)
evaluate_regression_model('MARS', model_mars)
print(model_mars.summary())

from sklearn.ensemble import GradientBoostingRegressor
model_gbr = GradientBoostingRegressor()
model_gbr.fit(x_train, y_train)
evaluate_regression_model('GBR', model_gbr)
