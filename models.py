import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

data_df = pd.read_csv('data/clean.csv')

validation_size = 0.2
target = 'Out'
seed = 2019

present_cols = ['Out', 'In', 'Delta']
x_cols = [c for c in data_df.columns if c not in ['Station ID', 'Time', 'Weekday'] + present_cols]

# partition by time
dates = np.sort(data_df['Time'].unique())
split_date = dates[int(dates.shape[0] * (1 - validation_size))]
train_split = data_df['Time'] <= split_date

x_df = data_df[x_cols]
y_df = data_df[target]
x_train, x_test = x_df.loc[train_split], x_df.loc[~(train_split)]
y_train, y_test = y_df.loc[train_split], y_df.loc[~(train_split)]

eval_scores = {
    'name': [],
    'rmse': [],
    'r2': [],
    'exp_var': []
}

# benchmark
benchmark_pred = x_test['Out (d-1)']
eval_scores['name'].append('BENCHMARK')
eval_scores['rmse'].append(np.sqrt(mean_squared_error(y_test, benchmark_pred)))
eval_scores['r2'].append(r2_score(y_test, benchmark_pred))
eval_scores['exp_var'].append(explained_variance_score(y_test, benchmark_pred))

scaler = StandardScaler().fit(x_df)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

def evaluate_model(name, regressor):
    regressor.fit(x_train, y_train)
    pred = regressor.predict(x_test)
    eval_scores['name'].append(name)
    eval_scores['rmse'].append(np.sqrt(mean_squared_error(y_test, pred)))
    eval_scores['r2'].append(r2_score(y_test, pred))
    eval_scores['exp_var'].append(explained_variance_score(y_test, pred))
    return eval_scores['rmse'][-1], eval_scores['r2'][-1], eval_scores['exp_var'][-1]

def plot_feature_importance(name, importance, save=False):
    x, y = (list(x) for x in zip(*sorted(zip(importance, x_cols), key=lambda x: np.abs(x[0]), reverse=True)))
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.barplot(x, y, ax=ax)
    plt.show()
    if save:
        fig.savefig('images/{}'.format(name), dpi=fig.dpi)


from sklearn.linear_model import LinearRegression
model_lr = LinearRegression()
model_lr.fit(x_train, y_train)
evaluate_model('LR', model_lr)
plot_feature_importance('LR', model_lr.coef_)

from sklearn.ensemble import RandomForestRegressor
model_rf = RandomForestRegressor(n_estimators=10, random_state=seed, n_jobs=-1)
model_rf.fit(x_train, y_train)
evaluate_model('RF', model_rf)
plot_feature_importance('RF', model_rf.feature_importances_)

from xgboost import XGBRegressor
model_xgb = XGBRegressor(random_state=seed)
model_xgb = model_xgb.fit(x_train, y_train)
evaluate_model('XGB', model_xgb)
plot_feature_importance('XGB', model_xgb.feature_importances_)

from pyearth import Earth
model_mars = Earth()
model_mars = model_mars.fit(x_train, y_train)
evaluate_model('MARS', model_mars)
print(model_mars.summary())

from sklearn.ensemble import GradientBoostingRegressor
model_gbr = GradientBoostingRegressor()
model_gbr.fit(x_train, y_train)
evaluate_model('GBR', model_gbr)
plot_feature_importance('GBR', model_gbr.feature_importances_)

from sklearn.model_selection import GridSearchCV


rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 50, 100],
    'random_state': [seed]
}

rf_gridsearch = GridSearchCV(RandomForestRegressor(), rf_param_grid, cv=5, verbose=False, n_jobs=-1)
rf_gridsearch = rf_gridsearch.fit(x_train, y_train)
model_rf_tuned = rf_gridsearch.best_estimator_
evaluate_model('RF_T', model_rf_tuned)
plot_feature_importance(model_rf_tuned.feature_importances_)
rf_gridsearch.best_params_
