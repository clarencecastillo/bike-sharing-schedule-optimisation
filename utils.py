import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

def tseriesplot(data, time_col, continuous_col):
    sns.lineplot(y=continuous_col, x=time_col, data=data, color='c')
    title = 'Timeseries data of ' + continuous_col
    plt.title(title, loc = 'center', y=1.1, fontsize = 25)
    plt.tight_layout()
    plt.show()
    plt.close()

def boxplot(data, category_col, continuous_col):
    sns.boxplot(y=continuous_col, x=category_col, data=data, color='c')
    title = 'Distribution of ' + continuous_col
    plt.title(title, loc = 'center', y=1.1, fontsize = 25)
    plt.tight_layout()
    plt.show()
    plt.close()

def haversine(lon1, lat1, lon2, lat2):
    """
    https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km
