from pandas import DataFrame
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

PathFrank = 'D:/pycharmProjects/TouchDynamics/datasets/Swipes/Frank/'
PathFrankArff = 'D:\pycharmProjects\TouchDynamics\datasets\Swipes\Frank\data_arff\dataset.arff'

CSV_fileName = 'FileName2.csv'
pickle_fileName = 'touchalytics.pikle'


def generate_pickle(PathFrank, pickle_fileName):
    data = pd.read_csv(PathFrank + CSV_fileName)
    for col in data.columns:
        print(col)
    print(len(data.columns))
    data.to_pickle(PathFrank + '/' + pickle_fileName)


def load_data():
    import numpy as np
    data = pd.read_pickle(PathFrank + '/' + pickle_fileName)

    clean_dataset(data)

    # print(data.dtypes)

    # data['user id'] = data['user id'].astype(np.float64)
    # data['doc id'] = data['doc id'].astype(np.float64)
    # data['up/down/left/right flag'] = data['up/down/left/right flag'].astype(np.float64)
    # data['phone id'] = data['phone id'].astype(np.float64)
    # data['change of finger orientation'] = data['change of finger orientation'].astype(np.float64)
    # data['phone orientation'] = data['phone orientation'].astype(np.float64)

    # print(data.dtypes)

    data_l = {}
    data_l['total'] = data.drop(columns=['doc id', 'phone id', 'change of finger orientation'])
    data_l['total_dropped'] = data.drop(columns=['user id', 'doc id', 'phone id', 'change of finger orientation'])
    data_l['user id'] = data['user id']
    # data_l['pca3'] = pd.DataFrame(PCA(n_components=3).fit_transform(data_l['total']))
    # data_l['pca10'] = pd.DataFrame(PCA(n_components=10).fit_transform(data_l['total']))

    return data_l, data['user id'].values


columns = ['user id', 'doc id', 'inter-stroke time', 'stroke duration',
           'start $x$', 'start $y$', 'stop $x$', 'stop $y$',
           'direct end-to-end distance', 'mean resultant lenght',
           'up/down/left/right flag', 'direction of end-to-end line', 'phone id',
           '20\%-perc. pairwise velocity', '50\%-perc. pairwise velocity',
           '80\%-perc. pairwise velocity', '20\%-perc. pairwise acc',
           '50\%-perc. pairwise acc', '80\%-perc. pairwise acc',
           'median velocity at last 3 pts',
           'largest deviation from end-to-end line',
           '20\%-perc. dev. from end-to-end line',
           '50\%-perc. dev. from end-to-end line',
           '80\%-perc. dev. from end-to-end line', 'average direction',
           'length of trajectory',
           'ratio end-to-end dist and length of trajectory', 'average velocity',
           'median acceleration at first 5 points', 'mid-stroke pressure',
           'mid-stroke area covered', 'mid-stroke finger orientation',
           'change of finger orientation', 'phone orientation'],


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


if __name__ == '__main__':
    data_l, values = load_data()
    data_X = data_l['total']
    data_Y = data_l['user id']

    print(data_X)
    print("\n")
    print(data_Y)
    print("\n", len(values))
