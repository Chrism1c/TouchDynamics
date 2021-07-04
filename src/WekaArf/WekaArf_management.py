from pandas import DataFrame
from scipy.io import arff
import pandas as pd
import numpy as np
import glob

from src.support import get_df_from_arff

Path_WekaArf = 'D:\pycharmProjects\TouchDynamics\datasets\KeystrokeTouch\WekaArf'
Path_WekaArf_Results = 'D:\pycharmProjects\TouchDynamics\doc\Results\WekaArf_Results'

" ID utenti database"
subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 21, 24, 25, 26, 27, 28, 29, 35, 36, 37, 38, 40, 41, 50, 51, 53, 54,
            55, 65, 66, 68, 69, 70, 71, 73, 80, 81, 82, 83, 84, 85]  # 41


def load_data(index):
    datasets_WekaArf = glob.glob(Path_WekaArf + '/' + '/*.arff')
    print(datasets_WekaArf[index])
    data = get_df_from_arff(datasets_WekaArf[index])

    data['user_id'] = pd.to_numeric(data['user_id'])

    print(data)
    print(data.shape)

    return data.drop(columns=['user_id']), data['user_id'].values


if __name__ == '__main__':
    db_index = 0  # 0 - 4
    data_X, data_Y = load_data(db_index)

    print(data_X.columns)
    print(data_X)
    print("\n")
    print(data_Y)
    print("\n", len(data_Y))
