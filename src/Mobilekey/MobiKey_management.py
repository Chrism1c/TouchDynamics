from pandas import DataFrame
from scipy.io import arff
import pandas as pd
import numpy as np
import glob

from src.support import get_df_from_arff

Path_MobileKEY = 'D:\pycharmProjects\TouchDynamics\datasets\KeystrokeTouch\MobileKey'
Path_MobileKEY_Results = 'D:\pycharmProjects\TouchDynamics\doc\Results\MobileKey_Results'


def load_data(index):
    datasets_MobileKey = glob.glob(Path_MobileKEY + '/' + '/*.arff')
    print(datasets_MobileKey[index])
    data = get_df_from_arff(datasets_MobileKey[index])

    data['user_id'] = pd.to_numeric(data['user_id'])

    print(data)
    print(data.shape)

    return data.drop(columns=['user_id']), data['user_id'].values


columns = ['holdtime1', 'holdtime2', 'holdtime3', 'holdtime4', 'holdtime5',
       'holdtime6', 'holdtime7', 'holdtime8', 'holdtime9', 'holdtime10',
       'holdtime11', 'holdtime12', 'holdtime13', 'holdtime14', 'holdtime15',
       'downdown1', 'downdown2', 'downdown3', 'downdown4', 'downdown5',
       'downdown6', 'downdown7', 'downdown8', 'downdown9', 'downdown10',
       'downdown11', 'downdown12', 'downdown13', 'downdown14', 'updown1',
       'updown2', 'updown3', 'updown4', 'updown5', 'updown6', 'updown7',
       'updown8', 'updown9', 'updown10', 'updown11', 'updown12', 'updown13',
       'updown14', 'pressure1', 'pressure2', 'pressure3', 'pressure4',
       'pressure5', 'pressure6', 'pressure7', 'pressure8', 'pressure9',
       'pressure10', 'pressure11', 'pressure12', 'pressure13', 'pressure14',
       'pressure15', 'fingerarea1', 'fingerarea2', 'fingerarea3',
       'fingerarea4', 'fingerarea5', 'fingerarea6', 'fingerarea7',
       'fingerarea8', 'fingerarea9', 'fingerarea10', 'fingerarea11',
       'fingerarea12', 'fingerarea13', 'fingerarea14', 'fingerarea15',
       'meanholdtime', 'meanpressure', 'meanfingerarea', 'meanxaccelaration',
       'meanyaccelaration', 'meanzaccelaration', 'velocity', 'totaltime',
       'totaldistance', 'user_id']

if __name__ == '__main__':
    data_X, data_Y = load_data(0)

    print(data_X.columns)
    print(data_X)
    print("\n")
    print(data_Y)
    print("\n", len(data_Y))