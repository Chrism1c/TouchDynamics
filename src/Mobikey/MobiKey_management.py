import pandas as pd
import glob
from src.support import get_df_from_arff

Pre_path = 'D:/pycharmProjects/'
Path_MobiKey = Pre_path + 'TouchDynamics/datasets/KeystrokeTouch/MobiKey/FeaturesExtracted'
Path_MobiKey_Results = Pre_path + 'TouchDynamics/doc/Results/MobiKey_Results'


def load_data(db_index):
    datasets_MobileKey = glob.glob(Path_MobiKey + '/' + '/*.arff')
    # print(datasets_WekaArf[db_index])
    data = get_df_from_arff(datasets_MobileKey[db_index])
    data['user_id'] = pd.to_numeric(data['user_id'])
    # print(data)
    return data, data['user_id'].values

" ID utenti database"
subjects = [100, 101, 102, 103, 104, 105, 200, 201, 202, 203, 300, 301, 302, 303, 400, 401, 402, 403, 404, 500, 501,
            502, 503, 600, 601, 602, 603, 604, 605, 700, 701, 702, 800, 801, 802, 900, 1000, 1001, 1002, 1003, 1004,
            1100, 1101, 1103, 1104, 1105, 1200, 1201, 1203, 1204, 1300, 1301, 1302, 1303]


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

# if __name__ == '__main__':
#     data_X, data_Y = load_data(0)
#
#     print(data_X.columns)
#     print(data_X)
#     print("\n")
#     print(data_Y)
#     print("\n", len(data_Y))
