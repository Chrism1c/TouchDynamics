import os
import pandas as pd
import numpy as np
import math
import arff

from src.support import get_df_from_arff

Pre_Path = "D:/pycharmProjects/"
Raw_Mobikey_path = Pre_Path + 'TouchDynamics/datasets/KeystrokeTouch/MobiKey/RawData/rawdata'
Raw_Mobikey_pickle_path = Pre_Path + 'TouchDynamics/datasets/KeystrokeTouch/MobiKey/RawData'

users_mobikey = [100, 101, 102, 103, 104, 105, 200, 201, 202, 203, 300, 301, 302, 303, 400, 401, 402, 403, 404, 500,
                 501, 502, 503, 600, 601, 602, 603, 604, 605, 700, 701, 702, 800, 801, 802, 900, 1000, 1001, 1002, 1003,
                 1004, 1100, 1101, 1103, 1104, 1105, 1200, 1201, 1203, 1204, 1300, 1301, 1302, 1303]

pickle_names = ['rawData_all_data_Easy_Keystrokes.pickle', 'rawData_all_data_Logicalstrong_Keystrokes.pickle',
                'rawData_all_data_Strong_Keystrokes.pickle']

cols_extracted = [
    'holdtime1', 'holdtime2', 'holdtime3', 'holdtime4', 'holdtime5', 'holdtime6', 'holdtime7', 'holdtime8',
    'holdtime9',
    'holdtime10', 'holdtime11', 'holdtime12', 'holdtime13', 'holdtime14', 'holdtime15',
    'downdown1', 'downdown2', 'downdown3', 'downdown4', 'downdown5', 'downdown6', 'downdown7', 'downdown8',
    'downdown9',
    'downdown10', 'downdown11', 'downdown12', 'downdown13', 'downdown14',
    'updown1', 'updown2', 'updown3', 'updown4', 'updown5', 'updown6', 'updown7', 'updown8', 'updown9', 'updown10',
    'updown11', 'updown12', 'updown13', 'updown14',
    'pressure1', 'pressure2', 'pressure3', 'pressure4', 'pressure5', 'pressure6', 'pressure7', 'pressure8',
    'pressure9',
    'pressure10', 'pressure11', 'pressure12', 'pressure13', 'pressure14', 'pressure15',
    'fingerarea1', 'fingerarea2', 'fingerarea3', 'fingerarea4', 'fingerarea5', 'fingerarea6', 'fingerarea7',
    'fingerarea8',
    'fingerarea9', 'fingerarea10', 'fingerarea11', 'fingerarea12', 'fingerarea13', 'fingerarea14', 'fingerarea15',
    'meanholdtime',
    'meanpressure',
    'meanfingerarea',
    'meanxaccelaration',
    'meanyaccelaration',
    'meanzaccelaration',
    'velocity',
    'totaltime',
    'totaldistance',
    'user_id'
]

to_remove_for_logicalstrong_strong_db = ['holdtime14', 'holdtime15',
                                         'downdown13', 'downdown14',
                                         'updown13', 'updown14',
                                         'pressure14', 'pressure15',
                                         'fingerarea14', 'fingerarea15',
                                         ]

to_keep_for_SOF_datasets = [
    'meanholdtime',
    'meanpressure',
    'meanfingerarea',
    'meanxaccelaration',
    'meanyaccelaration',
    'meanzaccelaration',
    'velocity',
    'totaltime',
    'totaldistance',
    'user_id'
]


def raw_data_to_pickle(Raw_Mobikey_path, Raw_Mobikey_pickle_path):
    main_dir = os.listdir(Raw_Mobikey_path)
    data_evoline1 = pd.read_csv(Raw_Mobikey_path + '/' + main_dir[0] + '/' + 'Keystrokes.csv')
    data_evoline2 = pd.read_csv(Raw_Mobikey_path + '/' + main_dir[1] + '/' + 'Keystrokes.csv')
    data_evoline3 = pd.read_csv(Raw_Mobikey_path + '/' + main_dir[2] + '/' + 'Keystrokes.csv')
    data_evoline6 = pd.read_csv(Raw_Mobikey_path + '/' + main_dir[3] + '/' + 'Keystrokes.csv')
    data_lynx = pd.read_csv(Raw_Mobikey_path + '/' + main_dir[4] + '/' + 'Keystrokes.csv')
    data_matekinfo1 = pd.read_csv(Raw_Mobikey_path + '/' + main_dir[5] + '/' + 'Keystrokes.csv')
    data_matekinfo2 = pd.read_csv(Raw_Mobikey_path + '/' + main_dir[6] + '/' + 'Keystrokes.csv')
    data_matekinfo3 = pd.read_csv(Raw_Mobikey_path + '/' + main_dir[7] + '/' + 'Keystrokes.csv')
    data_reea2 = pd.read_csv(Raw_Mobikey_path + '/' + main_dir[8] + '/' + 'Keystrokes.csv')
    data_smartsoft = pd.read_csv(Raw_Mobikey_path + '/' + main_dir[9] + '/' + 'Keystrokes.csv')
    data_villamos1 = pd.read_csv(Raw_Mobikey_path + '/' + main_dir[10] + '/' + 'Keystrokes.csv')
    data_villamos2 = pd.read_csv(Raw_Mobikey_path + '/' + main_dir[11] + '/' + 'Keystrokes.csv')
    data_villamos3 = pd.read_csv(Raw_Mobikey_path + '/' + main_dir[12] + '/' + 'Keystrokes.csv')

    raw_data = pd.concat([
        data_evoline1,
        data_evoline2,
        data_evoline3,
        data_evoline6,
        data_lynx,
        data_matekinfo1,
        data_matekinfo2,
        data_matekinfo3,
        data_reea2,
        data_smartsoft,
        data_villamos1,
        data_villamos2,
        data_villamos3
    ], ignore_index=True)

    print(raw_data)
    print(raw_data.shape)
    # Rimuovo spazi prima degli attributi
    raw_data = raw_data.set_axis(
        ['UserId', 'DeviceId', 'SessionId', 'Key', 'DownTime', 'UpTime', 'Pressure', 'FingerArea', 'RawX', 'RawY',
         'gravityX', 'gravityY', 'gravityZ', 'Hands', 'PasswordType', 'Repetition'], axis=1)
    print(raw_data)
    print(raw_data.shape)
    raw_data.to_pickle(Raw_Mobikey_pickle_path + '/' + 'rawData_Keystrokes.pickle')


def get_all_data(raw_data, pw_index):
    # print(raw_data)
    columns_names = list(raw_data.columns)
    print(columns_names)
    all_data = pd.DataFrame(columns=columns_names)
    z = 0
    for index, row in raw_data.iterrows():
        if row[14] == pw_index:
            print(list(raw_data.loc[index]))
            all_data.loc[z] = raw_data.loc[index]
            z = z + 1
        # else:
        # print('no')

    return all_data


def get_HTs(sample):
    HTS_list = list()
    for index, row in sample.iterrows():
        HTS_list.append(row['UpTime'] - row['DownTime'])
        # print(row['UpTime'] - row['DownTime'])
    return HTS_list


def get_DDs(sample):
    DDS_list = list()
    ix = 0
    for index, row in sample.iterrows():
        # print("get_DDs :", index)
        if ix < sample.shape[0] - 1:
            DDS_list.append(sample.loc[index + 1]['DownTime'] - row['DownTime'])
        ix = ix + 1
    return DDS_list


def get_UDs(sample):
    UDS_list = list()
    ix = 0
    for index, row in sample.iterrows():
        # print("get_UDs :", index)
        if ix < sample.shape[0] - 1:
            UDS_list.append(sample.loc[index + 1]['DownTime'] - row['UpTime'])
        ix = ix + 1
    return UDS_list


def get_pressure(sample):
    pressure_list = list()
    for index, row in sample.iterrows():
        pressure_list.append(row[6])
    pressure_list = [round(x, 4) for x in pressure_list]
    return pressure_list


def get_fingerarea(sample):
    fingerarea_list = list()
    for index, row in sample.iterrows():
        fingerarea_list.append(row[7])
    fingerarea_list = [round(x, 4) for x in fingerarea_list]
    return fingerarea_list


def get_meanholdtime(HTS_list):
    return round(np.mean(HTS_list), 4)


def get_meanpressure(pressure_list):
    return round(np.mean(pressure_list), 4)


def get_meanfingerarea(fingerarea_list):
    return round(np.mean(fingerarea_list), 4)


def get_meanxaccelaration(sample):
    xaccelaration_list = list()
    for index, row in sample.iterrows():
        xaccelaration_list.append(row['gravityX'])
    return round(np.mean(xaccelaration_list), 4)


def get_meanyaccelaration(sample):
    yaccelaration_list = list()
    for index, row in sample.iterrows():
        yaccelaration_list.append(row['gravityY'])
    return round(np.mean(yaccelaration_list), 4)


def get_meanzaccelaration(sample):
    zaccelaration_list = list()
    for index, row in sample.iterrows():
        zaccelaration_list.append(row['gravityZ'])
    return round(np.mean(zaccelaration_list), 4)


def get_velocity(sample):
    # get_totaldistance(sample) / get_totaltime(sample)
    return round(get_totaldistance(sample) / get_totaltime(sample), 4)


def get_totaltime(sample):
    # [last_index][5] - [fisrt_index][4]
    tail = sample.tail(1)
    head = sample.head(1)
    return list(tail['UpTime'])[0] - list(head['DownTime'])[0]


def get_totaldistance(sample):
    # length path in pixel
    totaldistance = 0
    ix = 0
    for index, row in sample.iterrows():
        if ix < sample.shape[0] - 1:
            start_x = row['RawX']
            start_y = row['RawY']
            stop_x = sample.iloc[ix + 1]['RawX']
            stop_y = sample.iloc[ix + 1]['RawY']
            dx2 = (stop_x - start_x) ** 2  # (200-10)^2
            dy2 = (stop_y - start_y) ** 2  # (300-20)^2
            distance = math.sqrt(dx2 + dy2)
            totaldistance = totaldistance + distance
        ix = ix + 1
    return round(totaldistance, 4)


def get_totaldistance_test():
    rawX = [621.25, 575.0, 301.875, 155.625, 579.375, 628.75, 504.375, 365.625, 428.125, 99.375, 365.0, 103.125,
            291.875, 637.5, 98.125]

    rawY = [961.8182400000001, 863.03033, 1055.7576, 942.42426, 847.2727699999999, 947.2727699999999, 857.5758,
            850.30304, 844.8485, 959.3939999999999, 849.697, 966.0606, 849.697, 969.697, 975.7576]

    totaldistance = 0
    for i in range(len(rawX)):
        if i + 1 < len(rawX):
            start_x = rawX[i]
            start_y = rawY[i]
            stop_x = rawX[i + 1]
            stop_y = rawY[i + 1]
            dx2 = (stop_x - start_x) ** 2  # (200-10)^2
            dy2 = (stop_y - start_y) ** 2  # (300-20)^2
            distance = math.sqrt(dx2 + dy2)
            print(" -- > distance: ", distance)
            # if(i%2==0):
            totaldistance = totaldistance + distance
    return totaldistance


def get_totaldistance_fist_last(sample):
    start_x = list(sample.head(1)['RawX'])[0]
    start_y = list(sample.head(1)['RawY'])[0]
    stop_x = list(sample.tail(1)['RawX'])[0]
    stop_y = list(sample.tail(1)['RawY'])[0]
    dx2 = (stop_x - start_x) ** 2  # (200-10)^2
    dy2 = (stop_y - start_y) ** 2  # (300-20)^2
    distance = np.math.sqrt(dx2 + dy2)
    return distance, stop_x - start_x, stop_y - start_y


def split_raw_data_for_passwords():
    """Split per password"""  # Test con solo Evoline1

    # main_dir = os.listdir(Raw_Mobikey_path)
    # data_evoline1 = pd.read_csv(Raw_Mobikey_path + '/' + main_dir[0] + '/' + 'Keystrokes.csv')
    # all_data_Easy = get_all_data(data_evoline1, 0)  # 0 - Easy
    # all_data_Strong = get_all_data(data_evoline1, 1)  # 1 - Strong
    # all_data_Logicalstrong = get_all_data(data_evoline1, 2)  # 2 - Logicalstrong

    data_raw = pd.read_pickle(Raw_Mobikey_pickle_path + '/' + 'rawData_Keystrokes.pickle')
    data_raw = data_raw.set_axis(
        ['UserId', 'DeviceId', 'SessionId', 'Key', 'DownTime', 'UpTime', 'Pressure', 'FingerArea', 'RawX', 'RawY',
         'gravityX', 'gravityY', 'gravityZ', 'Hands', 'PasswordType', 'Repetition'], axis=1)

    # all_data_Easy = get_all_data(data_raw, 0)  # 0 - Easy
    all_data_Easy = data_raw.query('PasswordType == 0')

    # all_data_Strong = get_all_data(data_raw, 1)  # 1 - Strong
    all_data_Strong = data_raw.query('PasswordType == 1')

    # all_data_Logicalstrong = get_all_data(data_raw, 2)  # 2 - Logicalstrong
    all_data_Logicalstrong = data_raw.query('PasswordType == 2')

    print(all_data_Easy)
    all_data_Easy.to_pickle(Raw_Mobikey_pickle_path + '/' + 'rawData_all_data_Easy_Keystrokes.pickle')
    print(all_data_Strong)
    all_data_Strong.to_pickle(Raw_Mobikey_pickle_path + '/' + 'rawData_all_data_Strong_Keystrokes.pickle')
    print(all_data_Logicalstrong)
    all_data_Logicalstrong.to_pickle(Raw_Mobikey_pickle_path + '/' + 'rawData_all_data_Logicalstrong_Keystrokes.pickle')
    print("FINE Split per password.pikle")


def generate_AllFeatures_datasets(pickle_index, cols_extracted):
    """Estrazione features"""

    """
    # EASY #
    holdtime * 15       [5] - [4]
    downdown * 15       [index+1][4] - [index][4]
    updown * 15         [index+1][4] - [index][5]
    pressure * 15       [index][6]
    fingerarea * 15     [index][7]
    meanholdtime        SUM(holdtimes)/15
    meanpressure        SUM(meanpressures)/15
    meanfingerarea      SUM(meanfingerareas)/15
    meanxaccelaration
    meanyaccelaration
    meanzaccelaration
    velocity            totaldistance / totaltime
    totaltime           [last_index][5] - [fisrt_index][4]
    totaldistance       length path in pixel

    user_id
    """

    pickle_sizes = [83, 73, 73]

    all_data = pd.read_pickle(Raw_Mobikey_pickle_path + '/' + pickle_names[pickle_index])

    columns_names = list(all_data.columns)
    print(columns_names)

    # all_data_Easy.to_csv(Raw_Mobikey_pickle_path + '/' + 'rawData_all_data_Easy_Keystrokes.csv', index=False)

    samples_list = list()
    for subject in np.unique(list(all_data['UserId'])):  # 600 - 601 - 602 - 603 - 604 - 605
        user_filtered = all_data.query('UserId' + ' == ' + str(subject))
        for session in np.unique(list(user_filtered['SessionId'])):  # 0 - 3 - 5
            session_filtered = all_data.query(
                'UserId' + ' == ' + str(subject) + ' and ' +
                'SessionId' + ' == ' + str(session)
            )
            for rep in np.unique(list(session_filtered['Repetition'])):
                rep_filtered = all_data.query(
                    'UserId' + ' == ' + str(subject) + ' and ' +
                    'SessionId' + ' == ' + str(session) + 'and ' +
                    'Repetition' + ' == ' + str(rep)
                )
                samples_list.append(rep_filtered)

    features = list()
    extracted_features_list = list()

    res_index = 0
    for sample in samples_list:
        try:
            # calcola ogni feature e crea il vettore

            features = get_HTs(sample) + get_DDs(sample) + get_UDs(sample) + \
                       get_pressure(sample) + get_fingerarea(sample)

            features.append(get_meanholdtime(get_HTs(sample)))
            features.append(get_meanpressure(get_pressure(sample)))
            features.append(get_meanfingerarea(get_fingerarea(sample)))

            features.append(get_meanxaccelaration(sample))
            features.append(get_meanyaccelaration(sample))
            features.append(get_meanzaccelaration(sample))

            features.append(get_velocity(sample))
            features.append(get_totaltime(sample))
            features.append(get_totaldistance(sample))

            features.append(list(sample.head(1)['UserId'])[0])

            if len(features) == pickle_sizes[pickle_index]:
                extracted_features_list.append(features)
                print((list(sample.head(1)['UserId'])[0]), " -- ", len(features), ' --> ', features)
            else:
                print('Jumped -')
        except:
            print("FAIL")

    print(len(samples_list))
    print(len(cols_extracted))
    if pickle_index != 0:  # not easy pw
        cols_extracted = [feature_ for feature_ in cols_extracted if
                          feature_ not in to_remove_for_logicalstrong_strong_db]
    res = pd.DataFrame(extracted_features_list, columns=cols_extracted)

    arff_name = (pickle_names[pickle_index].replace('rawData', 'Mobikey')).replace('data_', '').replace(
        '_Keystrokes.pickle', '')
    arff.dump(Raw_Mobikey_pickle_path + '/' + arff_name + '.arff',
              res.values,
              relation=arff_name,
              names=res.columns)


def generate_SecondOrderFeatures_datasets(pickle_index):
    # Load All data dataset
    arff_names = ['Mobikey_all_Easy.arff', 'Mobikey_all_Logicalstrong.arff', 'Mobikey_all_Strong.arff']

    all_data = get_df_from_arff(Raw_Mobikey_pickle_path + '/' + arff_names[pickle_index])
    # keep SOF only features
    SOF_data = all_data.filter(to_keep_for_SOF_datasets, axis=1)
    print(SOF_data)

    arff_name = (pickle_names[pickle_index].replace('rawData', 'Mobikey')).replace('data_', '').replace(
        '_Keystrokes.pickle', '').replace('all', 'secondorder')
    arff.dump(Raw_Mobikey_pickle_path + '/' + arff_name + '.arff',
              SOF_data.values,
              relation=arff_name,
              names=SOF_data.columns)


if __name__ == '__main__':
    print('')

    """ Genera il rawData_Keystrokes.pickle unendo tutti i file csv dei raw data"""
    # raw_data_to_pickle(Raw_Mobikey_path, Raw_Mobikey_pickle_path)
    # raw_data = pd.read_pickle(Raw_Mobikey_pickle_path + '/' + 'rawData_Keystrokes.pickle')
    # print(raw_data)

    #################################################################################àààà

    """ Divide il raw_data.pickele in un pickle per ogni password (Easy/Logicalstrong/Strong)"""
    split_raw_data_for_passwords()

    #################################################################################àààà

    """ Estrae le feature e genera un Arff per ogni dataset.pickle indicandone l'indice 'pickle_index'"""
    # seleziona il file pickle da utilizzare cambiando il valore di pickle_index (0-1-2)
    pickle_index = 2  # 0 - 1 - 2

    """ FirstOrderFeatures + SecondOrderFeatures datasets extraction"""
    # generate_AllFeatures_datasets(pickle_index, cols_extracted)

    """ SecondOrderFeatures datasets extraction. NB: i dataset 'All' devono essere già estratti"""
    # generate_SecondOrderFeatures_datasets(pickle_index)

# 124,108,116,108,99,114,108,104,90,111,116,127,100,66,96,
# 265,296,293,143,329,322,320,223,201,281,128,162,484,77,
# 141,188,177,35,230,208,212,119,111,170,12,35,384,11,
# 0.1248,0.168,0.1728,0.1056,0.1776,0.1104,0.1344,0.168,0.1536,0.0816,0.1872,0.408,0.36,0.1056,0.1728,
# 0.258065,0.354839,0.354839,0.290323,0.290323,0.322581,0.290323,0.322581,0.290323,0.258065,0.354839,0.193548,0.548387,0.322581,0.225806,
# 105.8,
# 0.17536,
# 0.311828,
# -0.031504,
# 4.81867,
# 8.516545,
# 1.490264,
# 3620,
# 279.761043,
# 600
