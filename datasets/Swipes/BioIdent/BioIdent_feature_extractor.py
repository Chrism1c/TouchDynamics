import math
import os
import pandas as pd
import numpy as np
import arff
from src.support import get_df_from_arff, clean_dataset

Pre_Path = "D:/pycharmProjects/"
Path_raw_data_BioIdent = Pre_Path + "TouchDynamics/datasets/Swipes/BioIdent/BIOIDENT_RAWDATA"

user_id_d1_d2 = [1, 8, 22, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 75, 76, 77, 78, 79,
                 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100]
gender_d3 = [0, 1]
touch_experience_d4 = [0, 1, 2, 3]

pickle_names = ['dataset1.arff', 'dataset2.arff', 'dataset3.arff', 'dataset4.arff']

column_names = ['user_id',
                'stroke_duration',
                'start_x',
                'start_y',
                'stop_x',
                'stop_y',
                'direct_end_to_end_distance',
                'mean_resultant_length',
                'up_down_left_right',
                'direction_of_end_to_end_line',
                'largest_deviation_from_end_to_end',
                'average_direction',
                'length_of_trajectory',
                'average_velocity',
                'mid_stroke_pressure',
                'mid_stroke_area_covered',
                'gender',
                'touchscreen_experience_level'
                ]

DEBUG = True


def get_stroke(df, index):
    # print(index, df.shape[0])
    z = 0
    stroke_points = pd.DataFrame(
        columns=['device_id', 'user_id', 'doc_type', 'time', 'action', 'phone_orientaton', 'x_coor', 'y_coor',
                 'pressure', 'finger_area'])
    try:
        if df.iloc[index]['action'] == 0:
            # print("== 0")
            stroke_points.loc[z] = df.loc[index]
            z = z + 1
            j = index + 1
            while True:
                if df.iloc[j][4] == 1:
                    # print("== 1")
                    # stroke_points.loc[z] = df.loc[j]
                    if stroke_points.shape[0] >= 3:
                        return stroke_points
                    else:
                        return None
                else:
                    stroke_points.loc[z] = df.loc[j]
                    # print("== 2")
                z = z + 1
                j = j + 1
        else:
            return None
    except:
        return None


def get_direction_of_end_to_end_line(start_x, start_y, stop_x, stop_y):
    x_range = stop_x - start_x
    y_range = stop_y - start_y
    if x_range == 0:
        x_range = 0.0001
    if y_range == 0:
        y_range = 0.0001
    return round(y_range / x_range, 4)


def get_largest_deviation_from_end_to_end(stroke_points, start_x, start_y, stop_x, stop_y):
    norm = np.linalg.norm

    p1 = np.array([start_x, start_y])
    p2 = np.array([stop_x, stop_y])

    largest_deviation_from_end_to_end = -1
    # per tutti i punti in stroke_points
    for _, row in stroke_points.iterrows():
        p3 = np.array([row['x_coor'], row['y_coor']])
        deviation_from_end_to_end = np.abs(norm(np.cross(p2 - p1, p1 - p3))) / norm(p2 - p1)
        if deviation_from_end_to_end >= largest_deviation_from_end_to_end:
            largest_deviation_from_end_to_end = deviation_from_end_to_end

    return round(largest_deviation_from_end_to_end, 4)


def get_average_direction(stroke_points):
    # # rimuovo l'ultima riga
    # stroke_points = stroke_points.iloc[:-1, :]
    sum_direction = 0
    for ix in range(stroke_points.shape[0] - 1):
        start_x = stroke_points.iloc[ix]['x_coor']
        start_y = stroke_points.iloc[ix]['y_coor']
        stop_x = stroke_points.iloc[ix + 1][6]
        stop_y = stroke_points.iloc[ix + 1][7]
        # print(" stop_y . start_y ", stop_y, start_y)
        # print(" stop_x . start_x ", stop_x, start_x)
        # print(row['action'], stroke_points.iloc[index + 1]['action'])
        # print(stop_y - start_y, ' ', stop_x - start_x)
        y_range = stop_y - start_y
        x_range = stop_x - start_x
        if x_range == 0:
            x_range = 0.0001
        if y_range == 0:
            y_range = 0.0001
        fraction = (y_range / x_range)
        sum_direction = sum_direction + fraction

    return round(sum_direction / (stroke_points.shape[0] - 1), 4)


def get_lengthOfTrajectory(stroke_points):
    lenght_of_trajectory = 0
    for ix in range(stroke_points.shape[0] - 1):
        start_x = stroke_points.iloc[ix]['x_coor']
        start_y = stroke_points.iloc[ix]['y_coor']
        stop_x = stroke_points.iloc[ix + 1][6]
        stop_y = stroke_points.iloc[ix + 1][7]

        # print("start_x: ", start_x, " start_y: ", start_y, " stop_x: ", stop_x, " stop_y: ", stop_y,)
        lenght_of_trajectory = lenght_of_trajectory + np.sqrt(
            np.power((stop_x - start_x), 2) + np.power((stop_y - start_y), 2))
        # print(lenght_of_trajectory)
    return round(lenght_of_trajectory, 4)


def get_average_velocity(stroke_points):
    sum_velocity = 0
    for ix in range(stroke_points.shape[0] - 1):
        start_x = stroke_points.iloc[ix]['x_coor']
        start_y = stroke_points.iloc[ix]['y_coor']
        start_time = stroke_points.iloc[ix]['time']
        stop_x = stroke_points.iloc[ix + 1][6]
        stop_y = stroke_points.iloc[ix + 1][7]
        stop_time = stroke_points.iloc[ix + 1][3]
        # Velocità = Spazio / Tempo
        # print('start ', stop_x - start_x, ' stop ', stop_y-start_y,' time ', stop_time-start_time)
        velocity = (np.sqrt(np.power(abs(stop_x - start_x), 2) + np.power(abs(stop_y - start_y), 2))) / \
                   abs(stop_time - start_time)
        sum_velocity = sum_velocity + velocity

    return round(sum_velocity / stroke_points.shape[0] - 1, 4)


def get_mid_stroke_pressure_and_area_covered(stroke_points):
    mid_point = round(stroke_points.shape[0] / 2)
    # print(mid_point)
    # print(list(stroke_points.iloc[mid_point]))
    return round(stroke_points.iloc[mid_point][8], 4), round(stroke_points.iloc[mid_point][9], 4)


def get_strokeV2(user_filtered, full_strokes_list):
    action_list = list(user_filtered['action'])
    indexes_0 = [x for x in range(len(action_list)) if action_list[x] == 0]
    indexes_1 = [x for x in range(len(action_list)) if action_list[x] == 1]
    # print(len(indexes_0), len(indexes_1))
    # print((indexes_0), '\n', (indexes_1))
    iz = 0
    for i in range(len(indexes_0)):
        stroke_sub_df = user_filtered.iloc[indexes_0[iz]:indexes_1[iz]]
        iz = iz + 1
        # print(stroke_sub_df)
        full_strokes_list.append(stroke_sub_df)
    return full_strokes_list


def extract_features(rawdataDF, dataset_index):
    ix = 0
    new_DF = pd.DataFrame(columns=column_names)
    print()

    full_strokes_list = list()
    for subject in np.unique(list(rawdataDF['user_id'])):  # 600 - 601 - 602 - 603 - 604 - 605
        user_filtered = rawdataDF.query('user_id' + ' == ' + str(subject))
        full_strokes_list = get_strokeV2(user_filtered, full_strokes_list)
        # print('full_strokes_list: ', len(full_strokes_list))

    for stroke_points in full_strokes_list:
        if stroke_points.shape[0] > 3:
            # print(stroke_points)
            # print(stroke_points['action'])
            # try:
            user_id = list(stroke_points['user_id'])[0]
            stroke_duration = round(
                list((stroke_points.tail(1)).iloc[0])[3] - list((stroke_points.head(1)).iloc[0])[3], 4)
            start_x = round(list((stroke_points.head(1)).iloc[0])[6], 4)
            start_y = round(list((stroke_points.head(1)).iloc[0])[7], 4)
            stop_x = round(list((stroke_points.tail(1)).iloc[0])[6], 4)
            stop_y = round(list((stroke_points.tail(1)).iloc[0])[7], 4)

            direct_end_to_end_distance = round(
                np.sqrt(np.power((stop_x - start_x), 2) + np.power((stop_y - start_y), 2)), 4)
            meanResultantLength = 0
            # horizontal = "left" if start_x > stop_x else "right"
            # vertical = "down" if start_y > stop_y else "up"
            horizontal = 2 if start_x > stop_x else 3
            vertical = 1 if start_y > stop_y else 0
            up_down_left_right = horizontal if (start_x - stop_x) > (start_y - stop_y) else vertical
            # print(stop_y, '-', start_y, ' / ', stop_x, '-', start_x)
            # print(stop_y - start_y, ' / ', stop_x - start_x)
            direction_of_end_to_end_line = get_direction_of_end_to_end_line(start_x, start_y, stop_x, stop_y)
            largest_deviation_from_end_to_end = get_largest_deviation_from_end_to_end(stroke_points, start_x, start_y,
                                                                                      stop_x, stop_y)
            average_direction = get_average_direction(stroke_points)
            length_of_trajectory = get_lengthOfTrajectory(stroke_points)
            average_velocity = get_average_velocity(stroke_points)
            mid_stroke_pressure, mid_stroke_area_covered = get_mid_stroke_pressure_and_area_covered(stroke_points)
            gender = list(stroke_points['gender'])[0],
            touch_level = list(stroke_points['touchscreen_experience_level'])[0]

            features = [
                user_id,
                stroke_duration,
                start_x,
                start_y,
                stop_x,
                stop_y,
                direct_end_to_end_distance,
                meanResultantLength,
                up_down_left_right,
                direction_of_end_to_end_line,
                largest_deviation_from_end_to_end,
                average_direction,
                length_of_trajectory,
                average_velocity,
                mid_stroke_pressure,
                mid_stroke_area_covered,
                gender[0],
                touch_level
            ]

            if dataset_index > 0 and up_down_left_right < 2:
                continue
            else:  # dataset1   # dataset2   #dataset3   #dataset4
                new_DF.loc[ix] = features
                ix = ix + 1

            print(user_id, ' -- ', features)

    return new_DF


def min_max_normalization(extrated_features_data):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    extrated_features_data = clean_dataset(extrated_features_data)
    scaled_data = pd.DataFrame(scaler.fit_transform(extrated_features_data),
                               columns=extrated_features_data.columns, index=extrated_features_data.index)
    # print(scaled_data)
    # scaled_data.to_csv(Path_raw_data_BioIdent + '/' + "shit.csv")
    return scaled_data


if __name__ == '__main__':
    print("")

    """ Caricamento raw data"""
    raw_files_CSV = ['devices.csv', 'rawdata.csv', 'users.csv']
    print(" filesCSV ", raw_files_CSV)
    devicesDF = pd.read_csv(Path_raw_data_BioIdent + '/' + raw_files_CSV[0])
    rawdataDF = pd.read_csv(Path_raw_data_BioIdent + '/' + raw_files_CSV[1])
    usersDF = pd.read_csv(Path_raw_data_BioIdent + '/' + raw_files_CSV[2])

    rawdataDF = rawdataDF.drop(columns=['Unnamed: 10'])

    # print(rawdataDF)
    # print(rawdataDF.columns)

    #########################################################################à

    "Estrazione delle Features"

    dataset_index = 1  # 0 - 1 - 2 - 3

    if dataset_index == 2:  # dataset3
        rawdataDF = rawdataDF.merge(usersDF.drop(columns=['birthyear']),
                                    how='left', left_on='user_id', right_on='userid')
        print(rawdataDF)
        print(rawdataDF.columns)
        extrated_features_data = extract_features(rawdataDF, dataset_index=dataset_index)
    elif dataset_index == 3:  # dataset4
        rawdataDF = rawdataDF.merge(usersDF.drop(columns=['birthyear']),
                                    how='left', left_on='user_id', right_on='userid')
        print(rawdataDF)
        print(rawdataDF.columns)
        extrated_features_data = extract_features(rawdataDF, dataset_index=dataset_index)
    else:  # dataset 1 e 2
        rawdataDF = rawdataDF.merge(usersDF.drop(columns=['birthyear']),
                                    how='left', left_on='user_id', right_on='userid')
        extrated_features_data = extract_features(rawdataDF, dataset_index=dataset_index)
    print(extrated_features_data)

    # extrated_features_data = pd.DataFrame({'user_id': [1, 1, 2, 2, 3, 3],
    #                                        'assists': [5, 7, 7, 9, 12, 9],
    #                                        'rebounds': [11, 8, 10, 6, 6, 5],
    #                                        'gender': [0, 1, 1, 0, 1, 0],
    #                                        'touch_experience': [0, 1, 2, 2, 3, 1]
    #                                        })

    # extrated_features_data.to_pickle(Path_raw_data_BioIdent + '/' + 'EF_' + pickle_names[dataset_index]
    #                                  .replace('arff', 'pickle'))

    X_scaled_data = pd.DataFrame()
    if dataset_index == 0 or dataset_index == 1:
        y = extrated_features_data['user_id']
        X = extrated_features_data.drop(columns=['gender', 'touchscreen_experience_level', 'user_id'])
        X_scaled_data = min_max_normalization(X).round(4)
        X_scaled_data.insert(0, 'user_id', y)
    elif dataset_index == 2:
        y = extrated_features_data['gender']
        X = extrated_features_data.drop(columns=['gender', 'touchscreen_experience_level', 'user_id'])
        X_scaled_data = min_max_normalization(X).round(4)
        X_scaled_data.insert(len(X_scaled_data.columns), 'gender', y)
    elif dataset_index == 3:
        y = extrated_features_data['touchscreen_experience_level']
        X = extrated_features_data.drop(columns=['gender', 'touchscreen_experience_level', 'user_id'])
        X_scaled_data = min_max_normalization(X).round(4)
        X_scaled_data.insert(len(X_scaled_data.columns), 'touchscreen_experience_level', y)
    print(y)
    print(X)
    print(X_scaled_data)

    final_data = X_scaled_data
    arff_name = (pickle_names[dataset_index])
    arff.dump(Path_raw_data_BioIdent + '/' + arff_name,
              final_data.values,
              relation=arff_name,
              names=final_data.columns)

    print('done')
