import os
import pandas as pd
import numpy as np

Pre_Path = "D:/pycharmProjects/"
Path_raw_data_BioIdent = Pre_Path + "TouchDynamics/datasets/Swipes/BioIdent/BIOIDENT_RAWDATA"

user_id_d1_d2 = [1, 8, 22, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 75, 76, 77, 78, 79,
                 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100]
gender_d3 = [0, 1]
touch_experience_d4 = [0, 1, 2, 3]

DEBUG = True


def get_stroke(df, index):
    # print("Key down ", list(df.iloc[index]))

    z = 0
    stroke_points = pd.DataFrame(
        columns=['device_id', 'user_id', 'doc_type', 'time', 'action', 'phone_orientaton', 'x_coor', 'y_coor',
                 'pressure', 'finger_area'])
    if df.iloc[index]['action'] == 0:
        print("== 0")
        stroke_points.loc[z] = df.loc[index]
        z = z + 1
        j = index+1
        while True:

            if df.iloc[j][4] == 1:
                print("== 1")
                # stroke_points.loc[z] = df.loc[j]
                if stroke_points.shape[0] >= 3:
                    return stroke_points
                else:
                    return None
            else:
                stroke_points.loc[z] = df.loc[index + 1]
                print("== 2")
            z = z + 1
            j = j +1
    else:
        return None


def get_largest_deviation_from_end_to_end(stroke_points, start_x, start_y, stop_x, stop_y):
    norm = np.linalg.norm

    p1 = np.array([start_x, start_y])
    p2 = np.array([stop_x, stop_y])

    largest_deviation_from_end_to_end = -1
    # per tutti i punti in stroke_points
    for index, row in stroke_points.iterrows():
        p3 = np.array([row['x_coor'], row['y_coor']])
        deviation_from_end_to_end = np.abs(norm(np.cross(p2 - p1, p1 - p3))) / norm(p2 - p1)
        if deviation_from_end_to_end >= largest_deviation_from_end_to_end:
            largest_deviation_from_end_to_end = deviation_from_end_to_end

    return largest_deviation_from_end_to_end


def get_average_direction(stroke_points):
    # # rimuovo l'ultima riga
    # stroke_points = stroke_points.iloc[:-1, :]
    sum_direction = 0
    for index, row in stroke_points.iterrows():
        if index < stroke_points.shape[0] - 1:
            start_x = row['x_coor']
            start_y = row['y_coor']
            stop_x = stroke_points.iloc[index + 1][6]
            stop_y = stroke_points.iloc[index + 1][7]
            # print(" stop_y . start_y ", stop_y, start_y)
            # print(" stop_x . start_x ", stop_x, start_x)
            # print(row['action'], stroke_points.iloc[index + 1]['action'])
            sum_direction = sum_direction + ((stop_y - start_y) / (stop_x - start_x))

    return sum_direction / (stroke_points.shape[0] - 1)


def get_lengthOfTrajectory(stroke_points):
    lenght_of_trajectory = 0
    for index, row in stroke_points.iterrows():
        if index < stroke_points.shape[0] - 1:
            start_x = row['x_coor']
            start_y = row['y_coor']
            stop_x = stroke_points.iloc[index + 1][6]
            stop_y = stroke_points.iloc[index + 1][7]
            lenght_of_trajectory = lenght_of_trajectory + np.sqrt(
                np.power((stop_x - start_x), 2) + np.power((stop_y - start_y), 2))
            # print("index: ", index, " index+1: ", index+1)
        # else:
        #     print("finito")

    return lenght_of_trajectory


def get_average_velocity(stroke_points):
    sum_velocity = 0
    for index, row in stroke_points.iterrows():
        if index < stroke_points.shape[0] - 1:
            start_x = row['x_coor']
            start_y = row['y_coor']
            start_time = row['time']
            stop_x = stroke_points.iloc[index + 1][6]
            stop_y = stroke_points.iloc[index + 1][7]
            stop_time = stroke_points.iloc[index + 1][3]
            # Velocità = Spazio / Tempo
            velocity = (np.sqrt(np.power((stop_x - start_x), 2) + np.power((stop_y - start_y), 2))) / (
                    stop_time - start_time)
            sum_velocity = sum_velocity + velocity
            # print("index: ", index, " index+1: ", index+1)
        # else:
        #     print("finito")

    return sum_velocity / stroke_points.shape[0] - 1


def get_mid_stroke_pressure_and_area_covered(stroke_points):
    mid_point = round(stroke_points.shape[0] / 2)
    # print(mid_point)
    # print(list(stroke_points.iloc[mid_point]))
    return stroke_points.iloc[mid_point][8], stroke_points.iloc[mid_point][9]


# SLOPE
# https://socratic.org/questions/if-a-line-segment-has-endpoints-2-5-and-1-1-what-is-its-slope

# def get_stroke_duration(df, index):
#     device_id, user_id, doc_type, time, action, phone_orientaton, x_coor, y_coor, pressure, finger_area, l = list(
#         df.iloc[index])
#     # print("Key down ", list(df.iloc[index]))
#     if df.iloc[index][4] == 0:
#         # print('Zero')
#         j = index + 1
#         count = 1
#         while True:
#             if df.iloc[j][4] == 1:
#                 # print("Key up ", list(df.iloc[j]), j, " count ", count)
#                 if count >= 3:
#                     return df.iloc[j][3] - time, j, df.iloc[j]
#                 else:
#                     print("to small --> count ", count)
#                     return 'to small', j, df.iloc[j]
#             j = j + 1
#             count = count + 1
#     else:
#         return 'jumped', 'no'


def extract_features_d1(rawdataDF):
    ix = 0
    new_DF = pd.DataFrame(columns=column_names)

    for index, row in rawdataDF.iterrows():
        stroke_points = get_stroke(rawdataDF, index)
        if stroke_points is not None and stroke_points.shape[0] > 3:
            print(stroke_points)
            # print(stroke_points['action'])
            # input()

            try:
                user_id = stroke_points['user_id'][0]
                stroke_duration = list((stroke_points.tail(1)).iloc[0])[3] - list((stroke_points.head(1)).iloc[0])[3]
                start_x = list((stroke_points.head(1)).iloc[0])[6]
                start_y = list((stroke_points.head(1)).iloc[0])[7]
                stop_x = list((stroke_points.tail(1)).iloc[0])[6]
                stop_y = list((stroke_points.tail(1)).iloc[0])[7]
                direct_end_to_end_distance = np.sqrt(np.power((stop_x - start_x), 2) + np.power((stop_y - start_y), 2))

                meanResultantLength = 0

                # horizontal = "left" if start_x > stop_x else "right"
                # vertical = "down" if start_y > stop_y else "up"
                horizontal = 2 if start_x > stop_x else 3
                vertical = 1 if start_y > stop_y else 0
                up_down_left_right = horizontal if abs(start_x - stop_x) > abs(start_y - stop_y) else vertical

                direction_of_end_to_end_line = (stop_y - start_y) / (stop_x - start_x)

                largest_deviation_from_end_to_end = get_largest_deviation_from_end_to_end(stroke_points, start_x, start_y,
                                                                                          stop_x, stop_y)

                average_direction = get_average_direction(stroke_points)

                length_of_trajectory = get_lengthOfTrajectory(stroke_points)

                average_velocity = get_average_velocity(stroke_points)  # 'average_velocity'

                mid_stroke_pressure, mid_stroke_area_covered = get_mid_stroke_pressure_and_area_covered(stroke_points)

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
                    mid_stroke_area_covered
                ]
                # print(features)
                new_DF.loc[ix] = features
                print(new_DF.shape[0])
                ix = ix + 1
            except:
                print("fail")
    return new_DF


if __name__ == '__main__':
    print("")

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
                    'mid_stroke_area_covered']

    new_DF = pd.DataFrame(columns=column_names)
    i = 0

    filesCSV = ['devices.csv', 'rawdata.csv', 'users.csv']
    print(" filesCSV ", filesCSV)
    # devicesDF = pd.read_csv(Path_raw_data_BioIdent + '/' + filesCSV[0])
    rawdataDF = pd.read_csv(Path_raw_data_BioIdent + '/' + filesCSV[1])
    # usersDF = pd.read_csv(Path_raw_data_BioIdent + '/' + filesCSV[2])

    rawdataDF = rawdataDF.drop(columns=['Unnamed: 10'])
    print(rawdataDF)
    print(rawdataDF.columns)

    #########################################################################à

    new_DF = extract_features_d1(rawdataDF)
    print(new_DF)

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()

    loan = pd.DataFrame(scaler.fit_transform(new_DF),
                        columns=new_DF.columns, index=new_DF.index)
    print(loan)
    loan.to_csv(Path_raw_data_BioIdent + '/' + "shit.csv")
