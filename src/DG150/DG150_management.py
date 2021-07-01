from pandas import DataFrame
from scipy.io import arff
import pandas as pd
import numpy as np
import glob

Path_DG150 = 'D:/pycharmProjects/TouchDynamics/datasets/KeystrokeTouch/DG150'
datasets_DG150 = ['Long_DG150', 'Short_DG150']


def merge_data_to_pickle(index):
    selected_DG150 = glob.glob(Path_DG150 + '/' + datasets_DG150[index] + '/*.csv')

    print(selected_DG150)
    df = pd.DataFrame()

    user_id = 0
    for csv in selected_DG150:
        data = pd.read_csv(csv, header=None, names=['sample', 'keydown', 'TAP', 'keyup', 'TAR', 'PressureSize', 'uno'],
                           index_col=False)
        data.insert(0, 'user_id', [user_id] * (data.shape[0]))
        data = data.drop(columns='uno')
        # print(data)
        # print(data.shape)
        user_id += 1
        df = pd.concat([df, data])
        print(df.shape)
    pd.to_pickle(df, Path_DG150 + '\\' + "pickle_" + datasets_DG150[index] + "DG150.pkl")

    # print(data)
    # print(data.shape)

    # data = pd.read_csv(Path_MobileKEY + '/' + datasets_MobileKey[index])

    # clean_dataset(data)

    # print(data.dtypes)
    #
    # data['user_id'] = pd.to_numeric(data['user_id'])
    #
    # return data.drop(columns=['user_id']), data['user_id'].values


def features_extraction(df):
    df_features_extracted = pd.DataFrame()
    dim = df.shape

    print(list(df.iloc[0]))


def get_DT(df, index):
    if df.iloc[index][2] != 'E':
        DT_time = df.iloc[index][5] - df.iloc[index][3]
        print(df.iloc[index][5], " - ", df.iloc[index][3], " = ", DT_time)
        return DT_time
    else:
        return None


def get_FT1(df, index):
    if df.iloc[index][2] != 'E':
        FT1_time = df.iloc[index + 1][3] - df.iloc[index][5]
        print(df.iloc[index + 1][3], " - ", df.iloc[index][5], " = ", FT1_time)
        return FT1_time
    else:
        return None


def get_FT2(df, index):
    if df.iloc[index][2] != 'E':
        FT2_time = df.iloc[index + 1][5] - df.iloc[index][5]
        print(df.iloc[index + 1][5], " - ", df.iloc[index][5], " = ", FT2_time)
        return FT2_time
    else:
        return None


def get_FT3(df, index):
    if df.iloc[index][2] != 'E':
        FT3_time = df.iloc[index + 1][3] - df.iloc[index][3]
        print(df.iloc[index + 1][3], " - ", df.iloc[index][3], " = ", FT3_time)
        return FT3_time
    else:
        return None


def get_FT4(df, index):
    if df.iloc[index][2] != 'E':
        FT4_time = df.iloc[index + 1][5] - df.iloc[index][3]
        print(df.iloc[index + 1][5], " - ", df.iloc[index][3], " = ", FT4_time)
        return FT4_time
    else:
        return None


def get_IT(df, index):
    user_id, sample, keydown, TAP, keyup, TAR, PressureSize = list(df.iloc[index])
    if df.iloc[index][2] != 'E':
        sample_rows = df.loc[(df['sample'] == sample) & (df['user_id'] == user_id)]
        TAP_firstKey = list((sample_rows.head(1)).iloc[0])[3]
        TAR_lastKey = list((sample_rows.tail(1)).iloc[0])[5]
        IT_time = TAR_lastKey - TAP_firstKey
        print(TAR_lastKey, " - ", TAP_firstKey, " = ", IT_time)
        return IT_time
    else:
        return None


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


columns = []

if __name__ == '__main__':
    # data_X, data_Y = load_data(0)
    #
    # print(data_X.columns)
    # print(data_X)
    # print("\n")
    # print(data_Y)
    # print("\n", len(data_Y))

    index = 0  # 0/1
    # merge_data_to_pickle(index)
    df = pd.read_pickle(Path_DG150 + '\\' + "pickle_" + datasets_DG150[index] + "DG150.pkl")
    print(df)
    # print(df.head(30))
    # print(df.tail(30))
    # df.to_csv(Path_DG150 + '\\' + "CSV_" + datasets_DG150[index] + "DG150.csv", index=False)
    print('OK')

    # features_extraction(df)

    row_id = 25498  # 0 - 25499
    print(list(df.iloc[row_id]))
    get_DT(df, row_id)
    get_IT(df, row_id)
