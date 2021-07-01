from pandas import DataFrame
from scipy.io import arff
import pandas as pd
import numpy as np
import glob

Path_MobileKEY = 'D:\pycharmProjects\TouchDynamics\datasets\KeystrokeTouch\MobileKey'



def get_df_from_arff(arff_path):
    from scipy.io import arff
    import pandas as pd
    try:
        data = arff.loadarff(arff_path)
        df = pd.DataFrame(data[0])

        # print(df.head())
        return df
    except:
        print("Path is not valid")



def load_data(index):
    datasets_MobileKey = glob.glob(Path_MobileKEY + '/' + '/*.arff')

    print(datasets_MobileKey[0])

    data = get_df_from_arff(datasets_MobileKey[index])

    print(data)
    print(data.shape)

    # data = pd.read_csv(Path_MobileKEY + '/' + datasets_MobileKey[index])

    # clean_dataset(data)

    print(data.dtypes)

    data['user_id'] = pd.to_numeric(data['user_id'])

    return data.drop(columns=['user_id']), data['user_id'].values


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

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
