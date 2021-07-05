import pandas as pd
import glob
from src.support import get_df_from_arff

Path_BioIdent = 'D:/pycharmProjects/TouchDynamics/datasets/Swipes/BioIdent/data_arff'
Path_BioIdent_Results = 'D:/pycharmProjects/TouchDynamics/doc/Results/BioIdent_Results'


def load_data(db_index):
    datasets_BioIdent = glob.glob(Path_BioIdent + '/' + '/*.arff')
    # print(datasets_BioIdent[db_index])
    data = get_df_from_arff(datasets_BioIdent[db_index])

    try:
        if db_index == 0 or db_index == 1:
            data['user_id'] = pd.to_numeric(data['user_id'])
            return data.drop(columns=['user_id']), data['user_id'].values
        elif db_index == 2:
            data['gender'] = pd.to_numeric(data['gender'])
            return data.drop(columns=['gender']), data['gender'].values
        elif db_index == 3:
            data['touch_experience'] = pd.to_numeric(data['touch_experience'])
            return data.drop(columns=['touch_experience']), data['touch_experience'].values
    except:
        print("Not valid index")


" ID users DB 1 - 2"
subjects = [1, 8, 22, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
            54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 75, 76, 77, 78, 79, 80, 82,
            83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100]

" gender DB 3"
gender = [0, 1]

" touch_experience 4"
touch_experience = [0, 1, 2, 3]
