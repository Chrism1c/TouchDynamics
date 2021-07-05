import pandas as pd
from src.support import clean_dataset

Path_TouchAnalytics = '/datasets/Swipes/TouchAnalytics/'
Path_TouchAnalyticsArff = '/datasets/Swipes/TouchAnalytics/data_arff/TouchAnalytics.arff'
Path_TouchAnalytics_Results = 'D:/pycharmProjects/TouchDynamics/doc/Results/TouchAnalytics_Results'

TouchAnalytics_CSV = 'FileName2.csv'
TouchAnalytics_pickle = 'touchalytics.pikle'


def generate_pickle(PathFrank, pickle_fileName):
    data = pd.read_csv(PathFrank + TouchAnalytics_CSV)
    for col in data.columns:
        print(col)
    print(len(data.columns))
    data.to_pickle(PathFrank + '/' + pickle_fileName)


def load_data():
    data = pd.read_pickle(Path_TouchAnalytics + '/' + TouchAnalytics_pickle)
    data = clean_dataset(data)

    # print(data.head())
    data_Up = data.loc[data['up/down/left/right flag'] == 1]
    # print(data_Up['up/down/left/rightflag'].head())
    data_Down = data.loc[data['up/down/left/right flag'] == 2]
    # print(data_Down['up/down/left/rightflag'].head())
    data_Left = data.loc[data['up/down/left/right flag'] == 3]
    # print(data_Left['up/down/left/rightflag'].head())
    data_Right = data.loc[data['up/down/left/right flag'] == 4]
    # print(data_Right['up/down/left/rightflag'].head())

    data['user id'] = data['user id'].astype(int)
    # data = data_Down  # data_Up     # data_Right  # data_Left

    return data.drop(columns=['user id', 'doc id', 'phone id', 'change of finger orientation']), data['user id'].values


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


#
# if __name__ == '__main__':
#     data_l, values = load_data()
#     data_X = data_l['total']
#     data_Y = data_l['user id']
#
#     print(data_X)
#     print("\n")
#     print(data_Y)
#     print("\n", len(values))
