import glob
from collections import Counter

import matplotlib.pyplot as plt
from scipy.io import arff
import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from src.EER import evaluateEER, evaluateEER2
from src.functions import randomforest, conf_matrix, evaluate, KNNx, sss_
from src.WekaArf.WekaArf_management import Path_WekaArf, Path_WekaArf_Results, subjects
from src.support import get_df_from_arff, evaluateCV_3Classifiers
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import os

# index = 0
# clasx = 600


if __name__ == '__main__':

    datasets_WekaArf = os.listdir(Path_WekaArf)
    print(datasets_WekaArf)

    "ID DEL DATASET DA UTILIZZARE"
    id_database_WekaArff = 4  # #   0-1-2       3-4
    Test_Size = 0.1

    " Parametri classificatori "
    listBootstrap = [0.5, 0.6, 0.7, 0.8, 0.9]
    listRandomization = ["sqrt", "log2"]
    listN_estimators = [10, 20, 30]
    NFold = 10
    n_neighbors = 3

    n_estimators = 100
    randomization = listRandomization[0]
    bootstrap = listBootstrap[0]

    results = pd.DataFrame(columns=['rf_eer', 'knn_eer', 'svm_eer'])

    db = datasets_WekaArf[id_database_WekaArff]
    for clasx in subjects:
        data = get_df_from_arff(Path_WekaArf + '/' + db)

        print("----> Clasx: ", clasx, " dataset: ", db)

        data['user_id'] = pd.to_numeric(data['user_id'])
        subjects = data['user_id'].values
        # print(data)
        # print(data.head())

        # Binarizzazione del dataset
        y = [0 if val == clasx else 1 for val in subjects]

        ### OVERSAMPLIG classe Minoritaria / DOWNSAMPLING classe Maggioritaria

        # summarize class distribution
        # print(Counter(y))

        sample = RandomOverSampler(sampling_strategy='minority')
        # sample = RandomUnderSampler(sampling_strategy='majority')

        # fit and apply the transform
        X_over, y_over = sample.fit_resample(data.iloc[:, :-1], y)

        # summarize class distribution
        # print(Counter(y_over))

        X = X_over
        Y = y_over

        # ## SENZA NULLA
        # X = data.iloc[:, :-1]
        # Y = y

        # print(Counter(Y))

        "Cross Validation"
        ListXTrain, ListXTest, ListYTrain, ListYTest = sss_(X, pd.Series(Y), NFold, Test_Size)

        avgTest = evaluateCV_3Classifiers(NFold, ListXTrain, ListXTest, ListYTrain, ListYTest,
                                          n_estimators=n_estimators,
                                          randomization=randomization,
                                          bootstrap=bootstrap, n_neighbors=n_neighbors)

        print("Results 10-FoldCV : ", avgTest)
        results.loc[clasx] = [avgTest[0], avgTest[1], avgTest[2]]

    results.loc['AVG'] = [results['rf_eer'].mean(), results['knn_eer'].mean(), results['svm_eer'].mean()]
    print("RESULTS AVG = ", results['rf_eer'].mean(), results['knn_eer'].mean(), results['svm_eer'].mean())
    results.to_csv(Path_WekaArf_Results + '/' + db.replace('.arff', '_Results_Shellow') + ".csv")

#      CON --> OVERSAMPLING
# RESULTS AVG: 42_users_51 samples_user_71_features_sample.arff  =
#       0.0032803328667990344      0.010455223556727315        0.08075266629777908
# RESULTS AVG: 42_users_51 samples_user_3_features_sample.arff   =
#       0.006066763228417365        0.009176431338085473        0.12057513914656771
# RESULTS AVG: 42_users_51 samples_user_71_features_sample.arff  =
#       0.0008242738881836633       0.007254310017467911        0.02844101595981295


#       CON --> DOWNSAMPLING
# RESULTS AVG: 42_users_51 samples_user_17_features_sample.arff  =
#       0.0776190476190476          0.10476190476190472         0.1117063492063492
# RESULTS AVG: 42_users_51 samples_user_3_features_sample.arff   =
#       0.07095238095238093         0.07523809523809523         0.14003968253968255
# RESULTS AVG: 42_users_51 samples_user_71_features_sample.arff  =
#       0.043809523809523805        0.08615079365079364         0.08043650793650793

# SENZA NULLA
# RESULTS AVG: 42_users_51 samples_user_17_features_sample.arff
#   =  0.38013605442176873 0.22129818594104303 0.4218197278911565
# RESULTS AVG: 42_users_51 samples_user_3_features_sample.arff
#   =  0.30032312925170074 0.2137414965986395 0.47989229024943314
# RESULTS AVG: 42_users_51 samples_user_71_features_sample.arff
#   =  0.3067233560090703 0.1528798185941043 0.29393424036281185


#      CON --> OVERSAMPLING
# RESULTS AVG: 42_users_51_inputPatterns_user_41_features_inputPattern.arff
#   =  0.0007730364873222022 0.015770541071292953 0.0
# RESULTS AVG: 42_users_51_inputPatterns_user_71_features_inputPattern.arff
#   =  0.001853470255725897 0.015770541071292953 0.0

#       CON --> DOWNSAMPLING (SVM PERSONAIZZATA)
# RESULTS AVG: 42_users_51_inputPatterns_user_41_features_inputPattern.arff
#   =  0.05023809523809524 0.2371825396825396 0.5
# RESULTS AVG: 42_users_51_inputPatterns_user_71_features_inputPattern.arff
#   =  0.06900793650793652 0.22035714285714283 0.5

#       CON --> DOWNSAMPLING (svm LINEAR)
# RESULTS AVG: 42_users_51_inputPatterns_user_41_features_inputPattern.arff
#   =  0.044285714285714275 0.231468253968254 0.19603174603174603
# RESULTS AVG: 42_users_51_inputPatterns_user_71_features_inputPattern.arff
#   =  0.06936507936507935 0.2224206349206349 0.19555555555555557

