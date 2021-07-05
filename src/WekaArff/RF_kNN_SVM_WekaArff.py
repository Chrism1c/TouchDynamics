import os
import pandas as pd

from src.WekaArff.WekaArf_management import Path_WekaArff, load_data, subjects, Path_WekaArff_Results
from src.core.RF_Knn_SVM import RF_kNN_SVM, listBootstrap, listRandomization, listN_estimators

if __name__ == '__main__':

    datasets_WekaArf = os.listdir(Path_WekaArff)
    print(datasets_WekaArf)

    "Parametri Dataset"
    indice_database_WekaArff = 4  # #  0-1-2  |   3-4
    db_name = datasets_WekaArf[indice_database_WekaArff]

    Test_Size = 0.1

    "Parametri Generali "
    NFold = 10
    Strategy = 'OverSampler'  # 'UnderSampler'

    "Knn"
    n_neighbors = 3

    "Random Forest"
    n_estimators = listN_estimators[3]
    randomization = listRandomization[0]
    bootstrap = listBootstrap[0]

    results = pd.DataFrame(columns=['rf_eer', 'knn_eer', 'svm_eer'])

    for clasx in subjects:
        X, subjects = load_data(indice_database_WekaArff)
        # print(data)

        avgTest_result = RF_kNN_SVM(X.drop(columns=['user_id']), subjects, db_name, clasx, Strategy, NFold, Test_Size,
                                    n_estimators, randomization, bootstrap, n_neighbors)

        results.loc[clasx] = [avgTest_result[0], avgTest_result[1], avgTest_result[2]]

    results.loc['AVG'] = [results['rf_eer'].mean(), results['knn_eer'].mean(), results['svm_eer'].mean()]
    print("RESULTS AVG = ", results['rf_eer'].mean(), results['knn_eer'].mean(), results['svm_eer'].mean())
    results.to_csv(Path_WekaArff_Results + '/' + db_name.replace('.arff', '_Results_Shellow') + ".csv")

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
