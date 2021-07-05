import os
import pandas as pd

from src.Mobikey.MobiKey_management import Path_MobiKey, load_data, subjects, Path_MobiKey_Results
from src.core.RF_Knn_SVM import RF_kNN_SVM, listBootstrap, listRandomization, listN_estimators

if __name__ == '__main__':

    datasets_WekaArf = os.listdir(Path_MobiKey)
    print(datasets_WekaArf)

    "Parametri Dataset"
    indice_database_MobiKey = 4  # #  0-1-2-3-4-5
    db_name = datasets_WekaArf[indice_database_MobiKey]

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
        X, subjects = load_data(indice_database_MobiKey)
        # print(data)

        avgTest_result = RF_kNN_SVM(X.drop(columns=['user_id']), subjects, db_name, clasx, Strategy, NFold, Test_Size,
                                    n_estimators, randomization, bootstrap, n_neighbors)

        results.loc[clasx] = [avgTest_result[0], avgTest_result[1], avgTest_result[2]]

    results.loc['AVG'] = [results['rf_eer'].mean(), results['knn_eer'].mean(), results['svm_eer'].mean()]
    print("RESULTS AVG = ", results['rf_eer'].mean(), results['knn_eer'].mean(), results['svm_eer'].mean())
    results.to_csv(Path_MobiKey_Results + '/' + db_name.replace('.arff', '_Results_Shellow') + ".csv")
