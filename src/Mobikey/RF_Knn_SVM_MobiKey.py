import os
import pandas as pd

from src.Mobikey.MobiKey_management import Path_MobiKey, load_data, subjects, Path_MobiKey_Results
from src.core.RF_Knn_SVM import RF_kNN_SVM, listBootstrap, listRandomization, listN_estimators

if __name__ == '__main__':

    datasets_WekaArf = os.listdir(Path_MobiKey)
    print(datasets_WekaArf)

    "Parametri Dataset"
    indice_database_MobiKey = 0  # #  0-1-2-3-4-5
    db_name = datasets_WekaArf[indice_database_MobiKey]

    Test_Size = 0.1

    "Parametri Generali "
    NFold = 10
    Strategy = 'UnderSampler'  # 'OverSampler'

    "Knn"
    n_neighbors = 1

    "Random Forest"
    n_estimators = listN_estimators[3]
    randomization = listRandomization[0]
    bootstrap = listBootstrap[0]

    results = pd.DataFrame(columns=['rf_eer', 'rf_oa', 'rf_ba',
                                    'knn_eer', 'knn_oa', 'knn_ba',
                                    'svm_eer', 'svm_oa', 'svm_ba'])

    for clasx in subjects:
        X, subjects = load_data(indice_database_MobiKey)
        # print(data)

        avgTest_result = RF_kNN_SVM(X.drop(columns=['user_id']), subjects, db_name, clasx, Strategy, NFold, Test_Size,
                                    n_estimators, randomization, bootstrap, n_neighbors)

        results.loc[clasx] = [avgTest_result[0], avgTest_result[1], avgTest_result[2],
                              avgTest_result[3], avgTest_result[4], avgTest_result[5],
                              avgTest_result[6], avgTest_result[7], avgTest_result[8]]

    results.loc['AVG'] = [
        results['rf_eer'].mean(), results['rf_oa'].mean(), results['rf_ba'].mean(),
        results['knn_eer'].mean(), results['knn_oa'].mean(), results['knn_ba'].mean(),
        results['svm_eer'].mean(), results['svm_oa'].mean(), results['svm_ba'].mean(),
    ]
    print("RESULTS AVG = ", results['rf_eer'].mean(), results['rf_oa'].mean(), results['rf_ba'].mean(),
          results['knn_eer'].mean(), results['knn_oa'].mean(), results['knn_ba'].mean(),
          results['svm_eer'].mean(), results['svm_oa'].mean(), results['svm_ba'].mean(),
          )
    results.to_csv(Path_MobiKey_Results + '/' + db_name.replace('.arff', '_Results_' + Strategy + '_Shellow') + ".csv")
