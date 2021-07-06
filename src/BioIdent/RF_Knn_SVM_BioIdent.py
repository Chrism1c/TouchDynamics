import os
import pandas as pd
import numpy as np
from src.BioIdent.BioIdent_management import Path_BioIdent, load_data, subjects, gender, touch_experience, \
    Path_BioIdent_Results
from src.core.RF_Knn_SVM import RF_kNN_SVM, listBootstrap, listRandomization, listN_estimators

if __name__ == '__main__':

    # datasets_BioIdent = os.listdir(Path_BioIdent)
    # print(datasets_BioIdent)

    "Parametri Dataset"
    # indice_database_BioIdent = 3  # #  0-1-2-3
    # db_name = datasets_BioIdent[indice_database_BioIdent]

    datasets_BioIdent = ['dataset1.arff', 'dataset2.arff', 'dataset3.arff', 'dataset4.arff']

    selected = ['dataset1.arff'] #, 'dataset2.arff']

    Test_Size = 0.1

    "Parametri Generali "
    NFold = 3
    Strategy = 'OverSampler'  # 'UnderSampler'   #

    "Knn"
    n_neighbors = 3

    "Random Forest"
    n_estimators = listN_estimators[3]
    randomization = listRandomization[0]
    bootstrap = listBootstrap[0]

    results = pd.DataFrame(columns=['rf_eer', 'rf_oa', 'rf_ba',
                                    'knn_eer', 'knn_oa', 'knn_ba',
                                    'svm_eer', 'svm_oa', 'svm_ba'])

    # db_index = 0
    for db_name in selected:
        db_index = datasets_BioIdent.index(db_name)
        print("\nindex : ", db_index)
        X, subjects = load_data(db_index)
        print('subjects ', np.unique(subjects))
        for clasx in np.unique(subjects):
            print()
            avgTest_result = RF_kNN_SVM(X, subjects, db_name, clasx, Strategy, NFold, Test_Size,
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
        results.to_csv(Path_BioIdent_Results + '/' + db_name.replace('.arff', '_Results_' + Strategy + '_Shellow') + ".csv")
