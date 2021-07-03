import glob
from collections import Counter

import matplotlib.pyplot as plt
from scipy.io import arff
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from src.DG150.DG150_management import PathDG150_Results, Path_DG150_FE, datasets_DG150
from src.EER import evaluateEER2
from src.SwipeDynamics.TouchAnalytics_management import PathFrank_Results
from src.functions import randomforest, KNNx, evaluate, conf_matrix
from src.support import clean_dataset, get_df_from_arff
import numpy as np

if __name__ == '__main__':

    # index_dataset = 0   # 0-1
    # clasx = 1

    for index_dataset in range(1):
        data = pd.read_pickle(Path_DG150_FE + '/' + "FE_dataset_pickle_" + datasets_DG150[index_dataset] + ".pkl")
        subjects = data['user_id'].values
        unique_subjects = np.unique(subjects)

        print(unique_subjects)
        print(data.head())

        results = pd.DataFrame(columns=['rf_eer', 'knn_eer', 'svm_eer'])

        for clasx in unique_subjects:
            print("Clasx --> ", clasx, " db --> ", datasets_DG150[index_dataset])

            # Binarizzazione del dataset
            y = [0 if val == clasx else 1 for val in subjects]

            ### OVERSAMPLIG classe Minoritaria / DOWNSAMPLING classe Maggioritaria
            # summarize class distribution
            # print(Counter(y))

            # define undersample strategy
            sample = RandomOverSampler(sampling_strategy='minority')
            # sample = RandomUnderSampler(sampling_strategy='majority')
            # fit and apply the transform
            X_over, y_over = sample.fit_resample(data.drop(columns=['user_id']), y)

            # summarize class distribution
            print(Counter(y_over))

            X = X_over
            Y = y_over

            ### SENZA OVERSAMPLING
            # X = data.iloc[:, :-1]
            # Y = y

            # print(Counter(Y))

            " Parametri classificatori "

            listBootstrap = [0.5, 0.6, 0.7, 0.8, 0.9]
            listRandomization = ["sqrt", "log2"]
            listN_estimators = [10, 20, 30]
            kfolds = 5
            n_neighbors = 3

            "Train Test split"
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

            n_estimators = listN_estimators[0]
            randomization = listRandomization[0]
            bootstrap = listBootstrap[0]

            "Addestramento classificatori"
            rf = randomforest(X, Y, n_estimators, randomization, bootstrap)
            neigh = KNNx(X, Y, n_neighbors)
            SVM = svm.LinearSVC(dual=False).fit(X, Y)

            "Predizioni dei classificatori sul TestSet"
            res_onTest_rf = rf.predict(X_test)
            res_onTest_knn = neigh.predict(X_test)
            res_onTest_SVM = SVM.predict(X_test)

            "Evaluate sul TestSet"
            # metrics_rf = evaluate(X_test, Y_test, rf)
            # metrics_knn = evaluate(X_test, Y_test, neigh)
            # metrics_svm = evaluate(X_test, Y_test, SVM)
            # print("\nMetrics RF ----> ", metrics_rf)
            # print("\nMetrics Knn ----> ", metrics_knn)
            # print("\nMetrics Svm ----> ", metrics_svm)

            " EER "
            # print(Y_test, "   ", res_onTest_rf)
            rf_eer, rf_roc_auc = evaluateEER2(Y_test, res_onTest_rf)
            print("Rf eer_threshold: ", rf_eer)

            # print(Y_test, "   ", res_onTest_knn)
            knn_eer, knn_roc_auc = evaluateEER2(Y_test, res_onTest_knn)
            print("knn eer_threshold: ", knn_eer)

            svm_eer, svmroc_auc = evaluateEER2(Y_test, res_onTest_SVM)
            print("svm eer_threshold: ", svm_eer)

            results.loc[clasx] = [rf_eer[0], knn_eer[0], svm_eer[0]]

            "Output delle tre matrici di confusione"
            # conf_matrix(Y_test, res_onTest_rf)
            # conf_matrix(Y_test, res_onTest_knn)
            # conf_matrix(Y_test, res_onTest_SVM)

            "Classification Report"
            # report = classification_report(Y_test, res_onTest_rf)
            # print(report)
            #
            # report = classification_report(Y_test, res_onTest_knn)
            # print(report)
            #
            # report = classification_report(Y_test, res_onTest_SVM)
            # print(report)

        results.to_csv(PathDG150_Results + '/' + "DG150_" + datasets_DG150[index_dataset] + "_Results_Shallow" + ".csv")
        results.drop(results.index, inplace=True)
