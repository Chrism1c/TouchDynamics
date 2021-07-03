import glob
from collections import Counter

import matplotlib.pyplot as plt
from scipy.io import arff
import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.svm import OneClassSVM
from src.EER import evaluateEER, evaluateEER2
from src.functions import skf_, bestAVG_fScore, randomforest, conf_matrix, evaluate, KNNx
from src.Mobilekey.MobiKey_management import Path_MobileKEY
from src.support import get_df_from_arff
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import os

MobileKey_DBs = ['MobileKey_all_easy.arff', 'MobileKey_all_logicalstrong.arff', 'MobileKey_all_strong.arff',
                 'MobilKey_secondorder_easy.arff', 'MobilKey_secondorder_logicalstrong.arff',
                 'MobileKey_secondorder_strong.arff']

# index = 0
# clasx = 600

if __name__ == '__main__':

    datasets_MobileKey = os.listdir(Path_MobileKEY)
    print(datasets_MobileKey)

    " ID utenti database"
    subjects = [100, 101, 102, 103, 104, 105, 200, 201, 202, 203, 300, 301, 302, 303, 400, 401, 402, 403, 404, 500, 501,
                502, 503, 600, 601, 602, 603, 604, 605, 700, 701, 702, 800, 801, 802, 900, 1000, 1001, 1002, 1003, 1004,
                1100, 1101, 1103, 1104, 1105, 1200, 1201, 1203, 1204, 1300, 1301, 1302, 1303]

    # subjects = [100, 101]

    # 54

    " Parametri classificatori "
    listBootstrap = [0.5, 0.6, 0.7, 0.8, 0.9]
    listRandomization = ["sqrt", "log2"]
    listN_estimators = [10, 20, 30]
    kfolds = 5
    n_neighbors = 3

    results = pd.DataFrame(columns=['rf_eer', 'knn_eer', 'svm_eer'])

    db = datasets_MobileKey[0]

    for clasx in subjects:
        data = get_df_from_arff(Path_MobileKEY + '/' + db)

        print("---> Clasx: ", clasx, " dataset: ", db)

        data['user_id'] = pd.to_numeric(data['user_id'])
        subjects = data['user_id'].values
        # print(data)
        # print(data.head())

        # Binarizzazione del dataset
        y = [0 if val == clasx else 1 for val in subjects]

        ### OVERSAMPLIG classe Minoritaria / DOWNSAMPLING classe Maggioritaria

        # summarize class distribution
        # print(Counter(y))
        # define undersample strategy
        sample = RandomOverSampler(sampling_strategy='minority')
        # sample = RandomUnderSampler(sampling_strategy='majority')

        # fit and apply the transform
        X_over, y_over = sample.fit_resample(data.iloc[:, :-1], y)

        # summarize class distribution
        # print(Counter(y_over))

        X = X_over
        Y = y_over

        ### SENZA OVERSAMPLING
        # X = data.iloc[:, :-1]
        # Y = y

        # print(Counter(Y))

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

        # "Evaluate sul TestSet"
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

    results.to_csv(Path_MobileKEY + '/' + db.replace('.arff', '_Results') + ".csv")
    results.drop(results.index, inplace=True)
