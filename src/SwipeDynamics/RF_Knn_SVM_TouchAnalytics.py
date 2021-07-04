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
from sklearn.svm import SVC

from src.EER import evaluateEER2
from src.SwipeDynamics.TouchAnalytics_management import PathFrank_Results
from src.functions import randomforest, KNNx, evaluate, conf_matrix, skf_, evaluateCV
from src.support import clean_dataset, get_df_from_arff
import numpy as np

if __name__ == '__main__':
    "FUNZIONA"

    index = 0
    clasx = 1.0

    data = get_df_from_arff('D:\pycharmProjects\TouchDynamics\datasets\Swipes\Frank\data_arff\dataset.arff')
    data = clean_dataset(data)

    # print(data.head())
    data_Up = data.loc[data['up/down/left/rightflag'] == 1]
    # print(data_Up['up/down/left/rightflag'].head())
    data_Down = data.loc[data['up/down/left/rightflag'] == 2]
    # print(data_Down['up/down/left/rightflag'].head())
    data_Left = data.loc[data['up/down/left/rightflag'] == 3]
    # print(data_Left['up/down/left/rightflag'].head())
    data_Right = data.loc[data['up/down/left/rightflag'] == 4]
    # print(data_Right['up/down/left/rightflag'].head())

    data = data_Right  # data_Left   # data_Down  # data_Up

    subjects = data['subject'].values
    unique_subjects = np.unique(subjects)
    print(unique_subjects)

    # input()

    results = pd.DataFrame(columns=['rf_eer', 'knn_eer', 'svm_eer'])

    for clasx in unique_subjects:
        print("Clasx --> ", clasx)

        # Binarizzazione del dataset
        y = [0 if val == clasx else 1 for val in subjects]

        ### OVERSAMPLIG classe Minoritaria / DOWNSAMPLING classe Maggioritaria
        # summarize class distribution
        # print(Counter(y))

        # define undersample strategy
        # sample = RandomOverSampler(sampling_strategy='minority')
        sample = RandomUnderSampler(sampling_strategy='majority')
        # fit and apply the transform
        X_over, y_over = sample.fit_resample(
            data.drop(columns=['subject', 'docid', 'phoneid', 'changeoffingerorientation']), y)

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
        n_estimators = 100
        randomization = listRandomization[0]
        bootstrap = listBootstrap[0]
        kfolds = 5
        n_neighbors = 2
        Test_size = 0.3
        # seed = 8


        "Train Test split"
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=Test_size, random_state=1)


        "Addestramento classificatori"
        rf = randomforest(X, Y, n_estimators, randomization, bootstrap)
        neigh = KNNx(X, Y, n_neighbors)
        SVM = svm.LinearSVC(dual=False).fit(X, Y)
        # SVM = SVC(kernel='rbf', probability=True, C=10, gamma='scale').fit(X, Y)


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

    results.loc['AVG'] = [results['rf_eer'].mean(), results['knn_eer'].mean(), results['svm_eer'].mean()]
    print("RESULTS AVG = ", results['rf_eer'].mean(), results['knn_eer'].mean(), results['svm_eer'].mean())
    results.to_csv(PathFrank_Results + '/' + "Swipes_Results_Shallow" + ".csv")
    results.drop(results.index, inplace=True)

    # RESULTS AVG =  0.006309526721887615 0.19573906144750342 0.4501502874003291
    # 0.63%    -    19.57%  -       45.01 %

    # RESULTS UP AVG =  0.008286533816425119 0.24711742424242428 0.13242366163425948
    # 0.82%  -      24.71%  -      13.24 %

    # RESULTS DOWN AVG = 0.0067321162850251266 0.20451794561550554 0.2335641403392358
    # 0.67 %     -  20.45%   -      23.35%

    # RESULTS LEFT AVG = 0.008379879385835318 0.21343205564525444 0.18247202999148168
    # 0.83%     -   21.3%   -       18.24%

    # RESULTS RIGHT AVG = 0.005491827232935916 0.1966138984137996 0.21677706796445304
    # 0.54%     -   19.66%  -       21.6%



