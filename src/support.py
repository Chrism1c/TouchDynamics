from sklearn import svm

from src.EER import evaluateEER2
from src.functions import randomforest, KNNx


def get_df_from_arff(arff_path):
    from scipy.io import arff
    import pandas as pd
    try:
        data = arff.loadarff(arff_path)
        df = pd.DataFrame(data[0])

        # print(df.head())
        return df
    except:
        print("Path is not valid")


def clean_dataset(df):
    import pandas as pd
    import numpy as np
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def evaluateCV_3Classifiers(folds, ListXTrain, ListXTest, ListyTrain, ListyTest, n_estimators=100, randomization="sqrt",
                            bootstrap=0.5, n_neighbors=3):
    avgTest = [0.0, 0.0, 0.0]

    for i in range(folds):
        "Addestramento classificatori"
        rf = randomforest(ListXTrain[i], ListyTrain[i], n_estimators, randomization, bootstrap)
        neigh = KNNx(ListXTrain[i], ListyTrain[i], n_neighbors)
        # SVM = svm.LinearSVC(dual=False).fit(ListXTrain[i], ListyTrain[i])         # Normalmente

        SVM = svm.SVC(C=10.55, gamma=1.86).fit(ListXTrain[i], ListyTrain[i])        # Per il 41 Features
        # SVM = svm.SVC(C=7.46, gamma=0.25).fit(ListXTrain[i], ListyTrain[i])       # Per il 71 Features

        "EER"
        rf_eer, rf_roc_auc = evaluateEER2(ListyTest[i], rf.predict(ListXTest[i]))
        # print("Rf eer_threshold: ", rf_eer)
        knn_eer, knn_roc_auc = evaluateEER2(ListyTest[i], neigh.predict(ListXTest[i]))
        # print("knn eer_threshold: ", knn_eer)
        svm_eer, svmroc_auc = evaluateEER2(ListyTest[i], SVM.predict(ListXTest[i]))
        # print("svm eer_threshold: ", svm_eer)

        avgTest[0] += rf_eer[0]
        avgTest[1] += knn_eer[0]
        avgTest[2] += svm_eer[0]

        # print(avgTest)
        # print(i)

    for j in range(0, len(avgTest)):
        avgTest[j] = avgTest[j] / folds

    return avgTest
