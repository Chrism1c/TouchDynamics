from collections import Counter

import pandas as pd
from src.functions import sss_, evaluateCV_3Classifiers
from src.support import data_strategy

listBootstrap = [0.5, 0.6, 0.7, 0.8, 0.9]
listRandomization = ["sqrt", "log2"]
listN_estimators = [10, 20, 30, 100]


def RF_kNN_SVM(X, subjects, db_name, clasx, Strategy, NFold, Test_Size, n_estimators, randomization, bootstrap,
               n_neighbors):
    print('Running ---> | ({}) database | ({}) class  | ({}) data strategy '.format(db_name, clasx, Strategy))

    print(Counter(subjects))
    "Binarize dataset"

    y = [0 if val == clasx else 1 for val in subjects]

    print(Counter(y))

    # OVERSAMPLIG classe Minoritaria / DOWNSAMPLING classe Maggioritaria
    X, Y = data_strategy(X, y, Strategy)

    "Cross Validation"
    ListXTrain, ListXTest, ListYTrain, ListYTest = sss_(X, pd.Series(Y), NFold, Test_Size)

    print("Evaluation on Folds")

    "Evaluation on Folds"
    avgTest_result = evaluateCV_3Classifiers(NFold, ListXTrain, ListXTest, ListYTrain, ListYTest,
                                             n_estimators=n_estimators,
                                             randomization=randomization,
                                             bootstrap=bootstrap, n_neighbors=n_neighbors)

    print("Results :", avgTest_result)
    return avgTest_result
