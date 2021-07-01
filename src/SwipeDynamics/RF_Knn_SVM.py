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

from Mobilekey.EER import evaluateEER2
from Mobilekey.functions import randomforest, KNNx, evaluate, conf_matrix
from Mobilekey.mobilekey_management import get_df_from_arff

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


"Random Forest"

index = 0
clasx = 1.0


data = get_df_from_arff('D:\pycharmProjects\TouchDynamics\datasets\Swipes\Frank\data_arff\dataset.arff')
data = clean_dataset(data)
subjects = data['subject'].values

print(data)
print(data.head())


# Binarizzazione del dataset
y = [0 if val == clasx else 1 for val in subjects]

# OVERSAMPLIG classe Minoritaria / DOWNSAMPLING classe Maggioritaria

# example of random undersampling to balance the class distribution
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# summarize class distribution
print(Counter(y))
# define undersample strategy
sample = RandomOverSampler(sampling_strategy='minority') # RandomUnderSampler(sampling_strategy='majority')   #
# fit and apply the transform
X_over, y_over = sample.fit_resample(data.drop(columns=['subject', 'docid', 'phoneid', 'changeoffingerorientation']), y)
# summarize class distribution
print(Counter(y_over))

independentList = list(data.columns)
print(independentList)

X = X_over
Y = y_over

# print(X)
# print(Y)

# X = data.iloc[:, :-1]
# Y = y

print(Counter(Y))

" Parametri classificatori "

listBootstrap = [0.5, 0.6, 0.7, 0.8, 0.9]
listRandomization = ["sqrt", "log2"]
listN_estimators = [10, 20, 30]
kfolds = 5
n_neighbors = 3
# seed = 8

"Train Test split"
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

n_estimators = listN_estimators[0]
randomization = listRandomization[0]
bootstrap = listBootstrap[0]

"Addestramento classificatori"
rf = randomforest(X, Y, n_estimators, randomization, bootstrap)
neigh = KNNx(X, Y, n_neighbors)
SVM = svm.LinearSVC().fit(X, Y)


"Predizioni dei classificatori sul TestSet"
res_onTest_rf = rf.predict(X_test)
res_onTest_knn = neigh.predict(X_test)
res_onTest_SVM = SVM.predict(X_test)

"Evaluate sul TestSet"
metrics_rf = evaluate(X_test, Y_test, rf)
metrics_knn = evaluate(X_test, Y_test, neigh)
metrics_svm = evaluate(X_test, Y_test, SVM)
print("\nMetrics RF ----> ", metrics_rf)
print("\nMetrics Knn ----> ", metrics_knn)
print("\nMetrics Svm ----> ", metrics_svm)

" EER "
# print(Y_test, "   ", res_onTest_rf)
rf_eer, rf_roc_auc = evaluateEER2(Y_test, res_onTest_rf)
print("Rf eer_threshold: ", rf_eer)

# print(Y_test, "   ", res_onTest_knn)
knn_eer, knn_roc_auc = evaluateEER2(Y_test, res_onTest_knn)
print("knn eer_threshold: ", knn_eer)

svm_eer, svmroc_auc = evaluateEER2(Y_test, res_onTest_SVM)
print("svm eer_threshold: ", svm_eer)


"Output delle tre matrici di confusione"
conf_matrix(Y_test, res_onTest_rf)
conf_matrix(Y_test, res_onTest_knn)
conf_matrix(Y_test, res_onTest_SVM)

"Classification Report"
report = classification_report(Y_test, res_onTest_rf)
print(report)

report = classification_report(Y_test, res_onTest_knn)
print(report)

report = classification_report(Y_test, res_onTest_SVM)
print(report)




