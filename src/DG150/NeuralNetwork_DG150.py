"""Suppress Tensorflow prints"""
import logging
import os

from src.DG150.DG150_management import Path_DG150_FE, datasets_DG150, PathDG150_Results
from src.EER import compute_eer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
"""/Suppress Tensorflow prints"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import CSVLogger
import seaborn as sn
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


def nn_model(input_dim, output_dim, nodes=40, dropout_rate=None):
    """Create neural network model with two hidden layers"""
    model = Sequential()
    model.add(Dense(nodes, input_dim=input_dim, activation='relu'))
    if dropout_rate: model.add(Dropout(dropout_rate))
    model.add(Dense(nodes, activation='relu'))
    if dropout_rate: model.add(Dropout(dropout_rate))

    if output_dim == 1:
        model.add(Dense(output_dim, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
    else:
        model.add(Dense(output_dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
    return model


def calulate_model(index_dataset, clasx, EPOCHS, nodes):
    print("////////////////")
    # nodes = 300
    # EPOCHS = 5

    # clasx = 0
    X = pd.read_pickle(Path_DG150_FE + '/' + "FE_dataset_pickle_" + datasets_DG150[index_dataset] + ".pkl")
    X['user_id'] = X['user_id'].astype(int)
    y = X['user_id'].values

    SOF_brokens = ["mn_dt", "mx_dt", "am_dt", "qm_dt", "hm_dt", "gm_dt", "md_dt", "rg_dt", "vr_dt", "sd_dt", "sk_dt",
                   "ku_dt", "fq_dt", "tq_dt", "ir_dt", "ma_dt", "mi_dt", "cv_dt", "se_dt", "mn_ft1", "mx_ft1", "am_ft1",
                   "qm_ft1", "hm_ft1", "gm_ft1", "md_ft1", "rg_ft1", "vr_ft1", "sd_ft1", "sk_ft1", "ku_ft1", "fq_ft1",
                   "tq_ft1", "ir_ft1", "ma_ft1", "mi_ft1", "cv_ft1", "se_ft1", "mn_ft2", "mx_ft2", "am_ft2", "qm_ft2",
                   "hm_ft2", "gm_ft2", "md_ft2", "rg_ft2", "vr_ft2", "sd_ft2", "sk_ft2", "ku_ft2", "fq_ft2", "tq_ft2",
                   "ir_ft2", "ma_ft2", "mi_ft2", "cv_ft2", "se_ft2", "mn_ft3", "mx_ft3", "am_ft3", "qm_ft3", "hm_ft3",
                   "gm_ft3", "md_ft3", "rg_ft3", "vr_ft3", "sd_ft3", "sk_ft3", "ku_ft3", "fq_ft3", "tq_ft3", "ir_ft3",
                   "ma_ft3", "mi_ft3", "cv_ft3", "se_ft3", "mn_ft4", "mx_ft4", "am_ft4", "qm_ft4", "hm_ft4", "gm_ft4",
                   "md_ft4", "rg_ft4", "vr_ft4", "sd_ft4", "sk_ft4", "ku_ft4", "fq_ft4", "tq_ft4", "ir_ft4", "ma_ft4",
                   "mi_ft4", "cv_ft4", "se_ft4", "mn_it", "mx_it", "am_it", "qm_it", "hm_it", "gm_it", "md_it", "rg_it",
                   "vr_it", "sd_it", "sk_it", "ku_it", "fq_it", "tq_it", "ir_it", "ma_it", "mi_it", "cv_it", "se_it",
                   "mn_ps", "mx_ps", "am_ps", "qm_ps", "hm_ps", "gm_ps", "md_ps", "rg_ps", "vr_ps", "sd_ps", "sk_ps",
                   "ku_ps", "fq_ps", "tq_ps", "ir_ps", "ma_ps", "mi_ps", "cv_ps", "se_ps"]

    X = X.drop(columns=SOF_brokens)

    print(X.shape)
    print(list(X.columns))

    # input()

    # d = Counter(y)
    # print('--> Class ', d)

    d = Counter(y)
    print('--> Class {} has occurred {} times'.format(clasx, d[clasx]))
    print('--> different classes elements are: ', len(y) - d[clasx])

    y = [0 if val == clasx else 1 for val in y]

    # # OVERSAMPLING
    # summarize class distribution
    print(Counter(y))
    sample = RandomOverSampler(sampling_strategy='minority')
    # sample = RandomUnderSampler(sampling_strategy='majority')
    # fit and apply the transform
    X_over, y_over = sample.fit_resample(X, y)

    ## summarize class distribution
    print(Counter(y_over))

    X = X_over
    Y = pd.get_dummies(y_over).values
    # print(X)
    # print(Y)

    # SENZA OVERSAMPLING
    # Y = pd.get_dummies(y).values

    n_classes = Y.shape[1]
    print('n_classes: ', n_classes)

    print('Running : ', datasets_DG150[index_dataset], clasx, index_dataset, nodes, X.shape)

    # np.argwhere(np.isnan(X))

    # Split data into training and testing data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    # Normalize data with mean 0 and std 1
    X_scaled = normalize(X_train)

    # Add callback that streams epoch results to a csv file
    # https://keras.io/callbacks/
    csv_logger = CSVLogger(
        'models/training_{}_{}_{}_{}.log'.format(datasets_DG150[index_dataset], clasx, index_dataset, nodes))

    # Train the neural network model
    n_features = X.shape[1]
    model = nn_model(n_features, n_classes, nodes, 0.2)
    history = model.fit(X_scaled, Y_train, epochs=EPOCHS, batch_size=5, verbose=1, callbacks=[csv_logger])

    # Serialize model to JSON
    model_json = model.to_json()
    with open('models/model_{}_{}_{}_{}.json'.format(datasets_DG150[index_dataset], clasx, index_dataset, nodes),
              'w') as f:
        f.write(model_json)

    # Serialize weights to HDF5
    model.save_weights('models/model_{}_{}_{}_{}.h5'.format(datasets_DG150[index_dataset], clasx, index_dataset, nodes))
    model.save('models/model_{}_{}_{}_{}.h5'.format(datasets_DG150[index_dataset], clasx, index_dataset, nodes))

    # # TESTING

    X_test_scaled = normalize(X_test)
    score = model.evaluate(X_test_scaled, Y_test, verbose=0)
    print('X_test_scaled loss:', score[0])
    print('X_test_scaled accuracy:', score[1])

    test_prediction = model.predict(X_test_scaled)
    test_prediction = np.argmax(test_prediction, axis=1)

    Y_test_labels = np.argmax(Y_test, axis=1)

    # cm = confusion_matrix(Y_test_labels, test_prediction)
    # # Normalise
    # cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # fig, ax = plt.subplots(figsize=(5, 4))
    # sn.heatmap(cmn, annot=True, fmt='.2f')
    # plt.ylabel('Actual')
    # plt.xlabel('Predicted')
    # plt.show(block=False)

    # dataCM = {'true_classes': Y_test_labels, 'predictions': test_prediction}
    # df_CM = pd.DataFrame(dataCM, columns=['true_classes', 'predictions'])
    # confusion_matrix = pd.crosstab(df_CM['true_classes'], df_CM['predictions'], rownames=['true_classes'],
    #                                colnames=['predictions'])
    # plt.figure(figsize=(10, 10))
    # sn.heatmap(confusion_matrix, linewidths=.5, annot=True)
    # plt.show()

    # print('test_prediction: ', test_prediction)
    # print('Y_test_labels: ', Y_test_labels)
    #
    # d = Counter(test_prediction)
    # print('--> Conuter test_prediction ', d)
    #
    # d = Counter(Y_test_labels)
    # print('--> Conuter Y_test_labels ', d)

    report = classification_report(Y_test_labels, test_prediction)
    print(report)

    fpr, tpr, threshold = roc_curve(Y_test_labels, test_prediction, pos_label=1)
    auc_keras = auc(fpr, tpr)
    print('auc: ', auc_keras)
    eer_point = compute_eer(fpr, tpr, threshold)
    print('EER: ', eer_point[0])

    # plt.figure(1)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr, tpr, label='NN (area = {:.3f})'.format(auc_keras))
    #
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve')
    # plt.legend(loc='best')
    # # plt.plot(tpr)
    # # plt.plot(1 - tpr)
    # # plt.scatter(eer_point[1], eer_point[0])
    # plt.show()

    return eer_point[0], auc_keras


if __name__ == '__main__':
    unique_subjects = np.unique(
        pd.read_pickle(Path_DG150_FE + '/' + "FE_dataset_pickle_" + datasets_DG150[0] + ".pkl")[
            'user_id'].values.astype(int))

    unique_subjects = [0, 1]

    EPOCHS = 10
    NODES = 300
    clasx = 1
    # db = 0     # 0 - 5

    results = pd.DataFrame(columns=['clase', 'eer', 'auc'])

    ix = 0

    # for index_dataset in range(2):
    #     for clasx in unique_subjects:
    #         eer2, auc2 = calulate_model(index_dataset, clasx, EPOCHS, NODES)
    #         print(clasx, eer2, auc2)
    #         results.loc[clasx] = [clasx, eer2, auc2]
    #     results.to_csv(PathDG150_Results + '/' + datasets_DG150[index_dataset] + "_Results_NN" + ".csv")
    #     results.drop(results.index, inplace=True)

    eer2, auc2 = calulate_model(0, clasx, EPOCHS, NODES)
    print(clasx, eer2, auc2)
