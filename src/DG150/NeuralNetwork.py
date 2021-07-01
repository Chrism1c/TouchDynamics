"""Suppress Tensorflow prints"""
import logging
import os

from imblearn.under_sampling import RandomUnderSampler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
"""/Suppress Tensorflow prints"""

import pandas as pd
from srcOLD import loader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import CSVLogger
import seaborn as sn
import matplotlib.pyplot as plt
from collections import Counter


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


def calulate_model(key, clasx):
    from DG150_management import load_data

    X, y = load_data(key)
    # print(y)

    d = Counter(y)
    print('--> Class ', d)

    d = Counter(y)

    print('--> Class {} has occurred {} times'.format(clasx, d[clasx]))
    print('--> different classes elements are: ', len(y) - d[clasx])
    y = [clasx if val == clasx else 1 for val in y]

    # One hot encoding of target vector

    # example of random undersampling to balance the class distribution
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
    # summarize class distribution
    print(Counter(y))
    # define undersample strategy
    undersample = RandomOverSampler(sampling_strategy='minority')
    # fit and apply the transform
    X_over, y_over = undersample.fit_resample(X, y)
    # summarize class distribution
    print(Counter(y_over))

    X = X_over
    # print(X)
    print(X.shape)
    Y = pd.get_dummies(y_over).values
    # print(Y)
    print(Y.shape)

    n_classes = Y.shape[1]
    print('n_classes: ', n_classes)

    nodes = 300
    EPOCHS = 5

    print('Running : ', key, nodes, X.shape)

    # np.argwhere(np.isnan(X))

    # Split data into training and testing data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    # Normalize data with mean 0 and std 1
    X_scaled = normalize(X_train)

    # Add callback that streams epoch results to a csv file
    # https://keras.io/callbacks/
    csv_logger = CSVLogger('models/training_{}_{}.log'.format(key, nodes))

    # Train the neural network model
    n_features = X.shape[1]
    model = nn_model(n_features, n_classes, nodes, 0.2)
    history = model.fit(X_scaled, Y_train, epochs=EPOCHS, batch_size=5, verbose=1, callbacks=[csv_logger])

    # Serialize model to JSON
    model_json = model.to_json()
    with open('model_{}_{}.json'.format(key, nodes), 'w') as f:
        f.write(model_json)

    # Serialize weights to HDF5
    model.save_weights('models/model_{}_{}.h5'.format(key, nodes))
    model.save('models/full_model_{}_{}.h5'.format(key, nodes))

    ## test
    print("////////////////")

    X_test_scaled = normalize(X_test)
    score = model.evaluate(X_test_scaled, Y_test, verbose=0)
    print('X_test_scaled loss:', score[0])
    print('X_test_scaled accuracy:', score[1])

    test_prediction = model.predict(X_test_scaled)
    test_prediction = np.argmax(test_prediction, axis=1)

    Y_test_labels = np.argmax(Y_test, axis=1)

    cm = confusion_matrix(Y_test_labels, test_prediction)
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(5, 4))
    sn.heatmap(cmn, annot=True, fmt='.2f')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)

    # dataCM = {'true_classes': Y_test_labels, 'predictions': test_prediction}
    # df_CM = pd.DataFrame(dataCM, columns=['true_classes', 'predictions'])
    # confusion_matrix = pd.crosstab(df_CM['true_classes'], df_CM['predictions'], rownames=['true_classes'],
    #                                colnames=['predictions'])
    # plt.figure(figsize=(10, 10))
    # sn.heatmap(confusion_matrix, linewidths=.5, annot=True)
    # plt.show()

    print('test_prediction: ', test_prediction)
    print('Y_test_labels: ', Y_test_labels)

    d = Counter(test_prediction)
    print('--> Conuter test_prediction ', d)

    d = Counter(Y_test_labels)
    print('--> Conuter Y_test_labels ', d)

    report = classification_report(Y_test_labels, test_prediction)
    print(report)

    fpr, tpr, threshold = roc_curve(Y_test_labels, test_prediction, pos_label=1)
    auc_keras = auc(fpr, tpr)
    print('auc: ', auc_keras)
    eer_point = compute_eer(fpr, tpr, threshold)
    print('EER: ', eer_point[0])

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='NN (area = {:.3f})'.format(auc_keras))

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    # plt.plot(tpr)
    # plt.plot(1 - tpr)
    # plt.scatter(eer_point[1], eer_point[0])
    plt.show()

    return eer_point[0], auc_keras


# def test_single_model(key, clasx):
#     from mobilekey_management import load_data
#
#     data, y = load_data(key)
#     # One hot encoding of target vector
#
#     # print(list(set(y)))
#     d = Counter(y)
#     print('--> Class {} has occurred {} times'.format(clasx, d[clasx]))
#     print('--> different classes elements are: ', len(y) - d[clasx])
#
#     y = [0 if val == clasx else 1 for val in y]
#     # print(y)
#
#     # print(list(set(y)))
#     d = Counter(y)
#     print('--> Class ' + str(clasx) + ' occurred {} times'.format(d[0]))
#     print('--> different classe occurred {} times'.format(len(y) - d[clasx]))
#
#     Y = pd.get_dummies(y).values
#     n_classes = Y.shape[1]
#     nodes = 300
#
#     print("n_classes: ", str(n_classes))
#     print("nodes: ", str(nodes))
#
#     # corregere proporzioni
#
#     X_train, X_test, Y_train, Y_test = train_test_split(data, Y, test_size=0.2, random_state=1, stratify=y)
#
#     # print(f"Training target statistics: {Counter(Y_train)}")
#     # print(f"Testing target statistics: {Counter(Y_test)}")
#
#     # Normalize data with mean 0 and std 1
#     X_test_scaled = normalize(X_test)
#
#     model = load_model('models/full_model_{}_{}'.format(key, nodes) + '.h5')
#
#     score = model.evaluate(X_test_scaled, Y_test, verbose=0)
#     print('X_test_scaled loss:', score[0])
#     print('X_test_scaled accuracy:', score[1])
#
#     test_prediction = model.predict(X_test_scaled)
#     test_prediction = np.argmax(test_prediction, axis=1)
#
#     Y_test_labels = np.argmax(Y_test, axis=1)
#
#     dataCM = {'true_classes': Y_test_labels, 'predictions': test_prediction}
#     df_CM = pd.DataFrame(dataCM, columns=['true_classes', 'predictions'])
#     confusion_matrix = pd.crosstab(df_CM['true_classes'], df_CM['predictions'], rownames=['true_classes'],
#                                    colnames=['predictions'])
#     print(confusion_matrix)
#     plt.figure(figsize=(4, 4))
#     sn.heatmap(confusion_matrix, linewidths=.5, annot=True)
#     plt.show()
#
#     print('test_prediction: ', test_prediction)
#     print('Y_test_labels: ', Y_test_labels)
#
#     d = Counter(test_prediction)
#     print('--> Class 0 occurred {} times'.format(d[0]))
#     print('--> Class 1 occurred {} times'.format(len(y) - d[clasx]))
#
#     d = Counter(Y_test_labels)
#     print('--> Class 0 occurred {} times'.format(d[0]))
#     print('--> Class 1 occurred {} times'.format(len(y) - d[clasx]))
#
#     report = classification_report(Y_test_labels, test_prediction)
#     print(report)
#
#     fpr, tpr, threshold = roc_curve(Y_test_labels, test_prediction, pos_label=1)
#     auc_keras = auc(fpr, tpr)
#     print('auc: ', auc_keras)
#     eer_point = compute_eer(fpr, tpr, threshold)
#     print('EER: ', eer_point[0])
#
#     plt.figure(1)
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.plot(fpr, tpr, label='NN 1vsAll (area = {:.3f})'.format(auc_keras))
#
#     plt.xlabel('False positive rate')
#     plt.ylabel('True positive rate')
#     plt.title('ROC curve')
#     plt.legend(loc='best')
#     # plt.plot(tpr)
#     # plt.plot(1 - tpr)
#     # plt.scatter(eer_point[1], eer_point[0])
#     plt.show()


def compute_eer(fpr, tpr, thresholds):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    return eer, thresholds[min_index]


if __name__ == '__main__':
    roc_aucs = []

    # eer, auc = calulate_model(3, 600)
    eer2, auc2 = calulate_model(3, 502)

    # np.mean([eer, eer2]), np.std([eer, eer2]), np.mean([auc, auc2]), np.std([auc, auc2])

    print(eer2, auc2)
