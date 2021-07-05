"""Suppress Tensorflow prints"""
import logging
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
# logging.getLogger('tensorflow').setLevel(logging.FATAL)
# """/Suppress Tensorflow prints"""

import pandas as pd
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
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from src.EER import compute_eer
from src.WekaArf.WekaArf_management import Path_WekaArf, Path_WekaArf_Results, subjects
import glob


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


def calulate_model(key, clasx, EPOCHS, nodes, Test_Size, Strategy):
    from WekaArf_management import load_data

    X, y = load_data(key)
    # print(y)

    d = Counter(y)
    print('--> Class ', d)

    d = Counter(y)

    print('--> Class {} has occurred {} times'.format(clasx, d[clasx]))
    print('--> different classes elements are: ', len(y) - d[clasx])
    y = [0 if val == clasx else 1 for val in y]

    # One hot encoding of target vector

    # summarize class distribution
    print(Counter(y))

    # define  strategy
    if Strategy == 'OverSampler':
        sample = RandomOverSampler(sampling_strategy='minority')
        X_over, y_over = sample.fit_resample(X, y)
        X = X_over
        Y = pd.get_dummies(y_over).values
    elif Strategy == 'UnderSampler':
        sample = RandomUnderSampler(sampling_strategy='majority')
        X_over, y_over = sample.fit_resample(X, y)
        X = X_over
        Y = pd.get_dummies(y_over).values

    # summarize class distribution
    print(Counter(y_over))

    n_classes = Y.shape[1]
    print('n_classes: ', n_classes)

    # nodes = 300
    # EPOCHS = 5

    print('Running : ', db, clasx, key, nodes, X.shape, Strategy)

    # np.argwhere(np.isnan(X))

    # Split data into training and testing data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=Test_Size, random_state=1)

    # Normalize data with mean 0 and std 1
    X_scaled = normalize(X_train)

    # Add callback that streams epoch results to a csv file
    # https://keras.io/callbacks/
    csv_logger = CSVLogger('models/training_{}_{}_{}_{}.log'.format(db, clasx, key, nodes))

    # Train the neural network model
    n_features = X.shape[1]
    model = nn_model(n_features, n_classes, nodes, 0.2)
    history = model.fit(X_scaled, Y_train, epochs=EPOCHS, batch_size=5, verbose=1, callbacks=[csv_logger])

    # Serialize model to JSON
    model_json = model.to_json()
    with open('models/model_{}_{}_{}_{}.json'.format(db, clasx, key, nodes), 'w') as f:
        f.write(model_json)

    # Serialize weights to HDF5
    model.save_weights('models/model_{}_{}_{}_{}.h5'.format(db, clasx, key, nodes))
    model.save('models/model_{}_{}_{}_{}.h5'.format(db, clasx, key, nodes))

    ## test
    print("////////////////")

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

    WekaArff_DBs = os.listdir(Path_WekaArf)  # glob.glob(Path_WekaArf + '\*.arff')

    print(WekaArff_DBs)

    EPOCHS = 10
    NODES = 300
    Test_Size = 0.1
    Strategy = 'OverSampler'  # 'UnderSampler'
    # clasx = 502
    # db = 0     # 0 - 5

    results = pd.DataFrame(columns=['classe', 'eer', 'auc'])

    ix = 0

    for db in WekaArff_DBs:
        for clasx in subjects:
            eer2, auc2 = calulate_model(ix, clasx, EPOCHS, NODES, Test_Size, Strategy)
            print(clasx, eer2, auc2)
            results.loc[clasx] = [clasx, eer2, auc2]
        ix = ix + 1
        results.loc['AVG'] = ['-->', results['eer'].mean(), results['auc'].mean()]
        print("RESULTS AVG = ", results['eer'].mean(), results['auc'].mean())
        results.to_csv(Path_WekaArf_Results + '/' + db + "_Results_" + Strategy + "_NN" + ".csv")
        results.drop(results.index, inplace=True)
    print("-------------------------------")

# CON --> OVERSAMPLING
# 42_users_51 samples_user_17_features_sample.arff
