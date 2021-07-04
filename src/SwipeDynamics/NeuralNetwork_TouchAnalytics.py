"""Suppress Tensorflow prints"""
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
"""/Suppress Tensorflow prints"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report, roc_curve, auc
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import CSVLogger
import seaborn as sn
import matplotlib.pyplot as plt
from collections import Counter
from TouchAnalytics_management import load_data, PathFrank_Results
from src.EER import compute_eer
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


def calulate_model(clasx, nodes, EPOCHS):
    data, y = load_data()
    # print(y)

    d = Counter(y)
    # print('--> Class {} has occurred {} times'.format(clasx, d[clasx]))
    # print('--> different classes elements are: ', len(y) - d[clasx])
    y = [0 if val == clasx else 1 for val in y]

    # d = Counter(y)
    # print('--> Class 0 occurred {} times'.format(d[0]))
    # print('--> diff Class occurred {} times'.format(d))

    # summarize class distribution
    # print(Counter(y))
    # define undersample strategy
    sample = RandomOverSampler(sampling_strategy='minority')
    # fit and apply the transform
    X_over, y_over = sample.fit_resample(data['total_dropped'], y)
    # summarize class distribution
    # print(Counter(y_over))

    X = X_over
    # print(X)
    Y = pd.get_dummies(y_over).values
    # print(Y)

    # d = Counter(y_over)
    # print('--> Counter(y) {}'.format(d))

    # Y = pd.get_dummies(y).values
    # print(Y)

    n_classes = Y.shape[1]
    # print('n_classes: ', n_classes)

    # NODES = 300
    # EPOCHS = 100

    # for nodes in [50, 100, 150, 200, 250, 300]:
    # for key, X in data.items():

    key = 'total_dropped'
    # X = data[key]

    print('Running : ', key, nodes, clasx, X.shape)

    # np.argwhere(np.isnan(X))

    # Split data into training and testing data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    # Normalize data with mean 0 and std 1
    X_scaled = normalize(X_train)

    # Add callback that streams epoch results to a csv file
    # https://keras.io/callbacks/
    csv_logger = CSVLogger('1vsAll_models/training_{}_{}_{}.log'.format(key, nodes, clasx))

    # Train the neural network model
    n_features = X.shape[1]
    model = nn_model(n_features, n_classes, nodes, 0.2)
    history = model.fit(X_scaled, Y_train, epochs=EPOCHS, batch_size=5, verbose=1, callbacks=[csv_logger])

    # Serialize model to JSON
    model_json = model.to_json()
    with open('1vsAll_models/model_{}_{}_{}.json'.format(key, nodes, clasx), 'w') as f:
        f.write(model_json)

    # Serialize weights to HDF5
    model.save_weights('1vsAll_models/model_{}_{}_{}.h5'.format(key, nodes, clasx))
    model.save('1vsAll_models/full_model_{}_{}_{}.h5'.format(key, nodes, clasx))


def test_single_model(clasx):
    from TouchAnalytics_management import load_data

    data, y = load_data()
    # One hot encoding of target vector

    # print(list(set(y)))
    # d = Counter(y)
    # print('--> Class {} has occurred {} times'.format(clasx, d[clasx]))
    # print('--> different classes elements are: ', d)

    y = [0 if val == clasx else 1 for val in y]
    # print(y)

    # summarize class distribution
    # print(Counter(y))
    # define undersample strategy
    undersample = RandomOverSampler(sampling_strategy='minority')
    # fit and apply the transform
    X_over, y_over = undersample.fit_resample(data['total_dropped'], y)
    # summarize class distribution
    print(Counter(y_over))

    X = X_over
    # print(X)
    print(X.shape)
    Y = pd.get_dummies(y_over).values
    # print(Y)
    print(Y.shape)

    # print(list(set(y)))
    # d = Counter(y)
    # print('--> Class ' + str(clasx) + ' occurred {} times'.format(d[0]))
    # print('--> different classe occurred {} times'.format(len(y) - d[clasx]))

    # n_classes = Y.shape[1]
    # print("n_classes: ", str(n_classes))
    # print("nodes: ", str(nodes))

    # corregere proporzioni
    # X = data['total_dropped']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    # print(f"Training target statistics: {Counter(Y_train)}")
    # print(f"Testing target statistics: {Counter(Y_test)}")

    # Normalize data with mean 0 and std 1
    X_test_scaled = normalize(X_test)

    model = load_model('1vsAll_models/full_model_total_dropped_300_' + str(clasx) + '.h5')

    score = model.evaluate(X_test_scaled, Y_test, verbose=0)
    print('X_test_scaled loss:', score[0])
    print('X_test_scaled accuracy:', score[1])

    test_prediction = model.predict(X_test_scaled)
    test_prediction = np.argmax(test_prediction, axis=1)

    Y_test_labels = np.argmax(Y_test, axis=1)

    data = {'true_classes': Y_test_labels, 'predictions': test_prediction}
    df = pd.DataFrame(data, columns=['true_classes', 'predictions'])
    confusion_matrix = pd.crosstab(df['true_classes'], df['predictions'], rownames=['true_classes'],
                                   colnames=['predictions'])
    plt.figure(figsize=(4, 3))
    sn.heatmap(confusion_matrix, linewidths=.5, annot=True)
    plt.show()

    # print('test_prediction: ', test_prediction)
    # print('Y_test_labels: ', Y_test_labels)

    # d = Counter(test_prediction)
    # print('--> Counter(test_prediction) =  {}'.format(d))
    # d = Counter(Y_test_labels)
    # print('--> Counter(Y_test_labels) =  {}'.format(d))

    report = classification_report(Y_test_labels, test_prediction)
    print(report)

    fpr, tpr, threshold = roc_curve(Y_test_labels, test_prediction, pos_label=1)
    auc_keras = auc(fpr, tpr)
    # print('auc: ', auc_keras)
    eer_point = compute_eer(fpr, tpr, threshold)
    # print('EER: ', eer_point[0])

    # plt.figure(1)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr, tpr, label='NN 1vsAll (area = {:.3f})'.format(auc_keras))
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
    "FUNZIONA"

    EPOCHS = 10
    NODES = 300

    # clasx = 502
    # calulate_model(clasx, NODES, EPOCHS)
    # test_single_model(clasx)

    results = pd.DataFrame(columns=['classe', 'eer', 'auc'])

    _, y = load_data()
    subjects = np.unique(y)
    print(subjects)

    for clasx in subjects:
        calulate_model(clasx, NODES, EPOCHS)
        eer2, auc2 = test_single_model(clasx)
        print(clasx, eer2, auc2)
        results.loc[clasx] = [clasx, eer2, auc2]
    results.loc['AVG'] = [clasx, results['eer'].mean(), results['auc'].mean()]
    print("RESULTS AVG = ", 'Results ', results['eer'].mean(), results['auc'].mean())
    results.to_csv(PathFrank_Results + '/' + "Swipes_Results_NN" + ".csv")

    # # Test Dataframe --> CSV
    # results = pd.DataFrame(columns=['A', 'B', 'C'])
    # results.loc[0] = [1, 5, 6]
    # results.to_csv(PathFrank_Results + '/' + "Swipes_Results" + ".csv")


    #  RESULTS AVG UP  =  0.09626951926074578       0.9037304807392542
    #  RESULTS AVG DOWN  =
    #  RESULTS AVG LEFT  =
    #  RESULTS AVG RIGHT  =