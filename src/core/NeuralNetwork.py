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
from sklearn.metrics import roc_curve, auc
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import CSVLogger
from collections import Counter
from src.EER import compute_eer
from src.functions import confusion_matrix_NN, roc_curve_plot
from src.support import data_strategy

PLOTS = False
REPORT = False
DEBUG = True


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


def calulate_and_test_model(X, y, db_name, clasx, EPOCHS, nodes, Test_Size, Strategy):
    print('------------------------------------')
    print('--> Class {} has occurred {} times'.format(clasx, Counter(y)[clasx]))
    print('--> Class Counter pre Binarize', Counter(y))

    "Binarize dataset"
    y = [0 if val == clasx else 1 for val in y]

    # summarize class distribution
    print('-> Class {} has occurred {} times'.format(clasx, Counter(y)[clasx]))
    print('-> Class Counter post Binarize', Counter(y))

    # define  strategy  #   OVERSMAPLING/DOWNSAMPLING/NORMAL
    X, Y = data_strategy(X, y, Strategy)
    Y = pd.get_dummies(Y).values

    n_classes = Y.shape[1]
    print('n_classes: ', n_classes)

    print('Running ---> | ({}) database | ({}) class  | ({}) NN nodes | ({}) NN EPOCHS | ({}) data strategy | ({}) X.shape'
          .format(db_name, clasx, nodes, EPOCHS, Strategy, X.shape))

    " Split data into training and testing data "
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=Test_Size, random_state=1)

    # Normalize data with mean 0 and std 1
    X_scaled = normalize(X_train)

    # Add callback that streams epoch results to a csv file
    csv_logger = CSVLogger('models/training_{}_{}_{}_{}_{}.log'.format(db_name, clasx, nodes, EPOCHS, Strategy))

    # Train the neural network model
    n_features = X.shape[1]
    model = nn_model(n_features, n_classes, nodes, 0.2)
    history = model.fit(X_scaled, Y_train, epochs=EPOCHS, batch_size=5, verbose=1, callbacks=[csv_logger])

    # Serialize model to JSON
    model_json = model.to_json()
    with open('models/model_{}_{}_{}_{}_{}.json'.format(db_name, clasx, nodes, EPOCHS, Strategy), 'w') as f:
        f.write(model_json)

    # Serialize weights to HDF5
    model.save_weights('models/model_{}_{}_{}_{}_{}.h5'.format(db_name, clasx, nodes, EPOCHS, Strategy))
    model.save('models/model_{}_{}_{}_{}_{}.h5'.format(db_name, clasx, nodes, EPOCHS, Strategy))

    #    TEST   #

    X_test_scaled = normalize(X_test)
    score = model.evaluate(X_test_scaled, Y_test, verbose=0)
    print('X_test_scaled loss:', score[0])
    print('X_test_scaled accuracy:', score[1])

    test_prediction = model.predict(X_test_scaled)
    test_prediction = np.argmax(test_prediction, axis=1)

    Y_test_labels = np.argmax(Y_test, axis=1)

    if PLOTS:
        confusion_matrix_NN(Y_test_labels, test_prediction)

    if DEBUG:
        print('test_prediction: ', test_prediction)
        print('Y_test_labels: ', Y_test_labels)
        print('--> Conuter test_prediction ', Counter(test_prediction))
        print('--> Conuter Y_test_labels ', Counter(Y_test_labels))

    if REPORT:
        from sklearn.metrics import classification_report
        report = classification_report(Y_test_labels, test_prediction)
        print(report)

    fpr, tpr, threshold = roc_curve(Y_test_labels, test_prediction, pos_label=1)
    auc_keras = auc(fpr, tpr)
    eer_point = compute_eer(fpr, tpr, threshold)

    if DEBUG:
        print('auc: ', auc_keras)
        print('EER: ', eer_point[0])

    if PLOTS:
        roc_curve_plot(fpr, tpr, auc_keras)

    return eer_point[0], auc_keras

# if __name__ == '__main__':

# WekaArff_DBs = os.listdir(Path_WekaArf)  # glob.glob(Path_WekaArf + '\*.arff')
#
# print(WekaArff_DBs)
#
# EPOCHS = 10
# NODES = 300
# Test_Size = 0.1
# Strategy = 'OverSampler'  # 'UnderSampler'
# # clasx = 502
# # db = 0     # 0 - 5
#
# results = pd.DataFrame(columns=['classe', 'eer', 'auc'])
#
# db_index = 0
#
# for db_name in WekaArff_DBs:
#     for clasx in subjects:
#         X, y = load_data(db_index)
#         eer, auc = calulate_and_test_model(X, y, db_name, clasx, EPOCHS, NODES, Test_Size, Strategy)
#         print(clasx, eer, auc)
#         results.loc[clasx] = [clasx, eer, auc]
#     db_index = db_index + 1
#     results.loc['AVG'] = ['-->', results['eer'].mean(), results['auc'].mean()]
#     print("RESULTS AVG = ", results['eer'].mean(), results['auc'].mean())
#     results.to_csv(Path_WekaArf_Results + '/' + db_name + "_Results_" + Strategy + "_NN" + ".csv")
#     results.drop(results.index, inplace=True)
# print("-------------------------------")

# CON --> OVERSAMPLING
# 42_users_51 samples_user_17_features_sample.arff
