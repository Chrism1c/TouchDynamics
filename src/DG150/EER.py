# -*- coding: utf-8 -*-

from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt
import numpy as np

from Mobilekey.NeuralNetwork import compute_eer


def evaluateEER(user_scores, imposter_scores):
    labels = [0] * len(user_scores) + [1] * len(imposter_scores)
    fpr, tpr, thresholds = roc_curve(labels, user_scores + imposter_scores)
    roc_auc = auc(fpr, tpr)

    missrates = 1 - tpr
    farates = fpr
    dists = missrates - farates
    idx1 = np.argmin(dists[dists >= 0])
    idx2 = np.argmax(dists[dists < 0])
    x = [missrates[idx1], farates[idx1]]
    y = [missrates[idx2], farates[idx2]]
    a = (x[0] - x[1]) / (y[1] - x[1] - y[0] + x[0])
    eer = x[0] + a * (y[0] - x[0])

    ## FAR/FRR curve
    # plt.plot(missrates)
    # plt.plot(farates)
    # plt.scatter(idx1, eer)
    # plt.show()

    ## ROC Curve
    # RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator').plot()
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.show()

    return eer, roc_auc

def evaluateEER2(Y_test_labels, test_prediction):
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

    return eer_point, auc_keras
