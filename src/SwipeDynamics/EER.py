# -*- coding: utf-8 -*-

from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt
import numpy as np


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

# def evaluateEER(user_scores, imposter_scores):
#     labels = [0] * len(user_scores) + [1] * len(imposter_scores)
#     fpr, tpr, thresholds = roc_curve(labels, user_scores + imposter_scores)
#     roc_auc = auc(fpr, tpr)
#     # RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator').plot()
#     # plt.show()
#
#     # print('fpr: ', list(fpr))
#     # print('tpr: ', list(tpr))
#     # print('thresholds: ', list(thresholds))
#     missrates = 1 - tpr
#     # print('missrates: ', list(missrates))
#     farates = fpr
#     dists = missrates - farates
#     # print('dists: ', list(dists))
#
#     idx1 = np.argmin(dists[dists >= 0])
#     idx2 = np.argmax(dists[dists < 0])
#     x = [missrates[idx1], farates[idx1]]
#     y = [missrates[idx2], farates[idx2]]
#     a = (x[0] - x[1]) / (y[1] - x[1] - y[0] + x[0])
#     eer = x[0] + a * (y[0] - x[0])
#
#     # plt.plot(missrates)
#     # plt.plot(farates)
#     # plt.scatter(idx1, eer)
#     # plt.show()
#
#     return eer, roc_auc
