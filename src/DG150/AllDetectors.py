import glob
from collections import Counter

from scipy.spatial.distance import cityblock, euclidean
import numpy as np

np.set_printoptions(suppress=True)
import pandas as pd
from src.EER import evaluateEER
import warnings
from src.Mobilekey.mobilekey_management import Path_MobileKEY, get_df_from_arff

warnings.filterwarnings("ignore")
from abc import ABCMeta, abstractmethod


class Detector:
    __metaclass__ = ABCMeta

    def __init__(self, subjects, test_size):
        self.cov_type = ''
        self.metrics = list()
        self.user_scores = []
        self.imposter_scores = []
        self.mean_vector = []
        self.subjects = subjects
        self.test_size = test_size

    @abstractmethod
    def training(self):
        pass

    @abstractmethod
    def testing(self):
        pass

    def evaluate(self):
        eers = []
        roc_aucs = []
        # test_size = 0.2

        for subject in subjects:
            # print("*************")

            genuine_user_data = data.loc[data['user_id'] == subject]
            # print(genuine_user_data)
            genuine_user_data = genuine_user_data.iloc[:, :-1]
            # print(genuine_user_data)

            imposter_data = data.loc[data['user_id'] != subject, :]
            # print(imposter_data)

            data_binary = [0 if val == subject else 1 for val in data['user_id'].values]

            print(data_binary)

            # example of random undersampling to balance the class distribution
            from imblearn.over_sampling import RandomOverSampler
            # summarize class distribution

            # print(data_binary)

            # print(Counter(data_binary))

            # define undersample strategy
            oversampler = RandomOverSampler(sampling_strategy='minority')
            # fit and apply the transform
            X_over, y_over = oversampler.fit_resample(data.iloc[:, :-1], data_binary)
            # summarize class distribution
            print(Counter(y_over))

            len_genuine, len_imposter = len(genuine_user_data), len(imposter_data)

            # print(len_genuine, " ", len_imposter)
            len_test_genuine = round(self.test_size * len_genuine)
            len_train_genuine = round(len_genuine - len_test_genuine)
            # print(len_test_genuine, " ", len_train_genuine)

            self.train = genuine_user_data[:len_train_genuine]  # 320
            self.test_genuine = genuine_user_data[len_train_genuine:]  # 80
            # print(len(self.train), " ", len(self.test_genuine))

            id_impostors = imposter_data['user_id'].values  # 50
            unique_impostors = np.unique(id_impostors)
            # print(len(unique_impostors))

            self.test_imposter = (imposter_data.sample(frac=1).head(len_test_genuine)).iloc[:, :-1]
            # print('test_imposter: ', len(self.test_imposter))

            self.training()
            self.testing()

            ev = evaluateEER(self.user_scores, self.imposter_scores)
            eers.append(ev[0])
            roc_aucs.append(ev[1])

        return np.mean(eers), np.std(eers), np.mean(roc_aucs), np.std(roc_aucs)


class ManhattanDetector(Detector):

    def training(self):
        self.mean_vector = self.train.mean().values

    def testing(self):
        for i in range(self.test_genuine.shape[0]):
            cur_score = cityblock(self.test_genuine.iloc[i].values, self.mean_vector)
            self.user_scores.append(cur_score)

        for i in range(self.test_imposter.shape[0]):
            cur_score = cityblock(self.test_imposter.iloc[i].values, self.mean_vector)
            self.imposter_scores.append(cur_score)


class ManhattanFilteredDetector(Detector):

    def training(self):
        self.mean_vector = self.train.mean().values
        self.std_vector = self.train.std().values
        dropping_indices = []
        for i in range(self.train.shape[0]):
            cur_score = euclidean(self.train.iloc[i].values, self.mean_vector)
            if (cur_score > 3 * self.std_vector).all() == True:
                dropping_indices.append(i)
        self.train = self.train.drop(self.train.index[dropping_indices])
        self.mean_vector = self.train.mean().values

    def testing(self):
        for i in range(self.test_genuine.shape[0]):
            cur_score = cityblock(self.test_genuine.iloc[i].values, self.mean_vector)
            self.user_scores.append(cur_score)

        for i in range(self.test_imposter.shape[0]):
            cur_score = cityblock(self.test_imposter.iloc[i].values, self.mean_vector)
            self.imposter_scores.append(cur_score)


class ManhattanScaledDetector(Detector):

    def training(self):
        self.mean_vector = self.train.mean().values
        self.mad_vector = self.train.mad().values

    def testing(self):
        for i in range(self.test_genuine.shape[0]):
            cur_score = 0
            for j in range(len(self.mean_vector)):
                cur_score = cur_score + abs(self.test_genuine.iloc[i].values[j] - self.mean_vector[j]) / \
                            self.mad_vector[j]
            self.user_scores.append(cur_score)

        for i in range(self.test_imposter.shape[0]):
            cur_score = 0
            for j in range(len(self.mean_vector)):
                cur_score = cur_score + abs(self.test_imposter.iloc[i].values[j] - self.mean_vector[j]) / \
                            self.mad_vector[j]
            self.imposter_scores.append(cur_score)


if __name__ == '__main__':
    MobileKey_DBs = ['MobilKey_all_easy', 'MobilKey_all_logicalstrong', 'MobilKey_all_strong',
                     'MobilKey_secondorder_easy', 'MobilKey_secondorder_logicalstrong', 'MobilKey_secondorder_strong']
    index = 0

    datasets_MobileKey = glob.glob(Path_MobileKEY + '/' + '/' + MobileKey_DBs[index] + '*.arff')
    print(datasets_MobileKey[index])
    data = get_df_from_arff(datasets_MobileKey[index])

    data['user_id'] = pd.to_numeric(data['user_id'])
    subjects = data['user_id'].unique()
    print(data)

    test_size = 0.2

    # print("Results Manhattan detector:")
    # # test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # # for test_size in test_sizes:
    # # print("--> test_size: ", test_size)
    # obj = ManhattanDetector(subjects, test_size)
    # result = obj.evaluate()
    # print('AVG(ERR): ', result[0],
    #       '\nSTD(ERR): ', result[1],
    #       '\nAVG(auc_roc): ', result[2],
    #       '\nSTD(auc_roc): ', result[3])
    #
    # print("=====================================================================")
    # print("Results Manhattan filtered detector:")
    # obj = ManhattanFilteredDetector(subjects, test_size)
    # result = obj.evaluate()
    # print('AVG(ERR): ', result[0],
    #       '\nSTD(ERR): ', result[1],
    #       '\nAVG(auc_roc): ', result[2],
    #       '\nSTD(auc_roc): ', result[3])
    #
    # print("=====================================================================")
    # print("Results Manhattan scaled detector:")
    # obj = ManhattanScaledDetector(subjects, test_size)
    # result = obj.evaluate()
    # print('AVG(ERR): ', result[0],
    #       '\nSTD(ERR): ', result[1],
    #       '\nAVG(auc_roc): ', result[2],
    #       '\nSTD(auc_roc): ', result[3])
