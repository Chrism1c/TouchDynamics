from scipy.spatial.distance import cityblock, euclidean
import numpy as np

np.set_printoptions(suppress=True)
from src.EER import evaluateEER
import warnings

warnings.filterwarnings("ignore")
from abc import ABCMeta, abstractmethod
from TouchAnalytics_management import PathFrankArff
from src.support import clean_dataset

from scipy.io import arff
import pandas as pd

Start = "interstroketime"
Finish = "phoneorientation"


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
            genuine_user_data = data.loc[data.subject == subject, Start:Finish]
            imposter_data = data.loc[data.subject != subject, :]

            len_genuine, len_imposter = len(genuine_user_data), len(imposter_data)
            # print(len_genuine, " ", len_imposter)
            len_test_genuine = round(self.test_size * len_genuine)
            len_train_genuine = round(len_genuine - len_test_genuine)
            # print(len_test_genuine, " ", len_train_genuine)

            self.train = genuine_user_data[:len_train_genuine]  # 320
            self.test_genuine = genuine_user_data[len_train_genuine:]  # 80
            # print(len(self.train), " ", len(self.test_genuine))

            groupByImpostrors = imposter_data.groupby("subject")  # 50
            lines4impostor = round(len_test_genuine / len(groupByImpostrors))  # 1.6
            # print(lines4impostor)

            self.test_imposter = groupByImpostrors.head(lines4impostor).loc[:, Start:Finish]
            # print(len(self.test_imposter))

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


    data = arff.loadarff(PathFrankArff)
    df = pd.DataFrame(data[0])

    df = clean_dataset(df)

    print(df.head())

    data = df.drop(columns=['docid', 'phoneid', 'changeoffingerorientation'])

    subjects = df["subject"].unique()
    test_size = 0.2

    # print(data.head())
    # print(data.shape)
    # print(subjects)

    print("Results Manhattan detector:")
    # test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # for test_size in test_sizes:
    # print("--> test_size: ", test_size)
    obj = ManhattanDetector(subjects, test_size)
    result = obj.evaluate()
    print('AVG(ERR): ', result[0],
          '\nSTD(ERR): ', result[1],
          '\nAVG(auc_roc): ', result[2],
          '\nSTD(auc_roc): ', result[3])

    print("=====================================================================")
    print("Results Manhattan filtered detector:")
    obj = ManhattanFilteredDetector(subjects, test_size)
    result = obj.evaluate()
    print('AVG(ERR): ', result[0],
          '\nSTD(ERR): ', result[1],
          '\nAVG(auc_roc): ', result[2],
          '\nSTD(auc_roc): ', result[3])

    print("=====================================================================")
    print("Results Manhattan scaled detector:")
    obj = ManhattanScaledDetector(subjects, test_size)
    result = obj.evaluate()
    print('AVG(ERR): ', result[0],
          '\nSTD(ERR): ', result[1],
          '\nAVG(auc_roc): ', result[2],
          '\nSTD(auc_roc): ', result[3])
