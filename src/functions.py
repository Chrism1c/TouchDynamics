import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as Knn
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sn

from src.EER import evaluateEER2


def loadData(path):
    """
    Scrivere la funzione loadData(path) che carichi un dataset memorizzato in un file in formato csv di cui si fornisce il percorso in un oggetto. Restituire l’oggetto caricato facendo uso della libreria Pandas. Si assume che il dataset contenga un insieme di connessioni a una rete. La natura delle connessioni (0=normale, 1=attacco) è descritta nell’ultima colonna del dataset.
    :param path: stringa del percorso del dataset
    :return: dataframe pandas del dataset estratto
    """
    data = pd.read_csv(path, na_values=['Infinity'], low_memory=False)
    # print(data.shape)
    return data


def preElaboration(data, list):
    """
    Scrivere la funzione preElaboration(data, list) che computa e stampa le statistiche che permetto la analisi della distribuzione dei valori di tutti gli attributi elencati in list (tali attributi sono quelli che appaiono come colonne nell’oggetto data).
    :param data: dataframe pandas del dataset
    :param list: lista delle variabili indipendenti del dataset
    :return: NULL
    """
    for c in list:
        print(c)
        # determine statistics on c
        stat = (data[c].describe())  # compute statistics on each colum
        print(stat)


def preElaborationBox(data, independentList, label):
    """
    Scrivere la funzione preElaborationBox(data, independentList, label) che visualizza i box plot della distribuzione dei valori di ciascun attributo della independentList raggruppati in base alla label (attributo da predire) – uno box plot per ciascun attributo di independent list
    :param data: dataframe pandas del dataset
    :param independentList: lista delle variabili indipendenti del dataset
    :param label: etichetta dell'attributo da predirre
    :return: NULL
    """
    for c in independentList:
        # print(c)
        stat = (data[c].describe())  # compute statistics on each colum
        # print(stat)
        boxplot = data.boxplot(column=c)  # compute boxplot on each column
        plt.xticks((1,), (c,))
        # plt.show()
        # show the boxplot of c by Label
        boxplot = data.boxplot(column=[c], by=label)
        plt.show()


def preElaborationScatter(data, independentList, label):
    """
    Scrivere la funzione preElaborationScatter(data, independentList, label) che visualizza gli scatter plot della distribuzione dei valori di ciascun attributo della independentList (asse X dello scatter plot) rispetto ai valori della label (asse Y) – uno scatter plot per ciascun attributo di independentList
    :param data: dataframe pandas del dataset
    :param independentList: lista delle variabili indipendenti del dataset
    :param label: etichetta dell'attributo da predirre
    :return: NULL
    """
    for c in independentList:
        # print(c)
        # stat = (data[c].describe())
        # print(stat)
        data.plot.scatter(x=c, y=label, title=c)
        plt.show()


def mutualInfoRank(data, independentList, label):
    """
    Calcola la Mutual Info
    :param data: dataset in input
    :param independentList: lista degli attributi da considerare
    :param label: etichetta dell'attributo
    :return:
    """
    from sklearn.feature_selection import mutual_info_classif
    res = dict(zip(independentList, mutual_info_classif(data[independentList], data[label], discrete_features=False)))
    sorted_x = sorted(res.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_x


def calculatePca(data, n):
    """
    Scrivere la funzione pca( data, n) che calcola le n componenti principali di data (senza la classe). Restituire il modello delle componenti principali
    :param data: database in input
    :param n: numero di pca da considerare
    :return: pca calcolata, array di stringhe relative alle "n" pca
    """
    pca = PCA(n_components=n)
    pca.fit(data)
    index = []
    for i in range(1, n + 1):
        index.append("PC" + str(i))
    return pca, index


def applyPca(X, pca, pcaIndex):
    """
    Scrivere la funzione applyPCA(pca,data) che trasforma data coerentemente con il modello pca passato . Restituire il dataset creato (comprensivo della label)
    :param X: dataset
    :param pca: pca già calcolata
    :param pcaIndex: Lista di stringhe indice della pca
    :return: nuovo dataset su cui è stata applicata la pca
    """
    X = pca.transform(X)
    pcaData = pd.DataFrame(X, columns=pcaIndex)
    return pcaData


def sss_(Xdata, ydata, NFold, Test_Size):
    # skf = StratifiedKFold(n_splits=NFold, shuffle=True)
    sss = StratifiedShuffleSplit(n_splits=NFold, random_state=0, test_size=Test_Size)
    ListXTrain = list()
    ListXTest = list()
    ListYTrain = list()
    ListYTest = list()
    for train_index, test_index in sss.split(Xdata, ydata):
        ListXTrain.append(Xdata.iloc[train_index])
        ListXTest.append(Xdata.iloc[test_index])
        ListYTrain.append(ydata.iloc[train_index])
        ListYTest.append(ydata.iloc[test_index])
    return ListXTrain, ListXTest, ListYTrain, ListYTest


def randomforest(Xdata, ydata, n_estimators, randomization, bootstrap):
    """
    Scrivere la funzione randomforest(Xdata,ydata,n_estimators,randomization,bootstrap) che costruisce una Random Forest in accordo ai parametri passati. Restituire la Random Forest costruita
    :param Xdata: Variabili indipendenti del dataset data
    :param ydata: Variabile dipendente del dataset data
    :param n_estimators: Numero di alberi utilizzati nella Random Forest (numero di alberi)
    :param randomization: Parametro randomization per la Random Forest (funzione per determinare il numero di attributi in input)
    :param bootstrap: Parametro bootstrap per la Random Forest (percentuale di esempi)
    :return: classificatore addestrato
    """
    clf = RandomForestClassifier(n_estimators, max_features=randomization, max_samples=bootstrap)
    clf.fit(Xdata, ydata)
    return clf


def KNNx(KNN_X, KNN_Y, n_neighbors):
    """
    Addestra il Classificatore KNN
    :param KNN_X: Variabili indipendenti del dataset X
    :param KNN_Y: Variabile dipendente
    :param n_neighbors: numero di vicini da considerare nella KNN
    :return: classificatore adestrato sul dataset X
    """
    neigh = Knn(n_neighbors)
    neigh.fit(KNN_X, KNN_Y)
    return neigh


def evaluate(X, y_true, clf):
    """
    Scrivere la funzione evaluate(Xdata,ydata_true,clf) che usa clf per predire le etichette di Xdata. Calcolare le metriche accuracy_score, balanced_accuracy_score, precision_score, recall_score ed f1_score usando i valori predetti e i valori reali. Restituire la lista delle metriche calcolate
    :param X: Fold di input
    :param y_true: etichette relative al fold X
    :param clf: classificatore pre-addestrato
    :return: array delle 5 metriche: Overall Accuracy / Balanced Accuracy / Precisione / Richiamo / F-Score
    """
    metrics = []
    y_pred = clf.predict(X)
    metrics.append(accuracy_score(y_true, y_pred))  # oa
    metrics.append(balanced_accuracy_score(y_true, y_pred))  # balanced accuracy
    metrics.append(precision_score(y_true, y_pred))  # precision
    metrics.append(recall_score(y_true, y_pred))  # recall
    metrics.append(f1_score(y_true, y_pred))  # fscore
    return metrics


def evaluateCV(folds, ListXTrain, ListXTest, ListyTrain, ListyTest, n_estimators=10, randomization="sqrt",
               bootstrap=0.5):
    """
    Scrivere la funzione evalauteCV(folds,ListXTrain,ListyTrain,ListXTest,ListyTest, n_estimators,randomization,bootstrap) che valuta la configurazione n_estimators,randomization,bootstrap nell’apprendimento della Random Forest usando la metrica fscore. Restituire lo fscore medio calcolato tramite la cross-validation
    :param folds: numero di folds utilizzati per la Stratified-CV
    :param ListXTrain: Lista di k fold contenente gli esempi di Train
    :param ListXTest: Lista di k fold contenente gli esempi di Test
    :param ListyTrain: Lista delle label di Train
    :param ListyTest: Lista delle label di Test
    :param n_estimators: Numero di alberi utilizzati nella Random Forest
    :param randomization: Parametro randomization per la Random Forest
    :param bootstrap: Parametro bootstrap per la Random Forest
    :return: avgTrain, avgTest = array delle medie delle metriche calcolate su ogni fold
    """
    avgTrain = [0.0, 0.0, 0.0, 0.0, 0.0]
    avgTest = [0.0, 0.0, 0.0, 0.0, 0.0]

    for i in range(folds):
        rf = randomforest(ListXTrain[i], ListyTrain[i], n_estimators, randomization, bootstrap)
        metricsTrain = evaluate(ListXTrain[i], ListyTrain[i], rf)
        for j in range(0, len(avgTrain)):
            avgTrain[j] += metricsTrain[j]
        metricsTest = evaluate(ListXTest[i], ListyTest[i], rf)
        for j in range(0, len(avgTest)):
            avgTest[j] += metricsTest[j]

    for j in range(0, len(avgTrain)):
        avgTrain[j] = avgTrain[j] / folds

    for j in range(0, len(avgTest)):
        avgTest[j] = avgTest[j] / folds

    return avgTrain, avgTest


def evaluateCV_3Classifiers(folds, ListXTrain, ListXTest, ListyTrain, ListyTest, n_estimators=100, randomization="sqrt",
                            bootstrap=0.5, n_neighbors=3):
    avgTest = [0.0, 0.0, 0.0]

    for i in range(folds):
        "Addestramento classificatori"
        rf = randomforest(ListXTrain[i], ListyTrain[i], n_estimators, randomization, bootstrap)
        neigh = KNNx(ListXTrain[i], ListyTrain[i], n_neighbors)

        SVM = svm.LinearSVC(dual=False).fit(ListXTrain[i], ListyTrain[i])         # Normalmente
        # SVM = svm.SVC(C=10.55, gamma=1.86).fit(ListXTrain[i], ListyTrain[i])      # Per il 41 Features
        # SVM = svm.SVC(C=7.46, gamma=0.25).fit(ListXTrain[i], ListyTrain[i])       # Per il 71 Features

        "EER"
        rf_eer, rf_roc_auc = evaluateEER2(ListyTest[i], rf.predict(ListXTest[i]))
        # print("Rf eer_threshold: ", rf_eer)
        knn_eer, knn_roc_auc = evaluateEER2(ListyTest[i], neigh.predict(ListXTest[i]))
        # print("knn eer_threshold: ", knn_eer)
        svm_eer, svmroc_auc = evaluateEER2(ListyTest[i], SVM.predict(ListXTest[i]))
        # print("svm eer_threshold: ", svm_eer)

        avgTest[0] += rf_eer[0]
        avgTest[1] += knn_eer[0]
        avgTest[2] += svm_eer[0]

        # print(avgTest)
        # print(i)

    for j in range(0, len(avgTest)):
        avgTest[j] = avgTest[j] / folds

    return avgTest


def applyPCAtoDataset(Lista, NPC):
    """
    Procedura che applica la PCA alla lista di folds (ES: ListXTrain / ListXTest)
    :param Lista: lista di folds (ES: ListXTrain / ListXTest)
    :param NPC: Numero di componenti proncipali per la PCA
    :return: NULL
    """
    for i in range(len(Lista)):
        X = Lista[i].drop(columns=['classification'], errors='ignore')
        pca, pcaIndex = calculatePca(X, NPC)
        pcaData = applyPca(X, pca, pcaIndex)
        Lista[i] = pcaData


def bestAVG_fScore(kfolds, ListXTrain, ListXTest, ListYTrain, ListYTest, listN_estimators, listRandomization,
                   listBootstrap, NPC=-1):
    """
    11)
    Con riferimento alle funzioni scritte in precedenza considerare il dataset OneClsNumeric.csv e usando la medesima 5-fold CV identificare la migliore configurazione ai punti (a) e (b)
    a. Random Forest costruita dal dataset originale variando randomization tra sqrt e log2, n_estimators tra 10, 20 e 30, bootstrap tra 0.5,0.6,0.7, 0.8 e 0.9
    b. Random Forest costruita dalle 10 top componenti principali variando randomization tra sqrt e log2, n_estimators tra 10, 20 e 30, bootstrap tra 0.5,0.6,0.7, 0.8 e 0.9
    :param kfolds: numero di folds utilizzati per la Stratified-CV
    :param ListXTrain: Lista di k fold contenente gli esempi di Train
    :param ListXTest: Lista di k fold contenente gli esempi di Test
    :param ListYTrain: Lista delle label di Train
    :param ListYTest: Lista delle label di Test
    :param listN_estimators: Lista dei parametri relativi al numero di alberi presenti nella Random Forest
    :param listRandomization: Lista dei parametri relativi alla randomization della Random Forest
    :param listBootstrap: Lista dei parametri relativi al bootstrap della Random Forest
    :param NPC: Numero di componenti principali da usare nella Random Forest con PCA
    :return: bestConf = Tripla della configurazione migliore per la Random Forest
    """
    bestConf = (0, "", 0)
    bestAVGTestF1measure = -1

    if NPC > 0:
        applyPCAtoDataset(ListXTrain, NPC)
        applyPCAtoDataset(ListXTest, NPC)

    for val_n_estimators in listN_estimators:
        for val_randomization in listRandomization:
            for val_bootstrap in listBootstrap:
                avgTrain, avgTest = evaluateCV(kfolds, ListXTrain, ListXTest, ListYTrain, ListYTest,
                                               val_n_estimators,
                                               val_randomization,
                                               val_bootstrap)
                # print("VAL = ", val_n_estimators, " ", val_randomization, " ", val_bootstrap)
                # if avgTrain[4] >= bestAVGTrainF1measure and avgTest[4] >= bestAVGTestF1measure:
                if avgTest[4] > bestAVGTestF1measure:
                    bestAVGTestF1measure = avgTest[4]
                    bestConf = (val_n_estimators, val_randomization, val_bootstrap)
    print("bestConf = ", bestConf)
    # print("bestAVGTestF1measure = ", bestAVGTestF1measure)
    return bestConf


def conf_matrix(labels, predictions):
    """
    Genera e stampa a video la matrice di confusione
    :param labels: variabile dipendente del dataset
    :param predictions: predizioni del classificatore della variabile dipendente
    :return: NULL
    """
    # listOfLabels = labels.values.tolist()
    CM = confusion_matrix(labels, predictions, normalize='true')
    print(CM)
    sn.heatmap(CM, annot=True)
    plt.show()


def confusion_matrix_NN(Y_test_labels, test_prediction):
    """
    Genera e stampa a video la matrice di confusione
    :param labels: variabile dipendente del dataset
    :param predictions: predizioni del classificatore della variabile dipendente
    :return: NULL
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y_test_labels, test_prediction)
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(5, 4))
    sn.heatmap(cmn, annot=True, fmt='.2f')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)

    dataCM = {'true_classes': Y_test_labels, 'predictions': test_prediction}
    df_CM = pd.DataFrame(dataCM, columns=['true_classes', 'predictions'])
    confusion_matrix = pd.crosstab(df_CM['true_classes'], df_CM['predictions'], rownames=['true_classes'],
                                   colnames=['predictions'])
    plt.figure(figsize=(10, 10))
    sn.heatmap(confusion_matrix, linewidths=.5, annot=True)
    plt.show()


def roc_curve_plot(fpr, tpr, auc_keras):
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

# RESULT OLD
# [0.9015, 0.8175, 0.9223359422034919, 0.9575, 0.9395890831033425]
# [0.898, 0.82375, 0.9266503667481663, 0.9475, 0.9369592088998764]
# [0.9015, 0.8175, 0.9223359422034919, 0.9575, 0.9395890831033425]

# [0.9025, 0.8190625, 0.9229379891631547, 0.958125, 0.9402023919043239]
# [0.899, 0.8225, 0.925700365408039, 0.95, 0.9376927822331893]
# [0.9025, 0.8190625, 0.9229379891631547, 0.958125, 0.9402023919043239]

# [0.9015, 0.8109375000000001, 0.9188059701492537, 0.961875, 0.9398473282442747]
# [0.8995, 0.80875, 0.9181111775254035, 0.96, 0.9385884509624198]
# [0.9015, 0.8109375000000001, 0.9188059701492537, 0.961875, 0.9398473282442747]
