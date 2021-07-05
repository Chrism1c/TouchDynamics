import os
import pandas as pd

from src.BioIdent.BioIdent_management import Path_BioIdent, load_data, subjects, gender, touch_experience, \
    Path_BioIdent_Results
from src.core.NeuralNetwork import calulate_and_test_model

if __name__ == '__main__':

    # BioIdent_DBs = os.listdir(Path_BioIdent)
    #
    # print(BioIdent_DBs)

    # BioIdent_DBs = ['dataset1.arff', 'dataset2.arff', 'dataset3.arff', 'dataset4.arff']
    BioIdent_DBs = ['dataset1.arff']

    EPOCHS = 100
    NODES = 300
    Test_Size = 0.1
    Strategy = 'UnderSampler'   # 'OverSampler'  #

    results = pd.DataFrame(columns=['classe', 'eer', 'auc'])

    #db_index = 0
    for db_name in BioIdent_DBs:
        if db_name == 'dataset1.arff':
            classes = subjects
            db_index = 0
        elif db_name == 'dataset2.arff':
            classes = subjects
            db_index = 1
        elif db_name == 'dataset3.arff':
            classes = gender
            db_index = 2
        elif db_name == 'dataset4.arff':
            db_index = 3
            classes = touch_experience
        for clasx in classes:
            print("classes ", classes)
            X, y = load_data(db_index)
            eer, auc = calulate_and_test_model(X, y, db_name, clasx, EPOCHS, NODES, Test_Size, Strategy)
            print(clasx, eer, auc)
            results.loc[clasx] = [clasx, eer, auc]
        # db_index = db_index + 1
        results.loc['AVG'] = ['-->', results['eer'].mean(), results['auc'].mean()]
        print("RESULTS AVG = ", results['eer'].mean(), results['auc'].mean())
        results.to_csv(Path_BioIdent_Results + '/' + db_name + "_Results_" + Strategy + "_NN" + ".csv")
        results.drop(results.index, inplace=True)
    print("-------------------------------")
