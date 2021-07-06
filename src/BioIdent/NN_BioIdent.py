import os
import pandas as pd
import numpy as np
from src.BioIdent.BioIdent_management import Path_BioIdent, load_data, subjects, gender, touch_experience, \
    Path_BioIdent_Results
from src.core.NeuralNetwork import calulate_and_test_model

if __name__ == '__main__':

    # BioIdent_DBs = os.listdir(Path_BioIdent)
    #
    # print(BioIdent_DBs)

    datasets_BioIdent = ['dataset1.arff', 'dataset2.arff', 'dataset3.arff', 'dataset4.arff']

    selected = ['dataset1.arff', 'dataset2.arff']

    EPOCHS = 100
    NODES = 300
    Test_Size = 0.1
    Strategy = 'UnderSampler'   # 'OverSampler'  #

    results = pd.DataFrame(columns=['classe', 'eer', 'auc'])

    #db_index = 0
    for db_name in selected:
        db_index = datasets_BioIdent.index(db_name)
        print("index : ", db_index)
        X, subjects = load_data(db_index)
        print('subjects ', np.unique(subjects))
        for clasx in np.unique(subjects):
            eer, auc = calulate_and_test_model(X, subjects, db_name, clasx, EPOCHS, NODES, Test_Size, Strategy)
            print(clasx, eer, auc)
            results.loc[clasx] = [clasx, eer, auc]
        # db_index = db_index + 1
        results.loc['AVG'] = ['-->', results['eer'].mean(), results['auc'].mean()]
        print("RESULTS AVG = ", results['eer'].mean(), results['auc'].mean())
        results.to_csv(Path_BioIdent_Results + '/' + db_name.replace('.arff', '_Results_') + Strategy + "_NN" + ".csv")
        results.drop(results.index, inplace=True)
        print("-------------------------------")
