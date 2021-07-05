import os
import pandas as pd
import numpy as np
from src.TouchAnalytics.TouchAnalytics_management import load_data, Path_TouchAnalytics_Results
from src.core.NeuralNetwork import calulate_and_test_model

if __name__ == '__main__':

    EPOCHS = 10
    NODES = 300
    Test_Size = 0.1
    Strategy = 'OverSampler'  # 'UnderSampler'

    X, y = load_data()
    unique_subjects = np.unique(y)
    print(unique_subjects)

    results = pd.DataFrame(columns=['classe', 'eer', 'auc'])

    db_name = "TouchAnalytics"
    for clasx in unique_subjects:
        eer, auc = calulate_and_test_model(X, y, db_name, clasx, EPOCHS, NODES, Test_Size, Strategy)
        print(clasx, eer, auc)
        results.loc[clasx] = [clasx, eer, auc]
    results.loc['AVG'] = ['-->', results['eer'].mean(), results['auc'].mean()]
    print("RESULTS AVG = ", results['eer'].mean(), results['auc'].mean())
    results.to_csv(Path_TouchAnalytics_Results + '/' + "TouchAnalytics_Results_ " + Strategy + "_"
                   + str(EPOCHS) + "_" + str(NODES) + "_" + str(Test_Size) + "_NN" + ".csv")
    results.drop(results.index, inplace=True)
    print("-------------------------------")
