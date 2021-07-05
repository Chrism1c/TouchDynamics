import os
import pandas as pd

from src.WekaArff.WekaArf_management import Path_WekaArff, load_data, subjects, Path_WekaArff_Results
from src.core.NeuralNetwork import calulate_and_test_model

if __name__ == '__main__':

    WekaArff_DBs = os.listdir(Path_WekaArff)

    print(WekaArff_DBs)

    EPOCHS = 10
    NODES = 300
    Test_Size = 0.1
    Strategy = 'OverSampler'  # 'UnderSampler'

    results = pd.DataFrame(columns=['classe', 'eer', 'auc'])

    db_index = 0
    for db_name in WekaArff_DBs:
        for clasx in subjects:
            X, y = load_data(db_index)
            eer, auc = calulate_and_test_model(X.drop(columns=['user_id']), y, db_name, clasx, EPOCHS, NODES, Test_Size, Strategy)
            print(clasx, eer, auc)
            results.loc[clasx] = [clasx, eer, auc]
        db_index = db_index + 1
        results.loc['AVG'] = ['-->', results['eer'].mean(), results['auc'].mean()]
        print("RESULTS AVG = ", results['eer'].mean(), results['auc'].mean())
        results.to_csv(Path_WekaArff_Results + '/' + db_name + "_Results_" + Strategy + "_NN" + ".csv")
        results.drop(results.index, inplace=True)
    print("-------------------------------")