import numpy as np
import pandas as pd

from src.TouchAnalytics.TouchAnalytics_management import load_data, Path_TouchAnalytics_Results
from src.core.RF_Knn_SVM import RF_kNN_SVM, listBootstrap, listRandomization, listN_estimators

if __name__ == '__main__':

    "Parametri Generali "
    NFold = 2
    Strategy = 'OverSampler'  # 'UnderSampler'
    Test_Size = 0.1

    "Knn"
    n_neighbors = 3

    "Random Forest"
    n_estimators = listN_estimators[3]
    randomization = listRandomization[0]
    bootstrap = listBootstrap[0]

    "Load dataset"
    db_name = "TouchAnalytics"
    X, y = load_data()
    unique_subjects = np.unique(y)
    print(unique_subjects)

    results = pd.DataFrame(columns=['rf_eer', 'knn_eer', 'svm_eer'])

    for clasx in unique_subjects:

        avgTest_result = RF_kNN_SVM(X, y, db_name, clasx, Strategy, NFold, Test_Size,
                                    n_estimators, randomization, bootstrap, n_neighbors)

        results.loc[clasx] = [avgTest_result[0], avgTest_result[1], avgTest_result[2]]

    results.loc['AVG'] = [results['rf_eer'].mean(), results['knn_eer'].mean(), results['svm_eer'].mean()]
    print("RESULTS AVG = ", results['rf_eer'].mean(), results['knn_eer'].mean(), results['svm_eer'].mean())
    results.to_csv(Path_TouchAnalytics_Results + '/' + db_name.replace('.arff', '_Results_Shellow') + ".csv")


# RESULTS AVG =  0.006309526721887615 0.19573906144750342 0.4501502874003291
# 0.63%    -    19.57%  -       45.01 %

# RESULTS UP AVG =  0.008286533816425119 0.24711742424242428 0.13242366163425948
# 0.82%  -      24.71%  -      13.24 %

# RESULTS DOWN AVG = 0.0067321162850251266 0.20451794561550554 0.2335641403392358
# 0.67 %     -  20.45%   -      23.35%

# RESULTS LEFT AVG = 0.008379879385835318 0.21343205564525444 0.18247202999148168
# 0.83%     -   21.3%   -       18.24%

# RESULTS RIGHT AVG = 0.005491827232935916 0.1966138984137996 0.21677706796445304
# 0.54%     -   19.66%  -       21.6%



