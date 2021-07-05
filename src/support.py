from collections import Counter

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn import svm
import pandas as pd
from src.EER import evaluateEER2
from src.functions import randomforest, KNNx


def get_df_from_arff(arff_path):
    from scipy.io import arff
    import pandas as pd
    try:
        data = arff.loadarff(arff_path)
        df = pd.DataFrame(data[0])
        # print(df.head())
        return df
    except:
        print("Path is not valid")


def data_strategy(X, y, Strategy):
    # define  strategy
    if Strategy == 'OverSampler':
        sample = RandomOverSampler(sampling_strategy='minority')
        X_over, y_over = sample.fit_resample(X, y)
        print(Counter(y_over))
        return X_over, y_over
    elif Strategy == 'UnderSampler':
        sample = RandomUnderSampler(sampling_strategy='majority')
        X_over, y_over = sample.fit_resample(X, y)
        print(Counter(y_over))
        return X_over, y_over
    else:
        print("-- No Sampler -- ")
        return X, y


def clean_dataset(df):
    import pandas as pd
    import numpy as np
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

