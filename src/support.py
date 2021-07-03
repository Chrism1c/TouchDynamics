
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


def clean_dataset(df):
    import pandas as pd
    import numpy as np
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)