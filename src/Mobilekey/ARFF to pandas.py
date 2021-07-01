

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


