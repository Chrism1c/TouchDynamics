from pandas import DataFrame
from scipy.io import arff
from scipy import stats
import pandas as pd
import numpy as np
import glob

PrePath = 'D:/Users/Francesco/gitProjects'
# PrePath = 'D:/pycharmProjects'
Path_DG150 = PrePath + '/TouchDynamics/datasets/KeystrokeTouch/DG150'
datasets_DG150 = ['Long_DG150', 'Short_DG150']

DEBUG = False


def merge_data_to_pickle(index):
    selected_DG150 = glob.glob(Path_DG150 + '/' + datasets_DG150[index] + '/*.csv')

    print(selected_DG150)
    df = pd.DataFrame()

    user_id = 0
    for csv in selected_DG150:
        data = pd.read_csv(csv, header=None, names=['sample', 'keydown', 'TAP', 'keyup', 'TAR', 'PressureSize', 'uno'],
                           index_col=False)
        data.insert(0, 'user_id', [user_id] * (data.shape[0]))
        data = data.drop(columns='uno')
        # print(data)
        # print(data.shape)
        user_id += 1
        df = pd.concat([df, data])
        print(df.shape)
    pd.to_pickle(df, Path_DG150 + '\\' + "pickle_" + datasets_DG150[index] + "DG150.pkl")

    # print(data)
    # print(data.shape)

    # data = pd.read_csv(Path_MobileKEY + '/' + datasets_MobileKey[index])

    # clean_dataset(data)

    # print(data.dtypes)
    #
    # data['user_id'] = pd.to_numeric(data['user_id'])
    #
    # return data.drop(columns=['user_id']), data['user_id'].values


def features_extraction(df):
    df_features_extracted = pd.DataFrame()
    dim = df.shape

    print(list(df.iloc[0]))


def get_MN(array):
    return np.min(array)


def get_MX(array):
    return np.max(array)


def get_AM(array):
    return np.mean(array)

def get_QM(array):
    current_array = list(array)

    for element in current_array:
        element = element ** 2
        
    return np.sqrt(np.mean(current_array))

def get_HM(array):
    return stats.hmean(array)

def get_GM(array):
    return stats.gmean(array)

def get_MD(array):
    return np.median(array)

def get_RG(array):
    return get_MX(array) - get_MN(array)

def get_VR(array):
    return np.var(array)

def get_SD(array):
    return np.std(array)

def get_SK(array):
    return stats.skew(array)

def get_KU(array):
    return stats.kurtosis(array)

def get_FQ(array):
    return np.quantile(array, 0.25)

def get_TQ(array):
    return np.quantile(array, 0.75)

def get_IR(array):
    return get_TQ(array) - get_FQ(array)

def get_MA(array):
    return stats.median_abs_deviation(array)

def get_MI(array):
    return stats.median_abs_deviation(array)

def get_CV(array):
    return stats.variation(array)

def get_SE(array):
    return stats.sem(array)

def get_SOF(FOF_array):
    current_array = FOF_array
    Not_none_values = filter(None.__ne__, current_array)
    current_array = list(Not_none_values)
    
    res = {}
    res["mn_list"] = (np.full((1, len(FOF_array)), get_MN(current_array))[0]).tolist()
    res["mx_list"] = (np.full((1, len(FOF_array)), get_MX(current_array))[0]).tolist()
    res["am_list"] = (np.full((1, len(FOF_array)), get_AM(current_array))[0]).tolist()
    res["qm_list"] = (np.full((1, len(FOF_array)), get_QM(current_array))[0]).tolist()
    res["hm_list"] = (np.full((1, len(FOF_array)), get_HM(current_array))[0]).tolist()
    res["gm_list"] = (np.full((1, len(FOF_array)), get_GM(current_array))[0]).tolist()
    res["md_list"] = (np.full((1, len(FOF_array)), get_MD(current_array))[0]).tolist()
    res["rg_list"] = (np.full((1, len(FOF_array)), get_RG(current_array))[0]).tolist()
    res["vr_list"] = (np.full((1, len(FOF_array)), get_VR(current_array))[0]).tolist()
    res["sd_list"] = (np.full((1, len(FOF_array)), get_SD(current_array))[0]).tolist()
    res["sk_list"] = (np.full((1, len(FOF_array)), get_SK(current_array))[0]).tolist()
    res["ku_list"] = (np.full((1, len(FOF_array)), get_KU(current_array))[0]).tolist()
    res["fq_list"] = (np.full((1, len(FOF_array)), get_FQ(current_array))[0]).tolist()
    res["tq_list"] = (np.full((1, len(FOF_array)), get_TQ(current_array))[0]).tolist()
    res["ir_list"] = (np.full((1, len(FOF_array)), get_IR(current_array))[0]).tolist()
    res["ma_list"] = (np.full((1, len(FOF_array)), get_MA(current_array))[0]).tolist()
    res["mi_list"] = (np.full((1, len(FOF_array)), get_MI(current_array))[0]).tolist()
    res["cv_list"] = (np.full((1, len(FOF_array)), get_CV(current_array))[0]).tolist()
    res["se_list"] = (np.full((1, len(FOF_array)), get_SE(current_array))[0]).tolist()
    return res

def merge(FOF_dict1, FOF_dict2):
    FOF_dict1["mn_list"].extend(FOF_dict2["mn_list"])
    FOF_dict1["mx_list"].extend(FOF_dict2["mx_list"])
    FOF_dict1["am_list"].extend(FOF_dict2["am_list"])
    FOF_dict1["qm_list"].extend(FOF_dict2["qm_list"])
    FOF_dict1["hm_list"].extend(FOF_dict2["hm_list"])
    FOF_dict1["gm_list"].extend(FOF_dict2["gm_list"])
    FOF_dict1["md_list"].extend(FOF_dict2["md_list"])
    FOF_dict1["rg_list"].extend(FOF_dict2["rg_list"])
    FOF_dict1["vr_list"].extend(FOF_dict2["vr_list"])
    FOF_dict1["sd_list"].extend(FOF_dict2["sd_list"])
    FOF_dict1["sk_list"].extend(FOF_dict2["sk_list"])
    FOF_dict1["ku_list"].extend(FOF_dict2["ku_list"])
    FOF_dict1["fq_list"].extend(FOF_dict2["fq_list"])
    FOF_dict1["tq_list"].extend(FOF_dict2["tq_list"])
    FOF_dict1["ir_list"].extend(FOF_dict2["ir_list"])
    FOF_dict1["ma_list"].extend(FOF_dict2["ma_list"])
    FOF_dict1["mi_list"].extend(FOF_dict2["mi_list"])
    FOF_dict1["cv_list"].extend(FOF_dict2["cv_list"])
    FOF_dict1["se_list"].extend(FOF_dict2["se_list"])
    return FOF_dict1


def get_DT(df, index):
    if df.iloc[index][2] != 'E':
        DT_time = df.iloc[index][5] - df.iloc[index][3]
        if DEBUG == True:
            print(df.iloc[index][5], " - ", df.iloc[index][3], " = ", DT_time)
        return DT_time
    else:
        if DEBUG == True:
            print("NaN")
        return None


def get_FT1(df, index):
    if df.iloc[index][2] != 'E':
        FT1_time = df.iloc[index + 1][3] - df.iloc[index][5]
        if DEBUG == True:
            print(df.iloc[index + 1][3], " - ", df.iloc[index][5], " = ", FT1_time)
        return FT1_time
    else:
        if DEBUG == True:
            print("NaN")
        return None


def get_FT2(df, index):
    if df.iloc[index][2] != 'E':
        FT2_time = df.iloc[index + 1][5] - df.iloc[index][5]
        if DEBUG == True:
            print(df.iloc[index + 1][5], " - ", df.iloc[index][5], " = ", FT2_time)
        return FT2_time
    else:
        if DEBUG == True:
            print("NaN")
        return None


def get_FT3(df, index):
    if df.iloc[index][2] != 'E':
        FT3_time = df.iloc[index + 1][3] - df.iloc[index][3]
        if DEBUG == True:
            print(df.iloc[index + 1][3], " - ", df.iloc[index][3], " = ", FT3_time)
        return FT3_time
    else:
        if DEBUG == True:
            print("NaN")
        return None


def get_FT4(df, index):
    if df.iloc[index][2] != 'E':
        FT4_time = df.iloc[index + 1][5] - df.iloc[index][3]
        if DEBUG == True:
            print(df.iloc[index + 1][5], " - ", df.iloc[index][3], " = ", FT4_time)
        return FT4_time
    else:
        if DEBUG == True:
            print("NaN")
        return None


def get_IT(df, index):
    user_id, sample, keydown, TAP, keyup, TAR, PressureSize = list(df.iloc[index])
    if df.iloc[index][2] != 'E':
        sample_rows = df.loc[(df['sample'] == sample) & (df['user_id'] == user_id)]
        TAP_firstKey = list((sample_rows.head(1)).iloc[0])[3]
        TAR_lastKey = list((sample_rows.tail(1)).iloc[0])[5]
        IT_time = TAR_lastKey - TAP_firstKey
        if DEBUG == True:
            print(TAR_lastKey, " - ", TAP_firstKey, " = ", IT_time)
        return IT_time
    else:
        if DEBUG == True:
            print("NaN")
        return None


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


columns = []

if __name__ == '__main__':
    # data_X, data_Y = load_data(0)
    #
    # print(data_X.columns)
    # print(data_X)
    # print("\n")
    # print(data_Y)
    # print("\n", len(data_Y))

    index_dataset = 1  # 0/1
    # merge_data_to_pickle(index_dataset)
    df = pd.read_pickle(Path_DG150 + '\\' + "pickle_" + datasets_DG150[index_dataset] + "DG150.pkl")
    print(df)
    # print(df.head(30))
    # print(df.tail(30))
    # df.to_csv(Path_DG150 + '\\' + "CSV_" + datasets_DG150[index_dataset] + "DG150.csv", index=False)
    print('OK')

    # features_extraction(df)

    # print(list(df.iloc[0]))

    # row_id = 25499  # 0 - 25499
    # print(list(df.iloc[row_id]))
    # get_DT(df, row_id)
    # get_IT(df, row_id)


    print('========= MAKING DATASET =========')
    print()


    user_counter = {}
    user_id_list = []
    dt_list = []
    ft1_list = []
    ft2_list = []
    ft3_list = []
    ft4_list = []
    it_list = []
    ps_list = []

    for index, row in df.iterrows():
    # # TO TEST ONLY ON 300 ITEMS
    # for index in range(300):
    #     row = df.iloc[index]
    #     print(index)

        if str(row['user_id']) not in user_counter:
            user_counter[str(row['user_id'])] = 1
        else:
            user_counter[str(row['user_id'])] += 1

        user_id_list.append(row['user_id'])
        dt_list.append(get_DT(df, index))
        ft1_list.append(get_FT1(df, index))
        ft2_list.append(get_FT2(df, index))
        ft3_list.append(get_FT3(df, index))
        ft4_list.append(get_FT4(df, index))
        it_list.append(get_IT(df, index))
        ps_list.append(row[6])


    current_start_index = 0
    is_first = True

    for user in user_counter.keys():
        counter = user_counter[user]
        
        if(is_first == True):
            total_dt    =   get_SOF(dt_list[current_start_index:current_start_index+counter])
            total_ft1   =   get_SOF(ft1_list[current_start_index:current_start_index+counter])
            total_ft2   =   get_SOF(ft2_list[current_start_index:current_start_index+counter])
            total_ft3   =   get_SOF(ft3_list[current_start_index:current_start_index+counter])
            total_ft4   =   get_SOF(ft4_list[current_start_index:current_start_index+counter])
            total_it    =   get_SOF(it_list[current_start_index:current_start_index+counter])
            total_ps    =   get_SOF(ps_list[current_start_index:current_start_index+counter])
            is_first = False
        else:
            total_dt    =   merge(total_dt,  get_SOF(dt_list[current_start_index:current_start_index+counter]))
            total_ft1   =   merge(total_ft1, get_SOF(ft1_list[current_start_index:current_start_index+counter]))
            total_ft2   =   merge(total_ft2, get_SOF(ft2_list[current_start_index:current_start_index+counter]))
            total_ft3   =   merge(total_ft3, get_SOF(ft3_list[current_start_index:current_start_index+counter]))
            total_ft4   =   merge(total_ft4, get_SOF(ft4_list[current_start_index:current_start_index+counter]))
            total_it    =   merge(total_it,  get_SOF(it_list[current_start_index:current_start_index+counter]))
            total_ps    =   merge(total_ps,  get_SOF(ps_list[current_start_index:current_start_index+counter]))

        current_start_index += counter

    # adding last user data
    dataset = pd.DataFrame({
        "user_id": user_id_list,
        "dt": dt_list,
        "ft1": ft1_list,
        "ft2": ft2_list,
        "ft3": ft3_list,
        "ft4": ft4_list,
        "it": it_list,
        "ps": ps_list,

        "mn_dt":      total_dt["mn_list"],
        "mx_dt":      total_dt["mx_list"],
        "am_dt":      total_dt["am_list"],
        "qm_dt":      total_dt["qm_list"],
        "hm_dt":      total_dt["hm_list"],
        "gm_dt":      total_dt["gm_list"],
        "md_dt":      total_dt["md_list"],
        "rg_dt":      total_dt["rg_list"],
        "vr_dt":      total_dt["vr_list"],
        "sd_dt":      total_dt["sd_list"],
        "sk_dt":      total_dt["sk_list"],
        "ku_dt":      total_dt["ku_list"],
        "fq_dt":      total_dt["fq_list"],
        "tq_dt":      total_dt["tq_list"],
        "ir_dt":      total_dt["ir_list"],
        "ma_dt":      total_dt["ma_list"],
        "mi_dt":      total_dt["mi_list"],
        "cv_dt":      total_dt["cv_list"],
        "se_dt":      total_dt["se_list"],

        "mn_ft1":     total_ft1["mn_list"],
        "mx_ft1":     total_ft1["mx_list"],
        "am_ft1":     total_ft1["am_list"],
        "qm_ft1":     total_ft1["qm_list"],
        "hm_ft1":     total_ft1["hm_list"],
        "gm_ft1":     total_ft1["gm_list"],
        "md_ft1":     total_ft1["md_list"],
        "rg_ft1":     total_ft1["rg_list"],
        "vr_ft1":     total_ft1["vr_list"],
        "sd_ft1":     total_ft1["sd_list"],
        "sk_ft1":     total_ft1["sk_list"],
        "ku_ft1":     total_ft1["ku_list"],
        "fq_ft1":     total_ft1["fq_list"],
        "tq_ft1":     total_ft1["tq_list"],
        "ir_ft1":     total_ft1["ir_list"],
        "ma_ft1":     total_ft1["ma_list"],
        "mi_ft1":     total_ft1["mi_list"],
        "cv_ft1":     total_ft1["cv_list"],
        "se_ft1":     total_ft1["se_list"],

        "mn_ft2":     total_ft2["mn_list"],
        "mx_ft2":     total_ft2["mx_list"],
        "am_ft2":     total_ft2["am_list"],
        "qm_ft2":     total_ft2["qm_list"],
        "hm_ft2":     total_ft2["hm_list"],
        "gm_ft2":     total_ft2["gm_list"],
        "md_ft2":     total_ft2["md_list"],
        "rg_ft2":     total_ft2["rg_list"],
        "vr_ft2":     total_ft2["vr_list"],
        "sd_ft2":     total_ft2["sd_list"],
        "sk_ft2":     total_ft2["sk_list"],
        "ku_ft2":     total_ft2["ku_list"],
        "fq_ft2":     total_ft2["fq_list"],
        "tq_ft2":     total_ft2["tq_list"],
        "ir_ft2":     total_ft2["ir_list"],
        "ma_ft2":     total_ft2["ma_list"],
        "mi_ft2":     total_ft2["mi_list"],
        "cv_ft2":     total_ft2["cv_list"],
        "se_ft2":     total_ft2["se_list"],

        "mn_ft3":     total_ft3["mn_list"],
        "mx_ft3":     total_ft3["mx_list"],
        "am_ft3":     total_ft3["am_list"],
        "qm_ft3":     total_ft3["qm_list"],
        "hm_ft3":     total_ft3["hm_list"],
        "gm_ft3":     total_ft3["gm_list"],
        "md_ft3":     total_ft3["md_list"],
        "rg_ft3":     total_ft3["rg_list"],
        "vr_ft3":     total_ft3["vr_list"],
        "sd_ft3":     total_ft3["sd_list"],
        "sk_ft3":     total_ft3["sk_list"],
        "ku_ft3":     total_ft3["ku_list"],
        "fq_ft3":     total_ft3["fq_list"],
        "tq_ft3":     total_ft3["tq_list"],
        "ir_ft3":     total_ft3["ir_list"],
        "ma_ft3":     total_ft3["ma_list"],
        "mi_ft3":     total_ft3["mi_list"],
        "cv_ft3":     total_ft3["cv_list"],
        "se_ft3":     total_ft3["se_list"],

        "mn_ft4":     total_ft4["mn_list"],
        "mx_ft4":     total_ft4["mx_list"],
        "am_ft4":     total_ft4["am_list"],
        "qm_ft4":     total_ft4["qm_list"],
        "hm_ft4":     total_ft4["hm_list"],
        "gm_ft4":     total_ft4["gm_list"],
        "md_ft4":     total_ft4["md_list"],
        "rg_ft4":     total_ft4["rg_list"],
        "vr_ft4":     total_ft4["vr_list"],
        "sd_ft4":     total_ft4["sd_list"],
        "sk_ft4":     total_ft4["sk_list"],
        "ku_ft4":     total_ft4["ku_list"],
        "fq_ft4":     total_ft4["fq_list"],
        "tq_ft4":     total_ft4["tq_list"],
        "ir_ft4":     total_ft4["ir_list"],
        "ma_ft4":     total_ft4["ma_list"],
        "mi_ft4":     total_ft4["mi_list"],
        "cv_ft4":     total_ft4["cv_list"],
        "se_ft4":     total_ft4["se_list"],

        "mn_it":      total_it["mn_list"],
        "mx_it":      total_it["mx_list"],
        "am_it":      total_it["am_list"],
        "qm_it":      total_it["qm_list"],
        "hm_it":      total_it["hm_list"],
        "gm_it":      total_it["gm_list"],
        "md_it":      total_it["md_list"],
        "rg_it":      total_it["rg_list"],
        "vr_it":      total_it["vr_list"],
        "sd_it":      total_it["sd_list"],
        "sk_it":      total_it["sk_list"],
        "ku_it":      total_it["ku_list"],
        "fq_it":      total_it["fq_list"],
        "tq_it":      total_it["tq_list"],
        "ir_it":      total_it["ir_list"],
        "ma_it":      total_it["ma_list"],
        "mi_it":      total_it["mi_list"],
        "cv_it":      total_it["cv_list"],
        "se_it":      total_it["se_list"],

        "mn_ps":      total_ps["mn_list"],
        "mx_ps":      total_ps["mx_list"],
        "am_ps":      total_ps["am_list"],
        "qm_ps":      total_ps["qm_list"],
        "hm_ps":      total_ps["hm_list"],
        "gm_ps":      total_ps["gm_list"],
        "md_ps":      total_ps["md_list"],
        "rg_ps":      total_ps["rg_list"],
        "vr_ps":      total_ps["vr_list"],
        "sd_ps":      total_ps["sd_list"],
        "sk_ps":      total_ps["sk_list"],
        "ku_ps":      total_ps["ku_list"],
        "fq_ps":      total_ps["fq_list"],
        "tq_ps":      total_ps["tq_list"],
        "ir_ps":      total_ps["ir_list"],
        "ma_ps":      total_ps["ma_list"],
        "mi_ps":      total_ps["mi_list"],
        "cv_ps":      total_ps["cv_list"],
        "se_ps":      total_ps["se_list"]
    })

    print(clean_dataset(dataset))
    clean_dataset(dataset).to_pickle(Path_DG150 + '\\' + "FE_dataset_pickle_" + datasets_DG150[index_dataset] + "DG150.pkl")
