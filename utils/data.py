from sklearn.model_selection import train_test_split
from utils.util import get_model_param

def dataSplit_sample(df, info_df, test_size=0.2):
    """
    Split data into training and test sets.
    """
    model_sha = df.keys()[0]
    model_list = info_df['models'].tolist()
    train_model_list, test_model_list = train_test_split(model_list, test_size=test_size)
    train_data = df[[model_sha] + list(train_model_list)]
    test_data = df[[model_sha] + list(test_model_list)]
    full_model_list = list(train_model_list) + list(test_model_list)
    full_data = df[[model_sha] + full_model_list]
    return full_data, full_model_list, train_data, train_model_list, test_data, test_model_list

def dataSplit_sample_basedParam(df, info_dict, test_size=0.2):
    """
    Split data into training and test sets based on the proportion of different categories in dataframes_dict.
    """
    train_model_list = []
    test_model_list = []

    for key, sub_df in info_dict.items():
        if key == 'All':
            continue
        model_list = sub_df['models'].tolist()
        train_list, test_list = train_test_split(model_list, test_size=test_size)
        train_model_list.extend(train_list)
        test_model_list.extend(test_list)
        # print(f'{key}: train {len(train_list)}, test {len(test_list)}')

    train_data = df[['model_sha'] + train_model_list]
    test_data = df[['model_sha'] + test_model_list]
    full_model_list = list(train_model_list) + list(test_model_list)
    full_data = df[['model_sha'] + full_model_list]
    return full_data, full_model_list, train_data, train_model_list, test_data, test_model_list