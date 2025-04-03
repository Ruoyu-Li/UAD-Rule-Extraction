import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from IPy import IP
from global_var import *


def load_data(dataset, subset, mode='train', **kwargs):
    if dataset == 'cicids':
        X, y = load_cicids(subset)
    elif dataset == 'unsw':
        X, y = load_unsw(subset)
    elif dataset == 'cicids_custom':
        X, y = load_cicids_custom(subset)
    elif dataset == 'toniot_custom':
        X, y = load_toniot_custom(subset)
    elif dataset == 'cicids_improved':
        X, y = load_cicids_improved(subset, **kwargs)
    elif dataset == 'cse_improved':
        X, y = load_cse_improved(subset, **kwargs)
    else:
        print('no such dataset')
        exit()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    if mode == 'train':
        X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.25, random_state=SEED)
        try:
            if kwargs['biclass'] == False:
                X_train, y_train = X_train[y_train == 0], y_train[y_train == 0]
        except:
            X_train, y_train = X_train[y_train == 0], y_train[y_train == 0]
        if 'random_select' in kwargs:
            idx_rand = np.random.randint(0, X_train.shape[0], kwargs['random_select'])
            X_train, y_train = X_train[idx_rand], y_train[idx_rand]

        return X_train, X_eval, y_train.astype(int), y_eval.astype(int)
    elif mode == 'test':
        return X_test, y_test

# CICIDS
def encode_label_cicids(col: pd.Series):
    all_labels = list(set(col))
    a2l, l2a = {'BENIGN': 0}, {0: 'BENIGN'}
    all_labels.remove('BENIGN')
    for i, att in enumerate(all_labels):
        a2l[att] = i + 1
        l2a[i + 1] = att
    return a2l, l2a


def load_cicids(subset):
    df = pd.read_csv(os.path.join(CICIDS_DIR, CICIDS_DICT[subset] + '.pcap_ISCX.csv'))
    df = df[CICIDS_IP_COLS + CICIDS_FEAT_COLS + [CICIDS_LABEL_COL]]
    df.dropna(how='any', inplace=True)
    # df.drop(df[df.sum(axis=1) == np.inf].index, inplace=True)

    # only include two web servers' external comms
    # cond = df[' Source IP'].isin(CICIDS_SERVER_IPS) | df[' Destination IP'].isin(CICIDS_SERVER_IPS)
    cond = df[' Destination IP'].isin(CICIDS_SERVER_IPS)
    df = df[cond | (df[CICIDS_LABEL_COL] != 'BENIGN')]
    # cond = (df[' Source IP'].str.startswith('192.168.10') & df[' Destination IP'].str.startswith('192.168.10'))
    # df = df[(~cond) | (df[CICIDS_LABEL_COL] != 'BENIGN')]

    X = df[CICIDS_FEAT_COLS].to_numpy()
    a2l, l2a = encode_label_cicids(df[CICIDS_LABEL_COL])
    y = df[CICIDS_LABEL_COL].apply(lambda x: a2l[x]).to_numpy()

    return X, y


def load_cicids_custom(subset):
    # df = pd.read_csv(os.path.join(CUSTOM_DATA_DIR, 'CICIDS-2017', f'{subset}.csv'))
    df = pd.read_csv(os.path.join('./dataset', 'CICIDS-2017', f'{subset}.csv'))

    # only include two web servers' external comms
    cond = df['dest-ip'].isin(CICIDS_SERVER_IPS)
    df = df[cond | (df[CUSTOM_LABEL_COL] != 0)]

    X = df[CUSTOM_FEAT_COLS].to_numpy()
    y = df[CUSTOM_LABEL_COL].to_numpy()

    return X, y

# CICIDS-improved
def load_cicids_improved(subset, **kwargs):
    subset = str(subset).lower()
    df = pd.read_csv(os.path.join(CICIDS_2_DIR, subset + '.csv'))
    try:
        feat_size = kwargs['feat_size']
        # print(f'feat_size: {feat_size}')
        df = df[CICIDS_2_IP_COLS + CICIDS_2_FEAT_ALL_COLS[:feat_size] + [CICIDS_2_LABEL_COL, CICIDS_2_ATTEMPT_COL]]
        columns_to_extract = CICIDS_2_FEAT_ALL_COLS[:feat_size]
    except:
        df = df[CICIDS_2_IP_COLS + CICIDS_2_FEAT_COLS + [CICIDS_2_LABEL_COL, CICIDS_2_ATTEMPT_COL]]
        columns_to_extract = CICIDS_2_FEAT_COLS

    # filter attempted
    df = df[df[CICIDS_2_ATTEMPT_COL] == -1]

    # only include 3 servers' external comms
    cond = df[CICIDS_2_IP_COLS[1]].isin(CICIDS_2_SERVER_IPS)
    # cond = df[CICIDS_2_IP_COLS[0]].isin(CICIDS_2_CLIENT_IPS)
    df = df[cond | (df[CICIDS_2_LABEL_COL] != 'BENIGN')]

    X = df[columns_to_extract].to_numpy()
    y = df[CICIDS_2_LABEL_COL].apply(lambda x: x != 'BENIGN').astype('int').to_numpy()

    return X, y

# CSE-CICIDS-2018-improved
def load_cse_improved(subset, **kwargs):
    subset = str(subset).lower()
    df = pd.read_csv(os.path.join(CSE_DIR, subset + '.csv'))
    try:
        feat_size = kwargs['feat_size']
        # print(f'feat_size: {feat_size}')
        df = df[CICIDS_2_IP_COLS + CICIDS_2_FEAT_ALL_COLS[:feat_size] + [CICIDS_2_LABEL_COL, CICIDS_2_ATTEMPT_COL]]
        columns_to_extract = CICIDS_2_FEAT_ALL_COLS[:feat_size]
    except:
        df = df[CICIDS_2_IP_COLS + CICIDS_2_FEAT_COLS + [CICIDS_2_LABEL_COL, CICIDS_2_ATTEMPT_COL]]
        columns_to_extract = CICIDS_2_FEAT_COLS

    # filter attempted
    df = df[df[CICIDS_2_ATTEMPT_COL] == -1]

    # only include 2 servers' external comms
    cond = df[CICIDS_2_IP_COLS[1]].isin(CSE_SERVER_IPS)
    # cond = df[CICIDS_2_IP_COLS[0]].isin(CICIDS_2_CLIENT_IPS)
    df = df[cond | (df[CICIDS_2_LABEL_COL] != 'BENIGN')]

    X = df[columns_to_extract].to_numpy()
    y = df[CICIDS_2_LABEL_COL].apply(lambda x: x != 'BENIGN').astype('int').to_numpy()

    return X, y

# UNSW-NB15
def encode_label_unsw(col: pd.Series):
    all_labels = list(set(col))
    a2l, l2a = {'Normal': 0}, {0: 'Normal'}
    all_labels.remove('Normal')
    for i, att in enumerate(all_labels):
        a2l[att] = i + 1
        l2a[i + 1] = att
    return a2l, l2a

def load_unsw(subset):
    df = pd.read_csv(UNSW_DICT[subset])
    df.dropna(how='any', inplace=True)

    df = df[(df['proto'] == 'tcp') | (df['proto'] == 'udp')]

    X = df[UNSW_FEAT_COLS].to_numpy()
    # a2l, l2a = encode_label_unsw(df[UNSW_CAT_COL])
    # y = df[UNSW_CAT_COL].apply(lambda x: a2l[x]).to_numpy()
    y = df[UNSW_LABEL_COL].to_numpy()
    return X, y


def load_toniot_custom(subset):
    # df = pd.read_csv(os.path.join(CUSTOM_DATA_DIR, 'TON-IoT', f'{subset}.csv'))
    df = pd.read_csv(os.path.join('./dataset', 'TON-IoT', f'{subset}.csv'))

    # cond = df['src_ip'].isin(TONIOT_SERVER_IPS) | df['dst_ip'].isin(TONIOT_SERVER_IPS) 
    # df = df[cond | (df[CUSTOM_LABEL_COL] != 0)]
    # cond1 = (df['dur'] > 0)
    # cond2 = df['dst_ip'].apply(lambda x: IP(x) < IP('224.0.0.0/4'))
    # df = df[cond1 & cond2]
    # cond3 = df['dst_ip'].str.startswith('192.168.1')
    # df = df[~cond3 | (df[CUSTOM_LABEL_COL] == 0)]
    
    df_att = df[df['label'] == 1]

    df_list = [df_att]
    for f in os.listdir(os.path.join('./dataset', 'TON-IoT')): 
        if f.startswith('normal'):
            df_norm = pd.read_csv(os.path.join('./dataset', 'TON-IoT', f))#CUSTOM_DATA_DIR
            cond = df_norm['src_ip'].isin(['3.122.49.24']) | df_norm['dst_ip'].isin(['3.122.49.24']) # TONIOT_IPS
            df_norm = df_norm[cond]
            df_list.append(df_norm)
    df = pd.concat(df_list)

    X = df[CUSTOM_FEAT_COLS].to_numpy()
    y = df[CUSTOM_LABEL_COL].to_numpy()

    return X, y