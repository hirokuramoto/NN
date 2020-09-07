import pandas as pd
import numpy as np


def load_data(filename, train_size, test_size, design, object, normalize=True):
    '''データセットの読み込み
    args
        normalize(bool) : 標準化
        filename(str)   : データセットのファイルパス
        train_size(int) : 訓練データの個数
        test_size(int)  : テストデータの個数
        design(int)     : 設計変数の数
        object(int)     : 目的関数の数
    '''
    data = pd.read_csv(filename, header = 0) # header = 0でヘッダー行を読み飛ばし

    x_train = np.array(data.iloc[0          : train_size          , 0      : design         ])
    t_train = np.array(data.iloc[0          : train_size          , design : design + object])
    x_test  = np.array(data.iloc[train_size : train_size+test_size, 0      : design         ])
    t_test  = np.array(data.iloc[train_size : train_size+test_size, design : design + object])

    if normalize:
        mean = x_train.mean(axis=0)
        std  = x_train.std(axis=0)
        x_train = (x_train - mean) / std
        x_test  = (x_test  - mean) / std

    return (x_train, t_train), (x_test, t_test)
