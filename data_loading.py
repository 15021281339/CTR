import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from dummyPy import OneHotEncoder
import random


class Data():
    """
    features are used for category
        C1:           int,     1001, 1002, ...
        banner_pos:   int,     0,1,2,3,...
        site_domain:  object,  large set of object variables
        site_id:      object,  large set of object variables
        site_category:object,  large set of object variables
        app_id:       object,  large set of object variables
        app_category: object,  small set of object variables
        device_type:  int,     0,1,2,3,4
        device_conn_type:int,  0,1,2,3
        C14:          int,     small set of int variables
        C15:          int,     ...
        C16:          int,     ...
    """
    def __init__(self, datapath, cols):
        self.datapath = datapath
        self.fp_train = self.datapath + 'train.csv'
        self.fp_test = self.datapath + "test.csv"
        self.cols = cols
        self.train_cols = ['id', 'click'].extend(cols)
        self.test_cols = ['id'].extend(cols)


    def data_read(self):

        df_train_ini = pd.read_csv(self.fp_train, nrows=10)
        # df_train_org = pd.read_csv(self.fp_train, chunksize=1000000, iterator=True)
        # df_test_org = pd.read_csv(self.fp_test,  chunksize=1000000, iterator=True)
        df_train_org = pd.read_csv(self.fp_train, dtype={'id': str}, chunksize=1000000, iterator=True)
        df_test_org = pd.read_csv(self.fp_test, dtype={'id': str}, chunksize=1000000, iterator=True)

        # %% 统计类别特征的类别数
        print('starting counting frequency...')
        cols_counts = {}
        for col in self.cols:
            cols_counts[col] = df_train_ini[col].value_counts()   # 初始化字典除了id和click特征
        for chunk in df_train_org:
            for col in self.cols:
                cols_counts[col] = cols_counts[col].append(chunk[col].value_counts())  # chunk迭代存储新的计数

        for chunk in df_test_org:
            for col in self.cols:
                cols_counts[col] = cols_counts[col].append(chunk[col].value_counts())

        for col in self.cols:
            cols_counts[col] = cols_counts[col].groupby(cols_counts[col].index).sum()  # 把重复的汇总 然后排序
            cols_counts[col] = cols_counts[col].sort_values(ascending=False)

        fp_col_counts = "data/col_counts"  # 存储字典路径
        pickle.dump(cols_counts, open(fp_col_counts, 'wb'))

        # %% 查看特征分布
        print('show feature distribution...')
        fig = plt.figure(1)
        for i, col in enumerate(self.cols):
            ax = fig.add_subplot(4, 3, i+1)
            ax.fill_between(np.arange(len(cols_counts[col])), cols_counts[col].values)
        plt.show()

        # %% 特征数值处理（稀疏化的取值统一归others类）
        print('starting handle raw frequent feat...')
        k = 99  # 取前99的高频类别
        self.cols_index = {}
        for col in self.cols:
            self.cols_index[col] = cols_counts[col][0:k].index   # 取取前99的index


        # %% 构造新的数据集
        print('starting creating new data...')
        hd_flag = True
        self.fp_train_f = "data/train_f.csv"  # 新训练集 存储路径
        for chunk in df_train_org:
            df = chunk.copy()
            for col in self.cols:
                df[col] = df[col].astype('object')
                df.loc[~df[col].isin(self.cols_index[col]), col] = 'other'  # 将元数据中的稀疏变量取值 赋值为other
            with open(self.fp_train_f, 'a') as f:
                df.to_csv(f, columns=self.train_cols, header=hd_flag, index=False)  # 第一个chunk保存header
            hd_flag = False

        self.fp_test_f = "./data/test_f.csv"
        hd_flag = True
        for chunk in df_test_org:
            df = chunk.copy()
            for col in cols:
                df[col] = df[col].astype('object')
                df.loc[~df[col].isin(self.cols_index[col]), col] = 'other'
            with open(self.fp_test_f, 'a') as f:
                df.to_csv(f, columns=self.test_cols, header=hd_flag, index=False)
            hd_flag = False

    # %% 特征编码
    def data_enc(self):
        fp_lb_enc = self.datapath + 'lb_enc'  # label encoding
        fp_oh_enc = self.datapath + "oh_enc"  # one-hot encoding
        df_train_f = pd.read_csv(self.fp_train_f, index_col=None, chunksize=500000, iterator=True)
        df_test_f = pd.read_csv(self.fp_test_f, index_col=None, chunksize=500000, iterator=True)

        lb_enc = {}
        # for col in self.cols:
        #     self.cols_index[col] = np.append(self.cols_index[col], 'other')  # 添加新 other
        # for col in self.cols:
        #     lb_enc[col] = LabelEncoder()
        # %% one-hot and label encode
        print('starting one-hot and label encoding...')
        oh_enc = OneHotEncoder(self.cols)
        for chunk in df_train_f:
            oh_enc.fit(chunk)    # dummy的one-hot不会
            # for col in self.cols:
            #     lb_enc[col].fit(chunk[col])  # fit会重置
        for chunk in df_test_f:
            oh_enc.fit(chunk)
            # for col in self.cols:
            #     lb_enc[col].fit(chunk[col])
        # pickle.dump(lb_enc, open(fp_lb_enc, 'wb'))
        pickle.dump(oh_enc, open(fp_oh_enc, 'wb'))

    def run(self):
        self.data_read()
        self.data_enc()


if __name__ == '__main__':
    cols = ['C1',
            'banner_pos',
            'site_domain',
            'site_id',
            'site_category',
            'app_id',
            'app_category',
            'device_type',
            'device_conn_type',
            'C14',
            'C15',
            'C16',
            ]
    # site_domain       10000 non-null  object
    #  7   site_category     10000 non-null  object
    #  8   app_id            10000 non-null  object
    #  9   app_domain        10000 non-null  object
    #  10  app_category      10000 non-null  object
    #  11  device_id         10000 non-null  object
    #  12  device_ip         10000 non-null  object
    #  13  device_model      10000 non-null  object
    #  14  device_type       10000 non-null  int64
    #  15  device_conn_type  10000 non-null  int64
    #  16  C14               10000 non-null  int64
    #  17  C15               10000 non-null  int64
    #  18  C16               10000 non-null  int64
    #  19  C17               10000 non-null  int64
    #  20  C18               10000 non-null  int64
    #  21  C19               10000 non-null  int64
    #  22  C20               10000 non-null  int64
    #  23  C21
    new_data = Data('data/', cols)
    new_data.run()