import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from dummyPy import OneHotEncoder  # for bigger one-hot
import random
from sklearn.metrics import log_loss
import gc
import xlearn as xl
from config import CONFIG


class FMTrainer():
    def __init__(self, train_path=None ,test_path=None,
                 oh_path=None, fp_lb_enc=None,
                 fp_train=None, fp_valid=None, fp_test=None, fp_fm_model=None):
        self.train_path = train_path  # 原数据路径
        self.test_path = test_path
        # self.oh_path = oh_path
        # self.fp_lb_enc = fp_lb_enc
        self.fp_fm_model = fp_fm_model  # 训练好的model 存储路径
        self.fp_train = fp_train  # convert 后的数据存储路径
        self.fp_valid = fp_valid
        self.fp_test = fp_test
    def load_data(self):
        df_train = pd.read_csv(self.train_path,)
        self.cols = ['C1',
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
                        'C16']
        cols_train = ['id', 'click']
        cols_test = ['id']
        cols_train.extend(self.cols)
        cols_test.extend(self.cols)

        # label encoding 后 构造新的数据
        label_enc = pickle.load(open(self.fp_lb_enc, 'rb'))
        df_train = pd.read_csv(self.train_path)
        df_train = df_train[cols_train].copy()

        print('starting encoding...')
        for col in self.cols:
            df_train[col] = label_enc[col].transform(df_train[col].values)
        # df_test = pd.read_csv(self.test_path)
        # for col in self.cols:
        #     df_test[col] = label_enc[col].transform(df_test[col].values)
        # df_test['click'] = -1   # add a placeholder
        self.df = df_train
        # 拼接训练和测试集
        self.n_train = len(df_train)
        # n_test = len(df_test)
        # self.df = df_train.append(df_test)
        # del df_train, df_test
        # gc.collect()

    # %% 将数据集转换为libffm格式
    def convert_to_ffm(self, df, numerics, categories, label='click', train_size=0.7):

        catdict = {}   # 用于判断特征类型
        catcodes = {}   # 用于存储类别特征的 label encode 形式的编码
        # %% flag categorical and numerical fields
        for x in numerics:
            catdict[x] = 0
        for x in categories:
            catdict[x] = 1

        nrows = df.shape[0]
        # ncolumns = len(featurse)
        n1 = self.n_train * train_size

        print('saving libffm...')
        with open(self.fp_train, "w") as file_train, \
                open(self.fp_valid, "w") as file_valid, \
                open(self.fp_test, "w") as file_test:
            # 转换每一行样本为libffm
            for n, r in enumerate(range(nrows)):
                datastring = ""   # 新样本
                datarow = df.iloc[r].to_dict()  # 每一行样本字典化
                datastring += str(int(datarow[label]))  # 先打印标签

                # 判断特征类型
                for i, x in enumerate(catdict.keys()):  # x = feature_name
                    if catdict[x] == 0:
                        # numerical 可以进行特征归一化
                        datastring = datastring + " " + str(i) + ":" + str(i) + ":" + str(datarow[x])
                    else:
                        datastring = datastring + " " + str(i) + ":" + str(int(datarow[x])) + ":1"
                datastring += '\n'
                # 存储
                if n < n1: file_train.write(datastring)  # 训练集
                elif n < self.n_train: file_valid.write(datastring)  # 验证集
                # else: file_test.write(datastring)  # 测试集

    def train(self):
        # %% fm
        fm_model = xl.create_fm()
        fm_model.setTrain(self.fp_train)
        fm_model.setSigmoid()    # 二分类
        param = CONFIG['fm']['params']
        fp_pred_fm = "../data/fm/output_fm.txt"
        print('starting training...')
        fm_model.fit(param, self.fp_fm_model)

        fm_model.setValidate(self.fp_valid)
        fm_model.predict(self.fp_fm_model, fp_pred_fm)

        # with open(fp_pred_fm, 'r') as f:
        #     pre = pickle.load(f)
        # with open(self.fp_valid, 'r') as f:
        #     pre = pickle.load(f)
        # self.scores = log_loss(train_y, y_pred)

    def show_training_curve(self):
        f1 = plt.figure(1)
        plt.plot(self.scores)
        plt.xlabel('nums')
        plt.ylabel('log_loss')
        plt.title('log_loss of training')
        plt.grid()
        plt.show()


    def run(self):
        self.load_data()
        self.convert_to_ffm(self, self.df, numerics=[], categories=self.cols,
                            label='click', train_size=0.5)

        self.train()
            

if __name__ == '__main__':
    # fp_train = "../data/fm/train_ffm.txt"
    # fp_valid = "../data/fm/valid_ffm.txt"
    # fp_test = "../data/fm/test_ffm.txt"
    Trainer = FMTrainer(train_path='data/sub_train_f_1m.csv',
                        test_path='data/test_f.csv',
                        fp_lb_enc='data/lb_enc',
                        oh_path='data/oh_enc',
                        fp_train='data/fm/fm_train.csv',
                        fp_valid='data/fm/fm_valid.txt',
                        fp_test='data/fm/fm_test.txt',
                        fp_fm_model='data/fm/model_fm.out')
    Trainer.run()
    ## submissions
    # fp_sub_fm = "../data/fm/Submission_FM.csv"
    # fp_sub_ffm = "../data/fm/Submission_FFM.csv"