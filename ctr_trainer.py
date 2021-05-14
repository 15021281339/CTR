import pandas as pd
import numpy as np
from dummyPy import OneHotEncoder  # for one-hot encoding on a large scale of chunks
from sklearn.linear_model import SGDClassifier  # using SGDClassifier for training incrementally
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import pickle


class CtrTrainer():
    def __init__(self, train_path, test_path, oh_path):
        self.train_path = train_path
        self.test_path = test_path
        self.oh_path = oh_path

    def load_data(self):
        self.df_train = pd.read_csv(self.train_path, dtype={'id': str},
                                    index_col=None,
                                    chunksize=10000, iterator=True)
        self.oh_enc = pickle.load(open(self.oh_path, 'rb'))

    def train(self):
        fp_lr_model = "./data/lr_model"

        lr_model = SGDClassifier(loss='log')   # using log-loss for lr  early_stopping=False with partial_fit
        self.scores = []

        for i, chunk in enumerate(self.df_train):
            print('starting {} chunk...'.format(i+1))
            df_train = self.oh_enc.transform(chunk)  # 转换为one-hot编码
            # other_feat = ['device_model', 'device_ip', 'device_id', 'app_domain', 'hour']  # one-hot编码没有考虑的object特征也要删除
            # ['id', 'click'].extend(other_feat)
            feat_train = df_train.columns.drop(['id', 'click', 'device_model',
                                                'device_ip', 'device_id',
                                                'app_domain', 'hour',
                                                'C17', 'C18',
                                                'C19', 'C20', 'C21'])
            train_x = df_train[feat_train]
            train_y = df_train['click'].astype('int')
            lr_model.partial_fit(train_x, train_y, classes=[0, 1])

            y_pred = lr_model.predict_proba(train_x)[:, 1]
            score = log_loss(train_y, y_pred)
            self.scores.append(score)
        print('saving model...')
        pickle.dump(lr_model, open(fp_lr_model, 'wb'))

    def show_training_curve(self):
        f1 = plt.figure(1)
        plt.plot(self.scores)
        plt.xlabel('iterations')
        plt.ylabel('log_loss')
        plt.title('log_loss of training')
        plt.grid()
        plt.show()

    def run(self):
        self.load_data()
        self.train()
        self.show_training_curve()

if __name__ == '__main__':
    newtrainer = CtrTrainer(train_path='data/sub_train_f_3m.csv',
                            test_path='data/test_f.csv',
                            oh_path='data/oh_enc')
    newtrainer.run()