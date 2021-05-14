import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from sklearn.metrics import log_loss
from datapre import DataPreProcess
from tree_lr import XgboostLr, GbdtLr


class tree_lr_trainer():
    def __init__(self):
        pass

    def load_data(self):
        data = DataPreProcess()
        continuous_feature = ['I'] * 13
        continuous_feature = [col + str(i + 1) for i, col in enumerate(continuous_feature)]
        category_feature = ['C'] * 26
        category_feature = [col + str(i + 1) for i, col in enumerate(category_feature)]
        self.x_train, self.x_val, self.y_train, self.y_val, self.target, self.test, self.data = \
            data.map_(category_feature, continuous_feature)
        self.train_len = pd.concat([self.x_train, self.x_val]).shape[0]

    def gbdt_lr_train(self):
        # gbdt+lr
        gbdt_lr = GbdtLr()
        x_fea = gbdt_lr.gbdt_fit(self.x_train, self.y_train, self.data)  # 所有数据的x_fea
        train_lr = x_fea[:self.train_len]
        test_lr = x_fea[self.train_len:]
        del x_fea
        x_train_lr, x_val_lr, y_train_lr, y_val_lr = train_test_split(train_lr, self.target,
                                                                      test_size=0.2, random_state=2018)
        gbdt_lr.lr_fit(x_train_lr, y_train_lr, x_val_lr, y_val_lr)
        gbdt_lr.predict(test_lr)  # 预测

    def xgb_lr_train(self):
        xgb_lr = XgboostLr()
        x_fea = xgb_lr.xgb_fit(self.x_train, self.y_train, self.data)  # 所有数据的x_fea
        # print(x_fea.shape)
        train_lr = x_fea[:self.train_len]
        test_lr = x_fea[self.train_len:]
        # print(train_lr.shape, test_lr.shape, target.shape)
        del x_fea
        x_train_lr, x_val_lr, y_train_lr, y_val_lr = train_test_split(train_lr, self.target,
                                                                      test_size=0.2,
                                                                      random_state=2018)
        xgb_lr.lr_fit(x_train_lr, y_train_lr, x_val_lr, y_val_lr)
        xgb_lr.predict(test_lr)  # 预测
