import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from config import CONFIG
from datapre import DataPreProcess
from sklearn.metrics import log_loss
import os
import warnings
warnings.filterwarnings('ignore')
os.getcwd()


# %% gbdt+lr
class GbdtLr():
    def __init__(self):
        self.learning_rate = CONFIG['tree_lr']['gbdt_params']['learning_rate']
        self.n_estimators = CONFIG['tree_lr']['gbdt_params']['n_estimators']
        self.subsample = CONFIG['tree_lr']['gbdt_params']['subsample']
        self.criterion = 'friedman_mse'
        self.min_samples_split = CONFIG['tree_lr']['gbdt_params']['min_samples_split']
        # self.min_samples_leaf = 1,
        # self.min_weight_fraction_leaf = 0.
        self.max_depth = CONFIG['tree_lr']['gbdt_params']['max_depth']

    def gbdt_fit(self, data_x, data_y, all_data):
        new_gbdt = GradientBoostingClassifier(n_estimators=self.n_estimators,   # 可以调lgb包 param：boosting='gbdt'
                                              random_state=10,
                                              subsample=self.subsample,
                                              max_depth=self.max_depth,
                                              min_samples_split=self.min_samples_split,
                                              )
        new_gbdt.fit(data_x, data_y)
        # alldata_x = pd.concat([data_x, test_x])
        train_new_fea = new_gbdt.apply(all_data)  # 返回所有样本每棵树的叶子节点索引
        train_new_fea = train_new_fea.reshape(-1, self.n_estimators)
        enc = OneHotEncoder()
        enc.fit(train_new_fea)   # 每一个tree都会得到一个one-hot transform
        train_new_embeddings = np.array(enc.transform(train_new_fea).toarray())
        with open('gbdtlr_gbdt.pkl', 'wb') as f:
            pickle.dump(new_gbdt, f)
        return train_new_embeddings   # 新输入特征样本

    def lr_fit(self, train_x, train_y, val_x, val_y):
        new_lr = LogisticRegression()
        new_lr.fit(train_x, train_y)

        self.train_logloss = log_loss(train_y, new_lr.predict_proba(train_x)[:, 1])
        print('training logloss:', self.train_logloss)
        self.val_logloss = log_loss(val_y, new_lr.predict_proba(val_x)[:, 1])
        print('validating logloss: ', self.val_logloss)
        auc = roc_auc_score(val_y, new_lr.predict_proba(val_x)[:, 1])
        print("validating auc:", auc)

        with open('gbdtlr_lr.pkl', 'wb') as f:
            pickle.dump(new_lr, f)

    def predict(self, test_x):
        with open('gbdtlr_lr.pkl', 'rb') as f2:
            model_lr = pickle.load(f2)

        # %% lr预测
        y_pre = model_lr.predict_proba(test_x)[:, 1]

        print('写入结果...')
        res = pd.read_csv('../data/test.csv')
        submission = pd.DataFrame({'Id': res['Id'], 'Label': y_pre})
        # if not ex
        submission.to_csv(r'submission_gbdt+lr_trlogloss_%s_vallogloss_%s.csv'
                          % (self.train_logloss, self.val_logloss),
                          index=False)
        print('结束')


# %% xgb+lr
class XgboostLr():
    def __init__(self):
        self.learning_rate = CONFIG['tree_lr']['xgb_params']['lr']
        self.n_estimators = CONFIG['tree_lr']['xgb_params']['n_estimators']
        self.gamma = CONFIG['tree_lr']['xgb_params']['gamma']
        self.min_child_weight = CONFIG['tree_lr']['xgb_params']['min_child_weight']
        # self.gamma = 1,
        self.base_score = CONFIG['tree_lr']['xgb_params']['base_score']
        self.max_depth = CONFIG['tree_lr']['xgb_params']['max_depth']

    def xgb_fit(self, data_x, data_y, all_data):
        new_xgb = XGBClassifier(n_estimators=self.n_estimators,   # 可以调lgb包 param：boosting='gbdt'
                                random_state=10,
                                gamma=self.gamma,
                                max_depth=self.max_depth,
                                min_child_weight=self.min_child_weight,
                                base_score=self.base_score,
                                colsample_bytree=0.5
                                )
        new_xgb.fit(data_x, data_y)
        # alldata_x = pd.concat([data_x, test_x])
        train_new_fea = new_xgb.apply(all_data)  # 返回所有样本每棵树的叶子节点索引
        train_new_fea = train_new_fea.reshape(-1, self.n_estimators)
        enc = OneHotEncoder()
        enc.fit(train_new_fea)   # 每一个tree都会得到一个one-hot transform
        train_new_embeddings = np.array(enc.transform(train_new_fea).toarray())
        with open('xgblr_xgb.pkl', 'wb') as f:
            pickle.dump(new_xgb, f)
        return train_new_embeddings   # 新输入特征样本

    def lr_fit(self, train_x, train_y, val_x, val_y):
        new_lr = LogisticRegression()
        new_lr.fit(train_x, train_y)

        self.train_logloss = log_loss(train_y, new_lr.predict_proba(train_x)[:, 1])
        print('training logloss:', self.train_logloss)
        self.val_logloss = log_loss(val_y, new_lr.predict_proba(val_x)[:, 1])
        print('validating logloss: ', self.val_logloss)
        auc = roc_auc_score(val_y, new_lr.predict_proba(val_x)[:, 1])
        print("validating auc:", auc)
        with open('xgblr_lr.pkl', 'wb') as f:
            pickle.dump(new_lr, f)

    def predict(self, test_x):
        # with open('gbdtlr_gbdt.pkl', 'rb') as f1:
        #     model_gbdt = pickle.load(f1)
        with open('xgblr_lr.pkl', 'rb') as f2:
            model_lr = pickle.load(f2)

        # %% lr预测
        y_pre = model_lr.predict_proba(test_x)[:, 1]

        print('写入结果...')
        res = pd.read_csv('../data/test.csv')
        submission = pd.DataFrame({'Id': res['Id'], 'Label': y_pre})
        # if not ex
        submission.to_csv(r'submission_xgb+lr_trlogloss_%s_vallogloss_%s.csv'
                          % (self.train_logloss, self.val_logloss),
                          index=False)
        print('结束')


if __name__ == '__main__':
    # n_estimator = 10
    # data_generator
    data = DataPreProcess()
    continuous_feature = ['I'] * 13
    continuous_feature = [col + str(i + 1) for i, col in enumerate(continuous_feature)]
    category_feature = ['C'] * 26
    category_feature = [col + str(i + 1) for i, col in enumerate(category_feature)]
    x_train, x_val, y_train, y_val, target, test, data = data.map_(category_feature, continuous_feature)
    train_len = pd.concat([x_train, x_val]).shape[0]

    # gbdt_lr = GbdtLr()
    # x_fea = gbdt_lr.gbdt_fit(x_train, y_train, data)  # 所有数据的x_fea
    xgb_lr = XgboostLr()
    x_fea = xgb_lr.xgb_fit(x_train, y_train, data)  # 所有数据的x_fea
    print(x_fea.shape)
    train_lr = x_fea[:train_len]
    test_lr = x_fea[train_len:]
    # print(train_lr.shape, test_lr.shape, target.shape)
    del x_fea

    # gbdt-lr
    # x_train_lr, x_val_lr, y_train_lr, y_val_lr = train_test_split(train_lr, target, test_size=0.2, random_state=2018)
    # gbdt_lr.lr_fit(x_train_lr, y_train_lr, x_val_lr, y_val_lr)
    # gbdt_lr.predict(test_lr)

    # xgb-lr
    x_train_lr, x_val_lr, y_train_lr, y_val_lr = train_test_split(train_lr, target, test_size=0.2, random_state=2018)
    xgb_lr.lr_fit(x_train_lr, y_train_lr, x_val_lr, y_val_lr)
    xgb_lr.predict(test_lr)



