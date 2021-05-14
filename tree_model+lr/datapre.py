import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import log_loss
from copy import copy
# %% binary class data
class DataPreProcess():
    def __init__(self, data_path=r'../data/'):
        self.data_path = data_path
        print('data loading...')
        df_train = pd.read_csv(self.data_path + 'train.csv')
        df_test = pd.read_csv(self.data_path + 'test.csv')

        df_train.drop(['Id'], axis=1, inplace=True)
        df_test.drop(['Id'], axis=1, inplace=True)

        df_test['Label'] = -1   # 消除测试标签
        self.data = pd.concat([df_train, df_test])
        self.data = self.data.fillna(-1)   # 空值补-1
        self.data.to_csv(self.data_path + 'data.csv', index=False)  # 保存所有数据
        print('data saved')

    def map_(self, category_feature, continuous_feature):

        print('starting mapping...')  # 可以不用one-hot
        for col in category_feature:
            onehot_feats = pd.get_dummies(self.data[col], prefix=col)
            self.data.drop([col], axis=1, inplace=True)
            self.data = pd.concat([self.data, onehot_feats], axis=1)
        data = self.data.copy()
        data = data.drop(['Label'], axis=1)

        # %% 训练集提取
        train = self.data[self.data['Label'] != -1]
        target = train.pop('Label')  # 训练集的目标y

        # %% 测试机提取
        test_x = self.data[self.data['Label'] == -1]
        test_x = test_x.drop(['Label'], axis=1)
        test = test_x.copy()

        print('data splitting...')
        x_train, x_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=2018)

        return x_train, x_val, y_train, y_val, target, test, data


if __name__ == '__main__':
    data = DataPreProcess()
    continuous_feature = ['I'] * 13
    continuous_feature = [col + str(i + 1) for i, col in enumerate(continuous_feature)]
    category_feature = ['C'] * 26
    category_feature = [col + str(i + 1) for i, col in enumerate(category_feature)]
    x_train, x_val, y_train, y_val, target, test, data = data.map_(category_feature, continuous_feature)
    print(x_train.shape, pd.concat([x_train, x_val]).shape[0], type(x_val), type(data))
