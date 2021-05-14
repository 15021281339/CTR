import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

def lr_ctr_predict():
    fp_test_f = 'data/test.csv'
    df_test_f = pd.read_csv(fp_test_f,
                            dtype={'id': str},
                            index_col=None,
                            chunksize=50000, iterator=True)

    fp_lr_model = "./data/lr_model"
    oh_path = 'data/oh_enc'

    lr_model = pickle.load(open(fp_lr_model, 'rb'))
    oh_enc = pickle.load(open(oh_path, 'rb'))
    fp_sub = "./data/LR_submission.csv"
    hd = True
    for chunk in df_test_f:
        df_test = oh_enc.transform(chunk)

        # ----- predict -----#
        feature_test = df_test.columns.drop(['id'])
        test_x = df_test[feature_test]
        y_pred = lr_model.predict_proba(test_x)[:, 1]  # get the probability of class 1

        # ----- generation of submission -----#
        chunk['click'] = y_pred
        with open(fp_sub, 'a') as f:
            chunk.to_csv(f, columns=['id', 'click'], header=hd, index=False)
        hd = False

    print(' - PY131 - ')
