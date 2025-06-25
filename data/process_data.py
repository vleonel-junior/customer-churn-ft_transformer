from typing import Any, Dict

import numpy as np
# import rtdl
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch


from sklearn.preprocessing import OneHotEncoder
import numpy as np 
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_data(path):

    online_df = pd.read_csv(path)
    online_df.head()
    online_df_2 = online_df.drop(['Revenue'], axis=1)
    data = online_df_2.to_numpy()
    numerical = data[:,:10]

    # Categorical data
    ohe = OneHotEncoder()
    month_onehot = ohe.fit_transform(online_df[['Month']]).toarray()
    operatingsystems_onehot = ohe.fit_transform(online_df[['OperatingSystems']]).toarray()
    browser_onehot = ohe.fit_transform(online_df[['Browser']]).toarray()
    region_onehot = ohe.fit_transform(online_df[['Region']]).toarray()
    trafficType_onehot = ohe.fit_transform(online_df[['TrafficType']]).toarray()
    visitorType_onehot = ohe.fit_transform(online_df[['VisitorType']]).toarray()
    weekend_onehot = ohe.fit_transform(online_df[['Weekend']]).toarray()

    categorical = np.concatenate((month_onehot,
                        operatingsystems_onehot,
                        browser_onehot,
                        region_onehot,
                        trafficType_onehot,
                        visitorType_onehot,
                        weekend_onehot), axis=1)

    encoded_data = np.concatenate((numerical, categorical), axis=-1)
    label = online_df['Revenue'].astype(int).to_numpy()

    X_all = encoded_data.astype('float32')
    y_all = label.astype('int64')

    return X_all, y_all

def get_data(seed):
    X_all, y_all = read_data('./data/online_shoppers_intention.csv')
    X = {}
    y = {}
    X['train'], X['test'], y['train'], y['test'] = sklearn.model_selection.train_test_split(
        X_all, y_all, train_size=0.8,random_state=seed
    )
    X['train'], X['val'], y['train'], y['val'] = sklearn.model_selection.train_test_split(
        X['train'], y['train'], train_size=0.85, random_state=0
    )
    print(len(y['train']), len(y['val']), len(y['test']))

    # Tỉ lệ train - val - test. (stratefied hay random)
    # not the best way to preprocess features, but enough for the demonstration
    preprocess = sklearn.preprocessing.StandardScaler().fit(X['train'])
    X = {
        k: torch.tensor(preprocess.transform(v), device=device)
        for k, v in X.items()
    }
    y = {k: torch.tensor(v, device=device) for k, v in y.items()}

    return X, y, X_all, y_all


