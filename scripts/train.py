import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scripts.scr import *


def train_model() -> None:
    df_train = pd.read_csv('./data/train_initial.csv')
    del df_train['Unnamed: 0']

    df_train['city'] = df_train['address_rus'].apply(get_city)
    df_train['region'] = df_train['address_rus'].apply(get_region)
    df_train = df_train.dropna()

    X_train, X_test, y_train, y_test = train_test_split(
        pd.get_dummies(df_train).drop(['id', 'target'], axis=1),
        pd.get_dummies(df_train)['target'],
        test_size=0.2,
        random_state=13)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predicts = model.predict(X_test)

    err = mean_absolute_error(y_test, predicts)

    df_mae = pd.DataFrame({'MAE': [err]})
    df_mae.to_csv('./outputs/mae.csv', index=False)