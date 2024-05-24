import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from catboost import Pool
from sklearn.metrics import f1_score
from datetime import datetime

from scripts.scr import *

DEFAULT_RANDOM_SEED = 42

def train_model(
        learning_rate:float=0.003,
        iterations:int=30000,
        early_stopping_rounds:int=2000,
        task_type:str='CPU'
        ) -> None:
    
    seedBasic(DEFAULT_RANDOM_SEED)

    df = pd.read_csv('./data/train_full.csv', index_col='Unnamed: 0')

    df['target'] = minmax_scale(df['target'], feature_range=(1, 5), axis=0)
    df['target'] = df['target'].apply(np.round).apply(int)

    cat_features=[
    'atm_group', 'city', 'city_area', 
    'city_district', 'federal_district',
    'administrative'
    ]

    df[cat_features] = df[cat_features].fillna('no_data')
    df['administrative'] = df['administrative'].apply(str)
    df['atm_group'] = df['atm_group'].apply(str)
    df = df.fillna(0)

    df.reset_index(drop=True, inplace=True)

    df_train_val = df.drop(
        ['id', 'address', 'address_rus','target'], axis=1
        )
    y_train_val = df['target']

    X_train, X_val, y_train, y_val = train_test_split(
        df_train_val, y_train_val, 
        shuffle=True, 
        stratify=y_train_val, 
        train_size=0.85,
        random_state=DEFAULT_RANDOM_SEED
        )
    
    train_pool = Pool(
        X_train, y_train,
        cat_features=[
            'atm_group', 'city', 'city_area', 
            'city_district', 'federal_district',
            'administrative'
            ]
            )

    validation_pool = Pool(
        X_val, y_val,
        cat_features=[
            'atm_group', 'city', 'city_area', 
            'city_district', 'federal_district',
            'administrative'
            ]
            )

    print('Train dataset shape: {}\n'.format(train_pool.shape))

    model = fit_model(train_pool, validation_pool, 
                      learning_rate, iterations,
                      early_stopping_rounds, task_type)

    err = f1_score(y_val, model.predict(X_val), average='micro')
    df_f1 = pd.DataFrame({'f1_score': [err]})
    df_f1.to_csv(f'./outputs/f1_score_{err}.csv', index=False)
    today_date = datetime.today().strftime('%d-%m-%Y')

    model_name = f'catboost_model_f1:{err}_date:{today_date}'

    model.save_model(
        f'./models/catboost_model_f1:{err}_date:{today_date}'
        )
    
    return err, model_name