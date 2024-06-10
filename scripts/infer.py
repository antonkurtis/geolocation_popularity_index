import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from catboost import Pool
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
import glob
import json

from scripts.scr import *

DEFAULT_RANDOM_SEED = 42

def inference_model(lats:list, longs:list, atm:str) -> list:
    '''
    Функция инференса модели.

    lats: list список широт точек размещения коорднат
    longs: list список долготы точек размещения коорднат
    atm: str наименование банка-владельца банкомата
    '''
    
    seedBasic(DEFAULT_RANDOM_SEED)

    fls = [x.split('/')[-1] for x in glob.glob('./models/*')]

    best_score = 0
    for mdl in fls:
        score = float(mdl.split('.c')[0].split('_')[2].split(':')[-1])
        if score > best_score:
            best_score = score
            best_model = mdl

    s3_model_name, s3_score = best_model_s3()

    if s3_score > best_score:
        model_path = load_model_s3(s3_model_name)
        model = CatBoostClassifier()
        model.load_model(model_path)
    else:
        model = CatBoostClassifier()
        model.load_model(f'./models/{best_model}')

    df = pd.DataFrame({'lat':lats, "long":longs, 'atm_group':atm})
    df = get_area_features(df)

    with open('./data/objects.json') as json_file:
        data = json.load(json_file)

    df = get_objects(data, df)
    df = get_population(df)

    cat_features=[
        'atm_group', 'city', 'city_area',
        'city_district', 'federal_district',
        'administrative'
        ]

    df.replace('', np.nan, inplace=True)
    df[cat_features] = df[cat_features].fillna('no_data')
    df['administrative'] = df['administrative'].apply(str)
    df['atm_group'] = df['atm_group'].apply(str)
    df = df.fillna(0)

    df.reset_index(drop=True, inplace=True)

    test_pool = Pool(
        df,
        cat_features=cat_features
        )

    predict = model.predict(test_pool)
    df['predict'] = predict

    return predict