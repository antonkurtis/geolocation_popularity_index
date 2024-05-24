from scripts.scr import get_area_features, get_objects, get_population
import pandas as pd
import json


def get_all_futures_dataset() -> None:
    '''
    Функция подгружает вводные данные из .csv 
    файла и собирает все необходимые признаки
    для каждой точки из датасета.
    '''
    df_train = pd.read_csv('./data/train_initial.csv')

    df_train = df_train.dropna(subset=['lat', 'long'])
    df_train = df_train.reset_index(drop=True)

    print('Area futures parsing start!')
    df = get_area_features(df_train)
    print('Area futures parsing DONE!!!')

    with open('./data/objects.json') as json_file:
        need_obj = json.load(json_file)

    print('Objects parsing start!')
    geo_data = get_objects(need_obj, df=df, flag=True)
    print('Objects parsing DONE!!!')

    print('Population parsing start!')
    geo_data = get_population(geo_data)
    print('Population parsing DONE!!!')

    geo_data.to_csv('./data/train_full.csv')