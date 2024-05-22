import numpy as np
import pandas as pd
import osmnx as ox
from shapely.geometry import Polygon
from geopy.geocoders import Nominatim
from catboost import CatBoostClassifier
import random
import os
from catboost import Pool


DEFAULT_RANDOM_SEED = 42
    

def get_area_features(df:pd.DataFrame) -> pd.DataFrame:
    '''
    функция принимает на вход Pandas DataFrame с
    координатами точек, собирает по ним географические признаки
    и возвращает Pandas DataFrame
    '''
    addresses = []
    cities, city_areas, city_district, federal_district  = [], [], [], []

    geolocator = Nominatim(user_agent='your_app_name')
    coordinates = np.array([list(df.loc[idx][['lat', 'long']].tolist()) for idx in range(len(df))])
    
    for coordinate in coordinates:
        addresses.append(geolocator.reverse(coordinate))

    for add in addresses:
        try:
            cities.append(add.raw['address']['city'])
        except:
            cities.append('')
        try:
            city_areas.append(add.raw['address']['county'])
        except:
            city_areas.append('')
        try:
            city_district.append(add.raw['address']['quarter'])
        except:
            city_district.append('')
        try:
            federal_district.append(add.raw['address']['state'])
        except:
            federal_district.append('')

    df['city'] = cities
    df['city_area'] = city_areas
    df['city_district'] = city_district
    df['federal_district'] = federal_district

    return df


def get_objects(tags:list, df:pd.DataFrame, flag:bool = False) -> pd.DataFrame:
    '''
    функция принимает на вход Pandas DataFrame с
    координатами точек и географическими признаками,
    а так же список из признаков для парсинга (дома, магазины и тп),
    собирает по ним признаки.
    Объекты парсятся в радиусе 100 метров от заданной точки.
    Так же flag переключает сбор даных
    для обучение (flag=True) и для инференса.
    Возвращает Pandas DataFrame
    '''
    if flag:
        col_names = []

        for idx in range(len(tags)):
            col = list(tags[idx].values())[0]
            col_names.append(col)
            col_names.append('id')
        
        dfs = []
        df_tmp = pd.DataFrame(columns=col_names)

        for idx in range(len(df)):
            if idx % 100 == 0:
                print(f'Current sample number: {idx}.')
                print(f'Parsed {idx/6250*100}% of samples.')

            point = df.loc[idx][['lat', 'long']].tolist()
            df_tmp.loc[0, 'id'] = df.loc[idx]['id']
            for tag in tags:
                try:
                    geodf = ox.features_from_point(
                        center_point=point,
                        tags=tag,
                        dist=100
                        ).reset_index()
                    df_tmp.loc[0, list(tag.values())[0]] = len(geodf)
                except:
                    df_tmp.loc[0, list(tag.values())[0]] = 0

            df_tmp = df_tmp.loc[:,~df_tmp.columns.duplicated()]
            dfs.append(df_tmp)
            df_final = pd.concat(dfs)
            df_final = pd.merge(df, df_final, on=['id'])

        return df_final       
    
    else:
        col_names = []

        for idx in range(len(tags)):
            col = list(tags[idx].values())[0]
            col_names.append(col)
            col_names.append('lat')
            col_names.append('long')

        df_final = pd.DataFrame(columns=col_names)

        for idx in range(len(df)):
            point = df.loc[idx][['lat', 'long']].tolist()
            df_final.loc[0, 'lat'] = point[0]
            df_final.loc[0, 'long'] = point[1]
            for tag in tags:
                try:
                    geodf = ox.features_from_point(
                        center_point=point,
                        tags=tag,
                        dist=100
                        ).reset_index()
                    df_final.loc[0, list(tag.values())[0]] = len(geodf)
                except:
                    df_final.loc[0, list(tag.values())[0]] = 0
            
            df_final = df_final.loc[:,~df_final.columns.duplicated()]
            df_final = pd.merge(df, df_final, on=['lat', 'long'])

        return df_final


def get_population(df:pd.DataFrame) -> pd.DataFrame:
    '''
    функция принимает на вход Pandas DataFrame с
    координатами точек, географическими и другими различными признаками
    (строения, школы, дома и тп) и подсчитывает ориентировочное население
    в радиусе 100 метров.
    Возвращает Pandas DataFrame
    '''
    houses = ['semidetached_house', 'terrace', 'detached', 'house']
    apartments = ['apartments' , 'dormitory']

    all_populations = []
    
    for idx in range(len(df)):
        try:
            population_resp = ox.features_from_point(center_point=df.iloc[idx][['lat', 'long']].tolist(), 
                       tags={'building' : [
                           'apartments' , 'dormitory','house', 'semidetached_house', 
                           'detached', 'terrace', 
                           ]}, dist=200).reset_index().fillna(1)
            
            population = (sum([int(x)*10*3 for x in population_resp[population_resp['building']
                                                                    .isin(apartments)]['building:levels'].tolist()])) + \
                         (sum([int(x)*3 for x in population_resp[population_resp['building']
                                                                 .isin(houses)]['building:levels'].tolist()]))

            all_populations.append(population)
        except:
            all_populations.append(0)

    df['population'] = all_populations

    return df


def fit_model(train_pool:Pool, validation_pool:Pool, **kwargs) -> CatBoostClassifier:
    '''
    Функция для тренировки модели.
    На вход подаются два Pool'а для обучения и валидации.
    Возвращает обученную модель. В процессе обучения логируются
    параметры.
    '''
    model = CatBoostClassifier(
        iterations=30000,
        random_seed=DEFAULT_RANDOM_SEED,
        learning_rate=0.001,
        eval_metric='AUC',
        early_stopping_rounds=2000,
        use_best_model= True,
        task_type='CPU',
        **kwargs
    )

    return model.fit(
        train_pool,
        eval_set=validation_pool,
        verbose=100,
    )


def seedBasic(seed:int=42) -> None:
    '''
    Установка  random seed в окружение python
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)