import pandas as pd
import utils_config
config = utils_config.load_config("./config.json")
import os

def base(str):
    list_arq = os.listdir(path='data/')
    for names in list_arq:
        if str == 'total':
            vars = [
        'fixed acidity',
        'volatile acidity',
        'citric acid',
        'residual sugar',
        'chlorides',
        'free sulfur dioxide',
        'total sulfur dioxide',
        'density',
        'pH',
        'sulphates',
        'alcohol',
            ]
            wines = config.total
            wines = pd.read_csv(wines)
            wines["category"] = (wines.quality > 5).astype(float)
            X = wines[vars]
            y = wines['category']
            return X, y
        elif str == 'white':
            vars = [
        'citric acid',
        'residual sugar',
        'total sulfur dioxide',
        'sulphates',
        'alcohol',
            ]
            white = config.white
            white = pd.read_csv(white, sep=';')
            white["category"] = (white.quality > 5).astype(float)
            X = white[vars]
            y = white['category']
            return X, y
        elif str == 'red':
            vars = [
        'volatile acidity',
        'total sulfur dioxide',
        'pH',
        'sulphates',
        'alcohol',
            ]
            red = config.red
            red = pd.read_csv(config.red, sep=';')
            red["category"] = (red.quality > 5).astype(float)
            X = red[vars]
            y = red['category']
            return X, y