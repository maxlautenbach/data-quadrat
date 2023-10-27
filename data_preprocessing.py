import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *


def preprocess_data(path):
    energy_consumption_data = pd.read_csv(path, index_col=0)
    train, test = train_test_split(energy_consumption_data, shuffle=False, test_size=0.2)
    columns = list(energy_consumption_data.columns)
    train = train.replace(-999, np.NaN)
    train = train.dropna()
    scaler = MinMaxScaler()

    for column in columns[:-3]:
        train[[column]] = scaler.fit_transform(train[[column]])

    return train, test
