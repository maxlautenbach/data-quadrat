import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *


def preprocess_data(path):
    energy_consumption_data = pd.read_csv(path, index_col=0)

    energy_consumption_data = energy_consumption_data.replace(-999, np.NaN).dropna()

    train, test = train_test_split(energy_consumption_data, shuffle=False, test_size=0.2)
    columns = list(energy_consumption_data.columns)
    scaler = MinMaxScaler()

    for column in columns[:-3]:
        scaler.fit(train[[column]])
        train[[column]] = scaler.transform(train[[column]])
        test[[column]] = scaler.transform(test[[column]])

    return train, test
