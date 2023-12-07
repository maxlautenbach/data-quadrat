import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *


def preprocess_data(path):
    # Load collected data
    residual_load_data = pd.read_csv(path, index_col=0)

    # Split into train and test split
    train, test = train_test_split(residual_load_data, shuffle=False, test_size=0.2)

    # Delete missing values
    columns = list(residual_load_data.columns)
    train = train.replace(-999, np.NaN)
    train = train.dropna()

    # Normalize data
    scaler = MinMaxScaler()
    for column in columns[:-3]:
        train[[column]] = scaler.fit_transform(train[[column]])
        test[[column]] = scaler.fit_transform(test[[column]])

    return train, test
