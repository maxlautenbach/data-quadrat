from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import itertools
import math
import numpy as np
import data_preprocessing

# Load the preprocessed data
train_pre, test_pre = data_preprocessing.preprocess_data("Residual Load Dataset.csv")

# Split the data into features and target
X_train = train_pre.iloc[:, :-1]
#print(X_train)
y_train = train_pre.iloc[:, -1]
X_test = test_pre.iloc[:, :-1]
y_test = test_pre.iloc[:, -1]

cols = X_train.columns
combinations = []
for i in range(6,7):
    for j in itertools.combinations(cols, i):
        combinations.append(j)
print(len(combinations))

lowest = {"num":100000,"i" : 0,"comb":"_"}
#print(combinations)
for combin in combinations:
    combi = np.asarray(combin)
    
    com_X_train = X_train[combi]
    com_X_test = X_test[combi]

    for i in range(1,2):
        #print(i)
        # Create the KNN regressor
        knn = KNeighborsRegressor(n_neighbors=i*50)

        # Train the model
        knn.fit(com_X_train, y_train)

        # Predict the Residual Load for the test data
        y_pred = knn.predict(com_X_test)

        # Calculate the root mean squared error
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))

        if rmse < lowest['num'] :
            lowest['num'] = rmse
            lowest['i'] = i
            lowest['comb'] = combi
    print(lowest) 
print(lowest) 
out = open("out.txt","w")
out.write(str(lowest))
out.close()