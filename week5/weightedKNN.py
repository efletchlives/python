import numpy as np
from sklearn.neighbors import KNeighborsClassifier as knn

def weightedKNN(X_train, y_train, X_test, sigma):
    n = X_train.shape[0]

    knn_weighted = knn(n_neighbors=n, weights = 'distance', metric = 'euclidean')
    knn_weighted.fit(X_train,y_train)

    y_predict = knn_weighted.predict(X_test)
    return y_predict
