import numpy as np
from scipy.spatial.distance import cdist

def weightedKNN(X_train,  y_train,   X_test,   sigma):
#            m x (n + 1)   m x 1   d x (n + 1)
    samples = np.size(X_test, 0)
    y_pred = np.zeros((samples, 1))

    d = cdist(X_test, X_train, 'euclidean')
    w = np.exp(-(d ** 2)/(sigma ** 2))

    # sorts the data into classes
    classes = np.unique(y_train)

    for i in range(samples):
        w_vote = []
        for c in classes:
            w_i = w[i,:]
            w_sum = sum((y_train.flatten() == c) * w_i) # gets the weighted sum per class
            w_vote.append(w_sum)
        
        y_pred[i] = classes[np.argmax(w_vote)]

    return y_pred
