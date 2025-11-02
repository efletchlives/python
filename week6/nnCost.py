import numpy as np

import predict

def nnCost(θ1, θ2, X, y, K, λ):
    m = X.shape[0]
    _, h_x = predict.predict(θ1, θ2, X) # compute h(x) using forward propagation

    # turn y_labels to y vector
    y = y.flatten()
    y_vec = np.zeros((m, K))
    for i in range(m):
        y_vec[i,y[i]-1] = 1 # puts 1 at the corresponding spot in the vector
    y = y_vec

    cost = -(1/m) * np.sum(y * np.log(h_x) + (1 - y) * np.log(1 - h_x)) # cost w/o regularization
    reg = λ/(2*m) * (np.sum(θ1[:,1:] ** 2) + np.sum(θ2[:,1:] ** 2)) # regularization

    J = cost + reg
    return J