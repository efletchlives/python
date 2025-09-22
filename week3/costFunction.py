import numpy as np
import sigmoid

def costFunction(θ, X, y):
    # number of rows
    m = np.size(X,axis=0)

    # compute h(x) = g(θTx)
    θTx = np.matmul(θ.T,X)
    h_x = sigmoid.sigmoid(θTx)

    # compute cost
    J = np.sum((1/m) * (np.matmul(-y,np.log10(h_x)) - np.matmul((1-y),np.log10(1-h_x))))

    return J