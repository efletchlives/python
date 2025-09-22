import numpy as np
import sigmoid

def costFunction(θ,     X,     y):
    #          1 x n  m x n  m x 1
    # number of rows
    m = np.size(X,axis=0)

    # compute h(x) = g(θTx)
    θTx = np.matmul(X,θ.T) # m x n * n x 1 => m x 1
    h_x = sigmoid.sigmoid(θTx) # m x 1

    # compute cost
    J_mtx = (1/m) * (np.matmul((-1)*y.T,np.log(h_x)) - np.matmul((1-y.T),np.log(1-h_x))) # m x n
    #                           1 x m           m x 1               1 x m           m x 1
    J = J_mtx
    return J