import numpy as np
import sigmoid

def gradFunction(θ,     X,     y):
#              1 x n  m x n  m x 1
    # number of rows
    m = np.size(X,axis=0)

    # compute h(x) = g(θTx)
    θTx = np.matmul(X,θ.T) # m x n * n x 1 => m x 1
    h_x = sigmoid.sigmoid(θTx) # m x 1

    h_x = h_x.T
    y = y.T

    grad = (1/m) * np.matmul((h_x - y),X)
#                              1 x m   m x n
    return grad.flatten() # 1 x n