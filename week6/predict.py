import numpy as np
import sigmoid

def predict(θ1, θ2, X):
    # add 1 
    m = X.shape[0]
    X = np.hstack([np.ones((m,1)),X]) # add bias
    a1 = X

    z2 = np.linalg.matmul(a1,θ1.T)
    a2 = sigmoid.sigmoid(z2)
    a2 = np.hstack([np.ones((m,1)), a2]) # add bias

    z3 = np.linalg.matmul(a2,θ2.T)
    a3 = sigmoid.sigmoid(z3)
    h_x = a3

    p = np.argmax(h_x, axis=1) + 1
    return p, h_x