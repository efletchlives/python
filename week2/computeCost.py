import numpy as np

def computeCost(X, y, θ): # works properly
    m = X.shape[0] # number of samples

    # θT = np.transpose(θ)
# (n + 1) x 1
    Xθ = np.matmul(X,θ) # find h(x)
#  m x 1
    err = y - Xθ # find error
#  m x 1
    J_mtx = 1/(2*m)*np.transpose(err) @ err # find variance
#                               1 x m      m x 1

    cost = J_mtx.item() # change cost to single value
    return cost