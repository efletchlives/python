import numpy as np

def computeCost(X, y, θ):
    
    # add X0 to the X matrix
    m = np.size(X,0) # returns m - # of features
    ones = np.ones((m,1))
    X = np.hstack((ones, X)) # merge X0 and rest of X
#  m x (n + 1)

    θ = np.transpose(θ)
# (n + 1) x 1
    Xθ = np.matmul(X,θ) # find h(x)
#  m x 1
    err = y - Xθ # find error
#  m x 1
    J_mtx =  np.matmul(np.transpose(err),err) # find variance
#                               1 x m      m x 1

    J = J_mtx.item() # change cost to single value
    return J