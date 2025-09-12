import numpy as np

def computeCost(X, y, θ):
    
    # add X0 to the X matrix (maybe add later)
    # X = np.vstack((ones_vec, X))

    θ = np.transpose(θ)
    Xθ = np.matmul(X,θ) # find h(x)
    err = y - Xθ # find error
    #  m x 1  m x 1
    J_matrix =  np.matmul(np.transpose(err),err) # find variance
    #        1 x m           m x 1

    J = J_matrix.item()
    return J