import numpy as np
import computeCost

def gradientDescent(X_train, y_train, alpha, iters):
    m = X_train.shape[0] # get number of rows

    # generate random initialization for θ
    θ = np.random.randn(m,1) # ask dallal
    J = np.array([])

    for i in range(iters):
        # update cost at beginning
        J[i,0] = computeCost.computeCost(X_train, y_train, θ)
#                                      m x (n + 1) m x 1 (n + 1) x 1
        Xθ = X_train @ θ
#     m x 1   m x n  n x 1

        # compute gradient
        gradient = (1/m) * np.matmul(np.transpose(X_train), (Xθ - y_train)) 
#     

        # find + update new θ values in a vector 
        θ = θ - alpha * gradient
        
    

    return θ, J