import numpy as np
import computeCost

def gradientDescent(X_train, y_train, alpha, iters):
    #              m x (n+1)  m x 1
    m,n = X_train.shape # get number of rows

    # generate random initialization for θ
    θ = np.random.randn(n,1)
# n x 1

    J = np.zeros((iters,1))
# iters x 1
    for i in range(iters):
        # update cost at beginning
        J[i,0] = computeCost.computeCost(X_train, y_train, θ)
#       1 x 1                         m x (n + 1) m x 1   n x 1
        Xθ = np.matmul(X_train, θ)
#     m x 1             m x n  n x 1

        # compute gradient
        gradient = (1/m) * np.matmul(np.transpose(X_train), (Xθ - y_train)) 

        # find + update new θ values in a vector (this is wrong)
        θ = θ - alpha * gradient

    return θ, J