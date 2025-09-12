import numpy as np
import computeCost

def gradientDescent(X_train, y_train, alpha, iters):
    # generate random initialization for θ
    # θ =  ask dallal
    tempθ = np.array([])

    for i in range(iters):
        # update cost at beginning
        J = computeCost.computeCost(X_train, y_train, θ)

        # find new θ values
        for j in range(X_train.shape[1]):
            tempθ[j] = θ[j] - alpha * np.gradient(J)

        # update θ values
        for k in range(X_train.shape[1]):
            θ[k] = tempθ[k]