import computeCost
import gradientDescent
import normalEqn
# θ = theta

# import csv data


# compute cost function
J = computeCost(X, y, θ)

# compute gradient descent solution to linear regression
[θ, J] = gradientDescent(X_train, y_train, alpha, iters)

# compute closed-form solution to linear regression using normal equation
[θ] = normalEqn(X_train, y_train) 


