# modules 
import numpy as np

# functions
import computeCost
import gradientDescent
import normalEqn
# θ = theta

# toy data set
X = np.array([[0,1],[1,1.5],[2,4],[3,2]]) # 4x3
y = np.array([[1.5],[4],[8.5],[8.5]]) # 4x1

# theta (i)
θ = np.array([[0.5,2,1]]) # 3x1
J = computeCost.computeCost(X, y, θ)
print('J(i):',J)

# theta (ii)
θ = np.array([[3,-1.5,-4]]) # 3x1
J = computeCost.computeCost(X, y, θ)
print('J(ii):',J)

# theta (iii)
θ = np.array([[0.5,1,2]]) # 3x1
J = computeCost.computeCost(X, y, θ)
print('J(iii):',J)




# compute gradient descent solution to linear regression
# [θ, J] = gradientDescent.gradientDescent(X_train, y_train, alpha, iters)

# compute closed-form solution to linear regression using normal equation
# [θ] = normalEqn(X_train, y_train) 


# 4.
# import csv data for linear regression with one variable
# X = np.loadtxt('hw2_data1.csv', usecols=0)
# y = np.loadtxt('hw2_data1.csv', usecols=1)

# 5.
# import csv data for linear regression with multiple variables 
# X = np.loadtxt('hw2_data3.csv',usecols=(0,1))
# y = np.loadtxt('hw2_data3.csv',usecols=2)

# compute cost function
# J = computeCost(X, y, θ)

# compute gradient descent solution to linear regression
# [θ, J] = gradientDescent(X_train, y_train, alpha, iters)

# compute closed-form solution to linear regression using normal equation
# [θ] = normalEqn(X_train, y_train)
