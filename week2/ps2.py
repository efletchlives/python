# modules 
import numpy as np
import matplotlib.pyplot as plot

# functions
import computeCost
import gradientDescent
import normalEqn
# θ = theta

# toy data set
X = np.array([[0,1],[1,1.5],[2,4],[3,2]]) # 4x2
y = np.array([[1.5],[4],[8.5],[8.5]]) # 4x1

# add X0 to the X matrix
m = np.size(X,0) # returns m - # of features
ones = np.ones((m,1))
X = np.hstack((ones, X)) # merge X0 and rest of X
#  m x (n + 1)

# 1) test toy dataset cost
print('1.')
# theta (i)
θ = np.array([[0.5],[2],[1]]) # 3x1
J = computeCost.computeCost(X, y, θ)
print('J(i):',J)

# theta (ii)
θ = np.array([[3],[-1.5],[-4]]) # 3x1
J = computeCost.computeCost(X, y, θ)
print('J(ii):',J)

# theta (iii)
θ = np.array([[0.5],[1],[2]]) # 3x1
J = computeCost.computeCost(X, y, θ)
print('J(iii):',J)


# 2) test toy dataset
print('\n2.')
X_train = X # m x (n+1)
y_train = y # m x 1
alpha = 0.001
iters = 15
[θ, J] = gradientDescent.gradientDescent(X_train, y_train, alpha, iters)

print('θ:\n',θ)
print('J:\n',J)

# 3) test toy dataset normal equation
θ = normalEqn.normalEqn(X_train, y_train)
print('\n3.')
print('θ:\n',θ)

# 4.
print('\n4.')
# a) import csv data for linear regression with one variable
data = np.loadtxt('/workspaces/python/week2/hw2_data1.csv', delimiter=',')
X = data[:,[0]]
y = data[:,[1]]

# b) plot the data
plot.plot(X,y,'rx')
plot.xlabel('horsepower of each car (in 100s hp)')
plot.ylabel('prices of automobiles (in $1,000s)')
plot.savefig('ps2-4-b.png')
plot.close()

# c) include X0 in X matrix and output size of X & y

# add X0 to the X matrix
m = np.size(X,0) # returns m - # of features
ones = np.ones((m,1))
X = np.hstack((ones, X)) # merge X0 and rest of X
#  m x (n + 1)

# print size of X & y
m, n = X.shape
print('size of X:',m,'rows &',n,'cols')

m, n = y.shape
print('size of y:',m,'rows &',n,'cols')

# d) randomly divide the data into a training and test set
indices = np.random.permutation(m) # randomize m indices
size = round(m*0.9)
idx_train = indices[:size]
idx_test = indices[size:]

# create X + y train & test data
X_train = X[idx_train]
y_train = y[idx_train]
X_test = X[idx_test]
y_test = y[idx_test]
alpha = 0.3
iters = 500

[θ, J] = gradientDescent.gradientDescent(X_train, y_train, alpha, iters)

iters_plot = np.arange(1,501,1)
plot.plot(iters_plot,J)
plot.xlabel('iteration #')
plot.ylabel('cost')
plot.savefig('ps2-4-e.png')

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
