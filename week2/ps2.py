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
print('There is a significant difference between the theta estimates in 2 and 3 because the alpha value is far too small. It isn’t able to make it to super close or at 0 cost. You should increase the value of alpha such that the two approaches give almost the same result.')

# 4.
print('\n4.')
# a) import csv data for linear regression with one variable
data = np.loadtxt('/workspaces/python/week2/hw2_data1.csv', delimiter=',')
X_og = data[:,[0]]
y_og = data[:,[1]]
y = y_og

# b) plot the data
plot.figure()
plot.plot(X_og,y_og,'x',color='orange')
plot.xlabel('horsepower of each car (in 100s hp)')
plot.ylabel('prices of automobiles (in $1,000s)')
plot.savefig('/workspaces/python/week2/ps2-4-b.png')
plot.close()

# c) include X0 in X matrix and output size of X & y

# add X0 to the X matrix
m = np.size(X_og,0) # returns m - # of features
ones = np.ones((m,1))
X = np.hstack((ones, X_og)) # merge X0 and rest of X
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

# e) compute gradient descent and plot the vector cost
alpha = 0.3
iters = 500
[θ, J] = gradientDescent.gradientDescent(X_train, y_train, alpha, iters)

iters_plot = np.arange(1,501,1)
plot.figure()
plot.plot(iters_plot,J,'g-')
plot.xlabel('iteration #')
plot.ylabel('cost')
plot.savefig('/workspaces/python/week2/ps2-4-e.png')
plot.close()

# f) plot the line for the learned model (y = θ0 + θ1*x)
plot.figure()
plot.plot(X_og,y_og,'x',color='orange',label='training data')

# get min and max points on line
x_pts = [np.min(X_train[:,:]), np.max(X_train[:,:])]
y_min = θ[0] + θ[1]*x_pts[0]
y_max = θ[0] + θ[1]*x_pts[1]
y_pts = [y_min,y_max]

plot.plot(x_pts,y_pts,'c-',label='learned model')
plot.title('learned model with scattered data')
plot.xlabel('horsepower of each car (in 100s hp)')
plot.ylabel('prices of automobiles (in $1,000s)')
plot.legend()
plot.savefig('/workspaces/python/week2/ps2-4-f.png')
plot.close()

# g) compute cost using obtained model parameters from e
J = computeCost.computeCost(X_test,y_test,θ)
print('\ng:')
print('my prediction error:',J)

# h) use normalEqn function and training dataset
θ = normalEqn.normalEqn(X_train,y_train)
J = computeCost.computeCost(X_test,y_test,θ)
print('\nh:')
print('my prediction error:',J)
# using the normalEqn function in h is a very good fit for the obtained model parameters in g

# i) study effect of learning rate
iters = 300
iters_plot = np.arange(1,301,1)
alpha = np.array([0.001,0.003,0.03,3])

# first iteration: alpha = 0.001
[θ, J] = gradientDescent.gradientDescent(X_train, y_train, 0.001, iters)
plot.figure()
plot.plot(iters_plot,J,'g-')
plot.title('cost function for alpha = 0.001')
plot.xlabel('iteration #')
plot.ylabel('cost')
plot.annotate('too small of an alpha value,\ndoesn\'t converge fast enough',xy=(150,80),xytext=(150,80))
plot.savefig('/workspaces/python/week2/ps2-4-i-1.png')
plot.close()

# second iteration: alpha = 0.003
[θ, J] = gradientDescent.gradientDescent(X_train, y_train, 0.003, iters)
plot.figure()
plot.plot(iters_plot,J,'g-')
plot.title('cost function for alpha = 0.003')
plot.xlabel('iteration #')
plot.ylabel('cost')
plot.annotate('closer to a good alpha value,\nstill doesn\'t converge fast enough',xy=(100,40),xytext=(100,80))
plot.savefig('/workspaces/python/week2/ps2-4-i-2.png')
plot.close()

# third iteration: alpha = 0.03
[θ, J] = gradientDescent.gradientDescent(X_train, y_train, 0.03, iters)
plot.figure()
plot.plot(iters_plot,J,'g-')
plot.title('cost function for alpha = 0.03')
plot.xlabel('iteration #')
plot.ylabel('cost')
plot.annotate('very good alpha value,\nconverges fast enough',xy=(30,21),xytext=(30,60))
plot.savefig('/workspaces/python/week2/ps2-4-i-3.png')
plot.close()

# fourth iteration: alpha = 3 (causes runtime errors since it diverges)
# [θ, J] = gradientDescent.gradientDescent(X_train, y_train, 3, iters)
# plot.figure()
# plot.plot(iters_plot,J,'g-')
# plot.title('cost function for alpha = 3')
# plot.xlabel('iteration #')
# plot.ylabel('cost')
# plot.annotate('diverges and causes runtime warning\nbecause it overshoots the minimum',xy=(0,1e307),xytext=(75,1e307))
# plot.savefig('/workspaces/python/week2/ps2-4-i-4.png')
# plot.close()


# 5.
# a) import + load csv data, then standardize
print('\n5.')
print('\na:')
data = np.loadtxt('/workspaces/python/week2/hw2_data3.csv',delimiter=',')
X = data[:,[0,1]]
y = data[:,[2]]

# standardize X data
X_mean = np.mean(X, axis=0)
print('mean of X col 0:',X_mean[0])
print('mean of X col 1:',X_mean[1])

X_std = np.std(X, axis=0)
print('standard deviation of X col 0:',X_std[0])
print('standard deviation of X col 1:',X_std[1])
X_stand = (X - X_mean)/X_std
X = X_stand

# add X0 to the X matrix
m = np.size(X,0) # returns m - # of features
ones = np.ones((m,1))
X = np.hstack((ones, X)) # merge X0 and rest of X
#  m x (n + 1)

# print size of X & y
m, n = X.shape
print('\nsize of X:',m,'rows &',n,'cols')

m, n = y.shape
print('size of y:',m,'rows &',n,'cols')

# b) compute gradient descent solution + plot vector cost that shows cost function for each iteration
print('\nb:')
alpha = 0.01
iters = 750
[θ, J] = gradientDescent.gradientDescent(X, y, alpha, iters)

# plot cost vs iteration num
iters_plot = np.arange(1,751,1)
plot.figure()
plot.plot(iters_plot,J,'g-')
plot.title('cost function for alpha = 0.01')
plot.xlabel('iteration #')
plot.ylabel('cost')
plot.savefig('/workspaces/python/week2/ps2-5-b.png')
plot.close()

print('θ:\n',θ)

# c) predict the CO2 emission
# standardize new sample
X_test = np.array([2100,1200])
X_test = (X_test - X_mean)/X_std
# [X1, X2]

# using calculate θ values, predict CO2 emission
CO2 = θ[0] + θ[1]*X_test[0] + θ[2]*X_test[1]
CO2 = CO2.item()
print('\nmy CO2 emissions prediction:',CO2)