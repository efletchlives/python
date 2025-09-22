import numpy as np
import matplotlib.pyplot as plot
import scipy as scipy

# functions
import sigmoid
import costFunction
import gradFunction

# load data
print('1.')
print('a.')
data = np.loadtxt('/workspaces/python/week3/hw3_data1.txt',delimiter=',')
cols = np.size(data, axis=1)
X = data[:,[0,cols-2]]
y = data[:,[cols-1]]

# plot training data
print('b.')
not_admit = np.where(y == 0)[0]
admit = np.where(y == 1)[0]

plot.figure()
plot.scatter(X[admit,0], X[admit,1],marker='+',color='black',label='admitted')
plot.scatter(X[not_admit,0], X[not_admit,1],marker='o',color='yellow',label='not admitted')
plot.xlabel('exam 1 scores')
plot.ylabel('exam 2 scores')
plot.legend()
plot.savefig('/workspaces/python/week3/ps3-1-b.png')
plot.close()

# add col of ones for bias feature X0
m = np.size(X,0) # returns m - # of features
ones = np.ones((m,1))
X = np.hstack((ones, X)) # merge X0 and rest of X


# randomly divide data into training and testing data
print('c.')
indices = np.random.permutation(m) # randomize m indices
size = round(m*0.9)
idx_train = indices[:size]
idx_test = indices[size:]

# create X + y train & test data
X_train = X[idx_train]
y_train = y[idx_train]
X_test = X[idx_test]
y_test = y[idx_test]

# computes sigmoid function
print('d.')
z = np.arange(-15,15,0.01)
gz = sigmoid.sigmoid(z)

plot.figure()
plot.plot(z,gz,'o',color='blue')
plot.title('sigmoid function plot')
plot.xlabel('z')
plot.ylabel('gz')
plot.savefig('/workspaces/python/week3/ps3-1-c.png')
plot.close()

# compute cost function and gradient descent
print('e.')
X_toy = np.array([[1,1,0],[1,1,3],[1,3,1],[1,3,4]]) # m x n
y_toy = np.array([[0],[1],[0],[1]]) # m x 1
θ = np.array([1,0.9,1.2]) # 1 x n

J = costFunction.costFunction(θ, X_toy, y_toy)
grad = gradFunction.gradFunction(θ, X_toy, y_toy)
print(J)

# optimize cost function with parameters θ
print('f')
θ = np.array([0,0,0])
#scipy.optimize.fmin_bfgs(lambda t: costFunction.costFunction(t, X_train, y_train),θ)



print('g.')
# get the x and y intercepts for the decision boundary line


plot.figure()
# plot decision boundary
#plot.plot()
plot.scatter(X[admit,0], X[admit,1],marker='+',color='black',label='admitted')
plot.scatter(X[not_admit,0], X[not_admit,1],marker='o',color='yellow',label='not admitted')
plot.xlabel('exam 1 scores')
plot.ylabel('exam 2 scores')
plot.legend()
plot.savefig('/workspaces/python/week3/ps3-1-b.png')
plot.close()
#grad = gradFunction.gradFunction(θ, X_toy, y_toy)


# J = costFunction.costFunction(θ, X_train, y_train)
# grad = gradFunction.gradFunction(θ, X_train, y_train)