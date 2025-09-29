import numpy as np
import scipy.io

# functions
import Reg_normalEqn # computes closed form solution to linear regression

print('1)')
print('a.')
# loading in data files
data1 = scipy.io.loadmat('/workspaces/python/week4/hw4_data1.mat')

print('b.')
X = data1['X_data']
m,n = np.size()
print('The size of the feature matrix is', m, 'rows by', n, 'columns.')

y = data1['y']

# add bias feature
m = np.size(X,0) # returns m - # of features
ones = np.ones((m,1))
X = np.hstack((ones, X)) # merge X0 and rest of X

print('c.')
λ = np.array([0, 0.001, 0.003, 0.005, 0.007, 0.009, 0.012, 0.017])

for i in range(20):
    indices = np.random.permutation(m) # randomize m indices
    size = round(m*0.85)
    idx_train = indices[:size]
    idx_test = indices[size:]

    X_train = X[idx_train]
    y_train = y[idx_train]
    X_test = X[idx_test]
    y_test = y[idx_test]

    for j in range(8):
        θ = Reg_normalEqn.Reg_normalEqn(X_train, y_train, λ[j])
        

   












# data2 = scipy.io.loadmat('/workspaces/python/week4/hw4_data2.mat')
# data3 = scipy.io.loadmat('/workspaces/python/week4/hw4_data3.mat')


