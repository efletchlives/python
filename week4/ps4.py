import numpy as np
import matplotlib.pyplot as plot
import scipy.io

# functions
import Reg_normalEqn # computes closed form solution to linear regression

# for training/testing error (mean squared error)
def mse(X, y, θ):
    y_pre = np.matmul(X,θ)
    m = np.size(y,axis=0)
    return (1/2)*(1/m)*np.sum((y_pre - y)**2)
    

print('1)')
print('a.')
# loading in data files
data1 = scipy.io.loadmat('/workspaces/python/week4/hw4_data1.mat')

print('b.')
X = np.array(data1['X_data'])
m = np.size(X,0)
n = np.size(X,1)
print('The size of the feature matrix is', m, 'rows by', n, 'columns.')

y = np.array(data1['y'])

# add bias feature
m = np.size(X,0) # returns m - # of features
ones = np.ones((m,1))
X = np.hstack((ones, X)) # merge X0 and rest of X

print('c.')
λ = np.array([0, 0.001, 0.003, 0.005, 0.007, 0.009, 0.012, 0.017])
train_mtx = np.empty((20,8))
test_mtx = np.empty((20,8))

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
        #print('parameters for iteration', i, ',', j, 'and λ =',λ[j],':',θ)
        train_error = mse(X_train, y_train, θ)
        test_error = mse(X_test, y_test, θ)

        # 20 x 8 matrices of training/testing errors
        train_mtx[i,j] = train_error
        test_mtx[i,j] = test_error

    # print('training error matrix:',train_mtx)
    # print('testing error matrix:',test_mtx)

avg_train_mtx = np.mean(train_mtx, axis=0)
avg_test_mtx = np.mean(test_mtx, axis=0)
# print(avg_train_mtx)
# print(avg_test_mtx)

plot.figure()
plot.plot(λ, avg_train_mtx, 'r-x', label='training error')
plot.plot(λ, avg_test_mtx, 'b-o', label='testing error')

plot.xlabel('λ')
plot.ylabel('average error')
plot.legend()
plot.savefig('/workspaces/python/week4/ps4-1-a.png')
plot.close()


print('2.')

data2 = scipy.io.loadmat('/workspaces/python/week4/hw4_data2.mat')









# data3 = scipy.io.loadmat('/workspaces/python/week4/hw4_data3.mat')
