import numpy as np
import matplotlib.pyplot as plot
import scipy.io
import sklearn.neighbors as skl
from sklearn.metrics import accuracy_score

# functions
import Reg_normalEqn # computes closed form solution to linear regression
import logReg_multi

# for training/testing error (mean squared error)
def mse(X, y, θ):
    y_pre = np.matmul(X,θ)
    m = np.size(y,axis=0)
    return (1/2)*(1/m)*np.sum((y_pre - y)**2)

def accuracy(test, pred):
    correct = np.sum(pred == test.ravel())
    total = len(test)
    return correct/total
    
# --------------------------------- Question 1 ------------------------------
print('1)')
# print('a.')
# loading in data files
data1 = scipy.io.loadmat('hw4_data1.mat')

# print('b.')
X = np.array(data1['X_data'])

# add bias feature
m = np.size(X,0) # returns m - # of features
ones = np.ones((m,1))
X = np.hstack((ones, X)) # merge X0 and rest of X

m = np.size(X,0)
n = np.size(X,1)
print('The size of the feature matrix is', m, 'rows by', n, 'columns.')

y = np.array(data1['y'])



# print('c.')
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

        # get training/testing error
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
#plot.savefig('/workspaces/python/week4/ps4-1-a.png')
plot.close()

# --------------------------------- Question 2 ------------------------------
print('2.')

data2 = scipy.io.loadmat('hw4_data2.mat')


X_folds = [data2['X1'],data2['X2'],data2['X3'],data2['X4'],data2['X5']]
y_folds = [data2['y1'],data2['y2'],data2['y3'],data2['y4'],data2['y5']]

rng_arr = np.arange(1,17,2)
# print(rng_arr)

avg_accuracies = [] # holds avg accuracies across all k values (1,3,...,15)
for k in rng_arr:
    fold_accuracies = [] # holds accuracies across five folds for one k value
    for i in range(5):
        X_test = X_folds[i] # m x n
        y_test = y_folds[i] # m x 1
        # print(np.size(X_test,0),np.size(X_test,1))

        X_train2 = np.vstack([X_folds[j] for j in range(5) if j != i])
        y_train2 = np.vstack([y_folds[j] for j in range(5) if j != i])
        X_test2 = np.array(data2['X4'])
        y_test2 = np.array(data2['y4'])

        # train KNN model
        knn = skl.KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train2, y_train2.ravel())

        # accuracy
        y_pred = knn.predict(X_test)
        acc = accuracy(y_test, y_pred)
        fold_accuracies = np.append(fold_accuracies, [acc])

    avg = np.mean(fold_accuracies)
    avg_accuracies = np.append(avg_accuracies,[avg])

# print(avg_accuracies)
plot.plot(rng_arr,avg_accuracies,'o-',color='mediumaquamarine')
plot.xlabel('K')
plot.ylabel('average accuracy')
plot.xticks(rng_arr)
plot.savefig('ps4-2-a.png')

print('I suggest using k = 9 for this particular problem as it has the highest accuracy.')
print('This value is not necessarily robust for any other problem as the best k value is dependent on the variance of the data.')

# --------------------------------- Question 3 ------------------------------

print('3.')
data3 = scipy.io.loadmat('hw4_data3.mat')
X_train = data3['X_train']
y_train = data3['y_train']
X_test = data3['X_test']
y_test = data3['y_test']

y_pred = logReg_multi.logReg_multi(X_train, y_train, X_test)
# print(y_pred) for debugging

train_pred = y_pred['train_pred']
test_pred = y_pred['test_pred']
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print('the training accuracy is:',train_acc)
print('the testing accuracy is:',test_acc)
print('the training accuracy is higher than the testing accuracy but the testing accuracy is still quite high.')
