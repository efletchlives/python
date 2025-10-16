import numpy as np
import matplotlib as plot
import scipy.io
from sklearn.metrics import accuracy_score

import weightedKNN

# ------------------------ Question 1 ----------------------------

data3 = scipy.io.loadmat('/workspaces/python/week5/input/hw4_data3.mat')
X_train = data3['X_train']
y_train = data3['y_train']
X_test = data3['X_test']
y_test = data3['y_test']


sigma = np.array([0.01,0.07,0.15,2,4])
y_predict = weightedKNN.weightedKNN(X_train, y_train, X_test, sigma)

accuracy = accuracy_score(y_test, y_predict)
print('weighted KNN accuracy using Euclidean distance:',accuracy)


