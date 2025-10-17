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
accuracy = []
for i in range(5):
    y_predict = weightedKNN.weightedKNN(X_train, y_train, X_test, sigma[i])
    accuracy.append(accuracy_score(y_test, y_predict))


print('weighted KNN accuracy using Euclidean distance:',accuracy)


# ------------------------ Question 2 ----------------------------

# randomizing and organize files in directories (input/all, input/train, input/test)
import os
import random

# clear files in the input/train and input/test
all_path = '/workspaces/python/week5/input/all'
train_path = '/workspaces/python/week5/input/train'
test_path = '/workspaces/python/week5/input/test'
train_imgs = os.listdir(train_path)
test_imgs = os.listdir(test_path)

for filename in train_imgs:
    file_path = os.path.join(train_path, filename)
    os.remove(file_path)

for filename in test_imgs:
    file_path = os.path.join(test_path, filename)
    os.remove(file_path)

for i in range(40):
    train_idx = np.sort(random.sample(range(1,11), 8))
    test_idx = np.setdiff1d(np.arange(1,11), train_idx)
    print(train_idx)
    print(test_idx)

    for j in train_idx:
        os.path.join(train_path, f"{i}_{j}.pgm")

        os.system(f"cp {all_path}/s{i}/{j}.pgm {train_path}/{i}_{j}.pgm")
    
    for j in test_idx:
        os.path.join(test_path, f"{i}_{j}.pgm")

        os.system(f"cp {all_path}/s{i}/{j}.pgm {test_path}/{i}_{j}.pgm")


# ------------------------ Question 2.1 ----------------------------

# a.
    
        
