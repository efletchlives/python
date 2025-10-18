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
import matplotlib.pyplot as plot

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

for i in range(1,41):
    train_idx = np.sort(random.sample(range(1,11), 8))
    test_idx = np.setdiff1d(np.arange(1,11), train_idx)
    # print(train_idx)
    # print(test_idx)

    for j in train_idx:
        os.path.join(train_path, f"{i}_{j}.pgm")

        os.system(f"cp {all_path}/s{i}/{j}.pgm {train_path}/{i}_{j}.pgm")

    for j in test_idx:
        os.path.join(test_path, f"{i}_{j}.pgm")

        os.system(f"cp {all_path}/s{i}/{j}.pgm {test_path}/{i}_{j}.pgm")

# do 3 face image part    IMPORTANT!!!!!!!!!!!!


# ------------------------ Question 2.1 ----------------------------

# a.
train_imgs = os.listdir(train_path)
test_imgs = os.listdir(test_path)

T = np.empty((10304,1))
for pmg_file in train_imgs:
    img = plot.imread(f"{train_path}/{pmg_file}")
    col_vec = img.reshape(-1, 1, order='F') # 'F' flattens img column-wise instead of default row-wise
    T = np.concatenate((T,col_vec), axis=1)


T = T[:,1:]
# n,m = T.shape
# print(f"{n} by {m}")

plot.imsave("/workspaces/python/week5/output/ps5-1-a.png", T, cmap='gray')

# b.
T_avg = np.mean(T, axis=1) # computes average across rows (returns 10304 x 1)
T_avg = T_avg.reshape(-1,1)
T_avg_b = np.resize(T_avg,(92,112))
T_avg_b = np.rot90(T_avg_b, k=-1)



# n,m = T_avg.shape
# print(f"{n} by {m}")

plot.imsave("/workspaces/python/week5/output/ps5-2-1-b.png", T_avg_b, cmap='gray')

# c.
# define centered data matrix
A = T - T_avg

# define data covariance
C = np.linalg.matmul(A,A.T)
# m,n = C.shape
# print(f"{m} by {n}")

plot.imsave("/workspaces/python/week5/output/ps5-2-1-c.png", C, cmap='gray')

# d.
# compute eigenvalues of A^T*A
eig_vals, eig_vecs = np.linalg.eig(A.T @ A)
N = eig_vals.shape
N = N[0]

v = []
sum = np.sum(eig_vals)
for k in range(N):
    v.insert(k, eig_vals[k]/sum)

k = np.arange(1,N + 1)

plot.plot(k,v,'-', color='aquamarine')
plot.title('k vs v(k)')
plot.xlabel('k')
plot.ylabel('v(k)')
plot.savefig("/workspaces/python/week5/output/ps5-2-1-d.png")

# K - min # of eigenvectors needed to capture 95% of variance in training data
min_v = 0.0
K = 0
for k in range(N):
    min_v += v[k]
    if min_v >= 0.95:
        K = k + 1
        break

print(f"the minimum # of eigenvectors K that capture 95% of variance is {K}")

# e.
