import numpy as np
import matplotlib as plot
import scipy.io
from sklearn.metrics import accuracy_score
import os
import random
import matplotlib.pyplot as plot
import sklearn.neighbors as skl
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
import time

import weightedKNN

# ------------------------ Question 1 ----------------------------
print('1.')
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


print('weighted KNN accuracy using Euclidean distance:',accuracy,'\n')


# ------------------------ Question 2 ----------------------------

# randomizing and organize files in directories (input/all, input/train, input/test)
print('2.')
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

# 3 face image part
sample_imgs = random.sample(train_imgs, 3)

# create a 1x3 subplot
fig, axes = plot.subplots(1, 3, figsize=(10, 4))

for i, pmg_file in enumerate(sample_imgs):
    img = plot.imread(f"{train_path}/{pmg_file}")  # your style

    # extract the person's ID (number before '_')
    id = pmg_file.split('_')[0]

    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"id: {id}")
    axes[i].axis('off')

plot.tight_layout()
plot.savefig("/workspaces/python/week5/output/ps5-2-0.png")
plot.close()


# ------------------------ Question 2.1 ----------------------------
print('\n2.1')
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
# C = np.linalg.matmul(A,A.T)
# m,n = C.shape
# print(f"{m} by {n}")

# plot.imsave("/workspaces/python/week5/output/ps5-2-1-c.png", C, cmap='gray')


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
# compute eigenfaces (U = A * V)
V = eig_vecs[:, :K]
U = A @ V

U = U / np.linalg.norm(U, axis=0)

m, n = U.shape
print(f"matrix U: {m} by {n}\n")  # Should be 10304 x K

# subplot of 9 eigenfaces
img_h, img_w = 112, 92
fig, axes = plot.subplots(3, 3, figsize=(8, 8))

for i, ax in enumerate(axes.flat[:9]):
    eigenface = U[:, i].reshape((img_h, img_w), order='F')
    eigenface = (eigenface - np.min(eigenface)) / (np.max(eigenface) - np.min(eigenface))
    ax.imshow(eigenface, cmap='gray')
    ax.set_title(f"Eigenface {i+1}")
    ax.axis('off')

plot.tight_layout()
plot.savefig("/workspaces/python/week5/output/ps5-2-1-e.png")


# ------------------------ Question 2.2 ----------------------------
print('2.2')
# a. 
w_training = np.linalg.matmul(U.T,A)
m,n = w_training.shape
print(f"w_training size: {m} by {n}")


# b.
# for the testing data
T = np.empty((10304,1))
for pmg_file in test_imgs:
    img = plot.imread(f"{test_path}/{pmg_file}")
    col_vec = img.reshape(-1, 1, order='F') # 'F' flattens img column-wise instead of default row-wise
    T = np.concatenate((T,col_vec), axis=1)

T = T[:,1:]
T_avg = np.mean(T, axis=1) # computes average across rows (returns 10304 x 1)
T_avg = T_avg.reshape(-1,1)
A_test = T - T_avg # define centered data matrix

w_testing = np.linalg.matmul(U.T,A_test)
m,n = w_testing.shape
print(f"w_testing size: {m} by {n}\n")


# ------------------------ Question 2.3 ----------------------------
print('2.3')
# create the labels
train_imgs = [f for f in train_imgs if f.endswith('.pgm')]
test_imgs  = [f for f in test_imgs if f.endswith('.pgm')]

# Extract labels by splitting on '_'
train_labels = np.array([int(f.split('_')[0]) for f in train_imgs])
test_labels  = np.array([int(f.split('_')[0]) for f in test_imgs])

# a.
print('a.')
k_vals = [1,3,5,7,9,11]

for k in k_vals:
    # train knn classifier
    knn = skl.KNeighborsClassifier(n_neighbors=k)
    knn.fit(w_training.T, train_labels)

    # accuracy
    pred_labels = knn.predict(w_testing.T)
    acc = accuracy_score(test_labels, pred_labels)

    print(f"k={k} | accuracy: {acc}")

# b.
# train one vs one and one vs all models (kernels: linear, 3rd order polynomial, gaussian rbf)
linear_svm = SVC(kernel='linear', random_state=42) 
poly_svm = SVC(kernel='poly', degree=3, random_state=42)
rbf_svm = SVC(kernel='rbf', random_state=42)

print('\nb.')
start = time.time()
linear_ovo = OneVsOneClassifier(linear_svm)
linear_ovo.fit(w_training.T, train_labels)
end = time.time()
pred_labels = linear_ovo.predict(w_testing.T)
accuracy = accuracy_score(test_labels, pred_labels)
print(f"linear svm one vs one training time: {end-start} secs")
print(f"linear svm one vs one testing accuracy: {accuracy}")

start = time.time()
linear_ova = OneVsRestClassifier(linear_svm)
linear_ova.fit(w_training.T, train_labels)
end = time.time()
pred_labels = linear_ova.predict(w_testing.T)
accuracy = accuracy_score(test_labels, pred_labels)
print(f"linear svm one vs all training time: {end-start} secs")
print(f"linear svm one vs all testing accuracy: {accuracy}\n")

start = time.time()
poly_ovo = OneVsOneClassifier(poly_svm)
poly_ovo.fit(w_training.T, train_labels)
end = time.time()
pred_labels = poly_ovo.predict(w_testing.T)
accuracy = accuracy_score(test_labels, pred_labels)
print(f"poly svm one vs one training time: {end-start} secs")
print(f"poly svm one vs one testing accuracy: {accuracy}")

start = time.time()
poly_ova = OneVsRestClassifier(poly_svm)
poly_ova.fit(w_training.T, train_labels)
end = time.time()
pred_labels = poly_ova.predict(w_testing.T)
accuracy = accuracy_score(test_labels, pred_labels)
print(f"poly svm one vs all training time: {end-start} secs")
print(f"poly svm one vs all testing accuracy: {accuracy}\n")

start = time.time()
rbf_ovo = OneVsOneClassifier(rbf_svm)
rbf_ovo.fit(w_training.T, train_labels)
end = time.time()
pred_labels = rbf_ovo.predict(w_testing.T)
accuracy = accuracy_score(test_labels, pred_labels)
print(f"rbf svm one vs one training time: {end-start} secs")
print(f"rbf svm one vs one testing accuracy: {accuracy}")

start = time.time()
rbf_ova = OneVsRestClassifier(rbf_svm)
rbf_ova.fit(w_training.T, train_labels)
end = time.time()
pred_labels = rbf_ova.predict(w_testing.T)
accuracy = accuracy_score(test_labels, pred_labels)
print(f"rbf svm one vs all training time: {end-start} secs")
print(f"rbf svm one vs all testing accuracy: {accuracy}")

