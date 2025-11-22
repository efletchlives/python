import numpy as np
import scipy.io as scy
import random
import matplotlib.pyplot as plot

# question 1
# a.
digits = scy.loadmat(r'C:\Users\eflet\repos\python\week8\input\HW8_data1.mat')
X = digits['X']
y = digits['y']

m, n = X.shape
img_nums = random.sample(range(0, m), 20)

f, arr = plot.subplots(4, 5, figsize=(10, 8))

for i in range(4):
    for j in range(5):
        idx = i * 5 + j
        img = X[img_nums[idx]].reshape(20, 20).T
        arr[i, j].imshow(img, cmap='gray')
        arr[i, j].axis('off')

plot.savefig(r'C:\Users\eflet\repos\python\week8\output\ps8-1-a-1.png')

# b. 
idx = np.random.permutation(range(5000))
X_train = X[idx[:4300]]
y_train = y[idx[:4300]]-1
X_test = X[idx[4300:5000]]
y_test = y[idx[4300:5000]]-1

# c.
idx1 = np.random.choice(len(X_train),size=1250, replace=True)
idx2 = np.random.choice(len(X_train),size=1250, replace=True)
idx3 = np.random.choice(len(X_train),size=1250, replace=True)
idx4 = np.random.choice(len(X_train),size=1250, replace=True)
idx5 = np.random.choice(len(X_train),size=1250, replace=True)

X1, y1 = X_train[idx1], y_train[idx1].ravel() # bc of error with SVM
X2, y2 = X_train[idx2], y_train[idx2].ravel()
X3, y3 = X_train[idx3], y_train[idx3].ravel()
X4, y4 = X_train[idx4], y_train[idx4].ravel()
X5, y5 = X_train[idx5], y_train[idx5].ravel()


# d. SVM 
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

svm = BaggingClassifier(estimator=OneVsRestClassifier(SVC(kernel='rbf')), n_estimators=5)
svm.fit(X1, y1)

# i.
print('SVM accuracies:')
y_pred = svm.predict(X1)
print('accuracy of SVM on training set (X1):',accuracy_score(y1,y_pred))

# ii.
y_pred = svm.predict(X2)
print('accuracy of SVM on training set (X2):',accuracy_score(y2,y_pred))
y_pred = svm.predict(X3)
print('accuracy of SVM on training set (X3):',accuracy_score(y3,y_pred))
y_pred = svm.predict(X4)
print('accuracy of SVM on training set (X4):',accuracy_score(y4,y_pred))
y_pred = svm.predict(X5)
print('accuracy of SVM on training set (X5):',accuracy_score(y5,y_pred))

# iii.
y_pred = svm.predict(X_test)
print('accuracy of SVM on testing set:',accuracy_score(y_test,y_pred),'\n')


# e. KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X2,y2)

# i.
print('KNN accuracies:')
y_pred = knn.predict(X2)
print('accuracy of KNN on training set (X2):',accuracy_score(y2,y_pred))

# ii.
y_pred = knn.predict(X1)
print('accuracy of KNN on training set (X1):',accuracy_score(y1,y_pred))
y_pred = knn.predict(X3)
print('accuracy of KNN on training set (X3):',accuracy_score(y3,y_pred))
y_pred = knn.predict(X4)
print('accuracy of KNN on training set (X4):',accuracy_score(y4,y_pred))
y_pred = knn.predict(X5)
print('accuracy of KNN on training set (X5):',accuracy_score(y5,y_pred))

# iii.
y_pred = knn.predict(X_test)
print('accuracy of KNN on testing set:',accuracy_score(y_test,y_pred),'\n')


# f. logistic regression
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X3,y3)

# i.
print('logistic regression accuracies:')
y_pred = knn.predict(X3)
print('accuracy of logistic regression on training set (X3):',accuracy_score(y3,y_pred))

# ii.
y_pred = knn.predict(X1)
print('accuracy of logistic regression on training set (X1):',accuracy_score(y1,y_pred))
y_pred = knn.predict(X2)
print('accuracy of logistic regression on training set (X2):',accuracy_score(y2,y_pred))
y_pred = knn.predict(X4)
print('accuracy of logistic regression on training set (X4):',accuracy_score(y4,y_pred))
y_pred = knn.predict(X5)
print('accuracy of logistic regression on training set (X5):',accuracy_score(y5,y_pred))

# iii.
y_pred = knn.predict(X_test)
print('accuracy of logistic regression on testing set:',accuracy_score(y_test,y_pred),'\n')


# g. neural network
import tensorflow as tf

nn = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
    ])
nn.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
nn.fit(X4,y4,epochs=10)

# i.
print('neural network accuracies:')
y_pred = knn.predict(X4)
print('accuracy of neural network on training set (X4):',accuracy_score(y4,y_pred))

# ii.
y_pred = knn.predict(X1)
print('accuracy of neural network on training set (X1):',accuracy_score(y1,y_pred))
y_pred = knn.predict(X2)
print('accuracy of neural network on training set (X2):',accuracy_score(y2,y_pred))
y_pred = knn.predict(X3)
print('accuracy of neural network on training set (X3):',accuracy_score(y3,y_pred))
y_pred = knn.predict(X5)
print('accuracy of neural network on training set (X5):',accuracy_score(y5,y_pred))

# iii.
y_pred = knn.predict(X_test)
print('accuracy of neural network on testing set:',accuracy_score(y_test,y_pred),'\n')


# h. random forest 
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=12)
rf.fit(X5,y5)

# i.
print('random forest accuracies:')
y_pred = knn.predict(X5)
print('accuracy of random forest on training set (X5):',accuracy_score(y5,y_pred))

# ii.
y_pred = knn.predict(X1)
print('accuracy of random forest on training set (X1):',accuracy_score(y1,y_pred))
y_pred = knn.predict(X2)
print('accuracy of random forest on training set (X2):',accuracy_score(y2,y_pred))
y_pred = knn.predict(X3)
print('accuracy of random forest on training set (X3):',accuracy_score(y3,y_pred))
y_pred = knn.predict(X4)
print('accuracy of random forest on training set (X4):',accuracy_score(y4,y_pred))

# iii.
y_pred = knn.predict(X_test)
print('accuracy of random forest on testing set:',accuracy_score(y_test,y_pred),'\n')


# i.
from scipy import stats
svm_pred = svm.predict(X_test)
knn_pred = knn.predict(X_test)
log_pred = log_reg.predict(X_test)
nn_pred = np.argmax(nn.predict(X_test), axis=1) # convert from probabilities to class labels
rf_pred = rf.predict(X_test)

all_pred = np.vstack([svm_pred, knn_pred, log_pred, nn_pred, rf_pred])
all_pred, _ = stats.mode(all_pred, axis=0) # take the mode to get majority voting rule prediction

print('\naccuracy of ensemble on testing set:', accuracy_score(y_test,all_pred),'\n')


# j. 
# the SVM has the best accuracy hanging in the range of 90%/98%. the KNN, logistic regression, neural network, and random forest models stay close to 90%
# bagging does not really help in this case because each model hangs around the same range anyways. 


# ------------------------------------------------------------------------------------------------------------------------
# 2. 
from PIL import Image
from segment_kmeans import kmeans


im1 = Image.open(r'C:\Users\eflet\repos\python\week8\input\photos\im1.jpg')
im1 = im1.resize((100,100))
im1 = np.array(im1).astype(float)/255.0

im2 = Image.open(r'C:\Users\eflet\repos\python\week8\input\photos\im2.jpg')
im2 = im2.resize((100,100))
im2 = np.array(im2).astype(float)/255.0

im3 = Image.open(r'C:\Users\eflet\repos\python\week8\input\photos\im3.png')
im3 = im3.resize((100,100))
im3 = np.array(im3).astype(float)/255.0

K = [3,5,7]
iters = [7,15,25]
R = [5,8,12]
for i in range(3):
    im1_out = kmeans(im1, K[i], iters[i], R[i])
    im1_out = (im1_out * 255).astype(np.uint8)
    plot.imsave(f'C:/Users/eflet/repos/python/week8/output/photos/im1_{K[i]}_{iters[i]}_{R[i]}.png', im1_out)
    
    im2_out = kmeans(im2, K[i], iters[i], R[i])
    im2_out = (im2_out * 255).astype(np.uint8)
    plot.imsave(f'C:/Users/eflet/repos/python/week8/output/photos/im2_{K[i]}_{iters[i]}_{R[i]}.png', im2_out)
    
    im3_out = kmeans(im3, K[i], iters[i], R[i])
    im3_out = (im3_out * 255).astype(np.uint8)
    plot.imsave(f'C:/Users/eflet/repos/python/week8/output/photos/im3_{K[i]}_{iters[i]}_{R[i]}.png', im3_out)

