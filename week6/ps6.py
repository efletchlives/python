import numpy as np
import scipy.io as scy
import random
import matplotlib.pyplot as plot

import predict

vehicles = ['Airplane','Automobile','Truck']

# question 0
data = scy.loadmat('/workspaces/python/week6/input/HW6_Data2_full.mat')

# a.
# randomly pick 16 images and display in 4x4 grid
X = data['X']
y_labels = data['y_labels']

# choose 16 images
idx = random.sample(range(len(X)), 16)
X_random = X[idx] 
y_random = y_labels[idx]

fig, axes = plot.subplots(4,4,figsize=(6,6))
for i, ax in enumerate(axes.flat):
    img = np.reshape(X_random[i], (32,32))
    img = np.rot90(img,-1)
    ax.imshow(img, cmap='gray')
    ax.set_title(vehicles[y_random[i].item()-1], fontsize=8, fontweight='bold')
    ax.axis('off')

plot.savefig('/workspaces/python/week6/output/ps6-0-a-1.png')

# b. first 13000 for train and 2000 for test
idx = np.arange(len(X))
np.random.shuffle(idx)

train_idx = idx[:13000]
test_idx = idx[13000:15000]

X_train = X[train_idx]
y_train = y_labels[train_idx]
X_test = X[test_idx]
y_test = y_labels[test_idx]

# 1.

# a.

p, h_x = predict.predict(θ1, θ2, X)



