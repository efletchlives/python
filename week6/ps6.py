import numpy as np
import scipy.io as scy
import random
import matplotlib.pyplot as plot
import os

# functions
import predict
import nnCost
import sigmoidGradient
import sGD

vehicles = ['Airplane','Automobile','Truck']

# question 0: done
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

# question 1: done

# a.
θ = scy.loadmat('/workspaces/python/week6/input/HW6_weights_3_full.mat')
θ1 = θ['Theta1']
θ2 = θ['Theta2']

p, h_x = predict.predict(θ1, θ2, X)

acc = np.mean(p == y_labels) * 100
print(f"prediction accuracy: {acc:.2f}%")


# question 2: done
# a.
K = 3
λ = [0.1,1,2]

for i in range(3):
    J = nnCost.nnCost(θ1, θ2, X, y_labels, K, λ[i])
    print(f"the cost for λ = {λ[i]}: {J}") 


# question 3: done
z = np.array([-10,0,10]).T
g_prime = sigmoidGradient.sigmoidGradient(z)
print(f'the sigmoid gradient when z=[-10,0,10]\': {g_prime}')

# question 4: 
input_layer_size = 1024
hidden_layer_size = 40
num_labels = 3
λ = 0.1
alpha = 0.10
MaxEpochs = 50

θ1, θ2 = sGD.sGD(input_layer_size, hidden_layer_size, num_labels, X_train, y_train, λ, alpha, MaxEpochs)


