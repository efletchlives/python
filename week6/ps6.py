import numpy as np
import scipy.io as scy
import random
import matplotlib.pyplot as plot
import os
import time

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
alpha = 0.01
MaxEpochs = 50

start_time = time.time()
θ1, θ2, costs = sGD.sGD(input_layer_size, hidden_layer_size, num_labels, X_train, y_train, λ, alpha, MaxEpochs)
end_time = time.time()

J = nnCost.nnCost(θ1, θ2, X_train, y_train, num_labels, λ)
print(f'cost after 50 epochs: {J}')
print(f'time to run 50 epochs: {end_time-start_time}')

plot.figure(figsize=(10, 6))
plot.plot(range(1, MaxEpochs+1), costs, 'b-', linewidth=2, marker='o', markersize=4)
plot.xlabel('Epoch', fontsize=12)
plot.ylabel('Cost', fontsize=12)
plot.title(f'Training Cost vs Epoch (λ={λ}, α={alpha})', fontsize=14)
plot.grid(True, alpha=0.3)
plot.tight_layout()
plot.savefig('/workspaces/python/week6/output/ps6-4-e-1.png')
plot.close()


# question 5:
λ = [0.01, 0.1, 0.2, 1]
MaxEpochs = [50,300]

results = {}

for num_epochs in MaxEpochs:
    for i in λ:
        θ1, θ2, costs = sGD.sGD(input_layer_size, hidden_layer_size, num_labels, X_train, y_train, i, alpha, num_epochs)

        # calculate training accuracy
        p_train , _ = predict.predict(θ1, θ2, X_train)
        train_acc = np.mean(p_train == y_train) * 100

        # calculate testing accuracy
        p_test, _ = predict.predict(θ1, θ2, X_test)
        test_acc = np.mean(p_test == y_test) * 100

        # calculate final cost
        final_cost = nnCost.nnCost(θ1, θ2, X_train, y_train, num_labels, i)

        results[(i, num_epochs)] = {train_acc, test_acc, final_cost}
        print(f'training accuracy (λ = {i}, max epochs = {num_epochs}): {train_acc:.2f}')
        print(f'testing accuracy (λ = {i}, max epochs = {num_epochs}): {test_acc:.2f}')
        print(f'final cost (λ = {i}, max epochs = {num_epochs}): {final_cost:.2f}\n')    




