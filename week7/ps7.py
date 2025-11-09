import numpy as np
import matplotlib.pyplot as plot
import scipy.io as scy
import random
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # hide info and warnings from TF
warnings.filterwarnings('ignore')         # hide python warnings

# cnn libraries
import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.utils import plot_model

# Load data
data = scy.loadmat("/workspaces/python/week7/input/HW6_Data2_full.mat")

X = data['X']
y_labels = data['y_labels'].flatten()  # Flatten to 1D array

# a. Randomly shuffle and split into train/test sets
idx = np.arange(len(X))
np.random.shuffle(idx)

train_idx = idx[:13000]
test_idx = idx[13000:15000]

X_train = X[train_idx]
y_train = y_labels[train_idx]
X_test = X[test_idx]
y_test = y_labels[test_idx]


imgs_train = X_train.reshape(-1, 32, 32, 1)
imgs_train = np.rot90(imgs_train, -1, axes=(1,2))

vehicles = ['Airplane', 'Automobile', 'Truck']

# b. choose 12 random images
idx = random.sample(range(len(imgs_train)), 12)

fig, axes = plot.subplots(3, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    img = imgs_train[idx[i]].squeeze()
    label = vehicles[y_train[idx[i]] - 1]
    ax.imshow(img, cmap='gray')
    ax.set_title(label, fontsize=8, fontweight='bold')
    ax.axis('off')

plot.tight_layout()
plot.savefig('/workspaces/python/week7/output/ps7-1-b-1.png')
plot.show()

# c. 
imgs_test = X_test.reshape(-1, 32, 32, 1)
imgs_test = np.rot90(imgs_test, -1, axes=(1,2))

# choose 6 random images
idx = random.sample(range(len(imgs_test)), 6)

fig, axes = plot.subplots(3, 2, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    img = imgs_test[idx[i]].squeeze()
    label = vehicles[y_test[idx[i]] - 1]
    ax.imshow(img, cmap='gray')
    ax.set_title(label, fontsize=8, fontweight='bold')
    ax.axis('off')

plot.tight_layout()
plot.savefig('/workspaces/python/week7/output/ps7-1-c-1.png')
plot.show()

# d. 
# turn y_labels to one hot endcoded vectors (y_train)
m = X_train.shape[0]
y_train_vec = y_train.flatten()
y_vec = np.zeros((m, 3))
for i in range(m):
    y_vec[i,y_train_vec[i]-1] = 1 # puts 1 at the corresponding spot in the vector
y_train_vec = y_vec

# turn y_labels to one hot endcoded vectors (y_test)
m = X_test.shape[0]
y_test_vec = y_test.flatten()
y_vec = np.zeros((m, 3))
for i in range(m):
    y_vec[i,y_test_vec[i]-1] = 1 # puts 1 at the corresponding spot in the vector
y_test_vec = y_vec

# train cnn model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(32,32,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
model.summary()
trained_model = model.fit(imgs_train, y_train_vec, epochs=15, batch_size=150, validation_data=(imgs_test, y_test_vec))

plot.figure()
plot.plot(trained_model.history['accuracy'], label='accuracy')
# plot.plot(trained_model.history['val_accuracy'], label = 'val_accuracy')
plot.xlabel('epoch')
plot.ylabel('accuracy')
plot.ylim([0.5, 1])
plot.savefig('/workspaces/python/week7/output/ps7-1-e-2.png')

print(f'accuracy on training data: {trained_model.history['accuracy'][-1]:.3f}')

# test the model
test_loss, test_acc = model.evaluate(imgs_test, y_test_vec)
print(f'accuracy on testing data: {test_acc:.3f}')


# question 2: window based recognition
