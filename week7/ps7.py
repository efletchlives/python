import numpy as np
import matplotlib.pyplot as plot
import matplotlib.image as matimg
import scipy.io as scy
import random
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # hide info and warnings from tf
warnings.filterwarnings('ignore')         # hide python warnings

# cnn libraries
import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.utils import plot_model

# Load data
# data = scy.loadmat("/workspaces/python/week7/input/HW6_Data2_full.mat")

# X = data['X']
# y_labels = data['y_labels'].flatten()  # Flatten to 1D array

# # a. Randomly shuffle and split into train/test sets
# idx = np.arange(len(X))
# np.random.shuffle(idx)

# train_idx = idx[:13000]
# test_idx = idx[13000:15000]

# X_train = X[train_idx]
# y_train = y_labels[train_idx]
# X_test = X[test_idx]
# y_test = y_labels[test_idx]


# imgs_train = X_train.reshape(-1, 32, 32, 1)
# imgs_train = np.rot90(imgs_train, -1, axes=(1,2))

# vehicles = ['Airplane', 'Automobile', 'Truck']

# # b. choose 12 random images
# idx = random.sample(range(len(imgs_train)), 12)

# fig, axes = plot.subplots(3, 4, figsize=(6, 6))
# for i, ax in enumerate(axes.flat):
#     img = imgs_train[idx[i]].squeeze()
#     label = vehicles[y_train[idx[i]] - 1]
#     ax.imshow(img, cmap='gray')
#     ax.set_title(label, fontsize=8, fontweight='bold')
#     ax.axis('off')

# plot.tight_layout()
# plot.savefig('/workspaces/python/week7/output/ps7-1-b-1.png')
# plot.show()

# # c. 
# imgs_test = X_test.reshape(-1, 32, 32, 1)
# imgs_test = np.rot90(imgs_test, -1, axes=(1,2))

# # choose 6 random images
# idx = random.sample(range(len(imgs_test)), 6)

# fig, axes = plot.subplots(3, 2, figsize=(6, 6))
# for i, ax in enumerate(axes.flat):
#     img = imgs_test[idx[i]].squeeze()
#     label = vehicles[y_test[idx[i]] - 1]
#     ax.imshow(img, cmap='gray')
#     ax.set_title(label, fontsize=8, fontweight='bold')
#     ax.axis('off')

# plot.tight_layout()
# plot.savefig('/workspaces/python/week7/output/ps7-1-c-1.png')
# plot.show()

# # d. 
# # turn y_labels to one hot endcoded vectors (y_train)
# m = X_train.shape[0]
# y_train_vec = y_train.flatten()
# y_vec = np.zeros((m, 3))
# for i in range(m):
#     y_vec[i,y_train_vec[i]-1] = 1 # puts 1 at the corresponding spot in the vector
# y_train_vec = y_vec

# # turn y_labels to one hot endcoded vectors (y_test)
# m = X_test.shape[0]
# y_test_vec = y_test.flatten()
# y_vec = np.zeros((m, 3))
# for i in range(m):
#     y_vec[i,y_test_vec[i]-1] = 1 # puts 1 at the corresponding spot in the vector
# y_test_vec = y_vec

# # train cnn model
# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(32,32,1)),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Conv2D(32, kernel_size=5, activation='relu'),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(3, activation='softmax')
# ])

# model.compile(optimizer='adam',loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
# model.summary()
# trained_model = model.fit(imgs_train, y_train_vec, epochs=15, batch_size=150, validation_data=(imgs_test, y_test_vec))

# plot.figure()
# plot.plot(trained_model.history['accuracy'], label='accuracy')
# # plot.plot(trained_model.history['val_accuracy'], label = 'val_accuracy')
# plot.xlabel('epoch')
# plot.ylabel('accuracy')
# plot.ylim([0.5, 1])
# plot.savefig('/workspaces/python/week7/output/ps7-1-e-2.png')

# print(f'accuracy on training data: {trained_model.history['accuracy'][-1]:.3f}')

# # test the model
# test_loss, test_acc = model.evaluate(imgs_test, y_test_vec)
# print(f'accuracy on testing data: {test_acc:.3f}')


# question 2: window based recognition

# read imgs_train and y_train in 
imgs_train = np.zeros((100,96,32,3))
y_train = np.zeros((100,1))

idx = 0
for i in range(0,2):
    for j in range(1,51):
        img = matimg.imread(f'/workspaces/python/week7/input/p2/train_imgs/{i}_{j:02}.jpg')
        # img = img/255.0 in case normalization is needed
        imgs_train[idx] = img
        y_train[idx] = i
        idx += 1

print(f'shape of imgs_train: {imgs_train.shape}')
print(f'shape of y_train: {y_train.shape}')

# read imgs_test and y_test in
imgs_test = np.zeros((10,192,192,3))
y_test = np.zeros((10,1))

idx = 0
for i in range(0,2):
    for j in range(1,6):
        img = matimg.imread(f'/workspaces/python/week7/input/p2/test_imgs/{i}_{j:02}.jpg')
        # img = img/255.0 in case normalization is needed
        imgs_test[idx] = img
        y_test[idx] = i
        idx += 1

print(f'shape of imgs_test: {imgs_test.shape}')
print(f'shape of y_test: {y_test.shape}')


# train cnn model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(96,32,3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
trained_model = model.fit(imgs_train, y_train, epochs=10, batch_size=16, verbose=1)


i_max, j_max = -1,-1 # holds top left corner of the window with highest probability of car being there
stride = 1
blue = (0,0,255)
thres = 0.7

# nested for loop for sliding window
for K in range(1,11): # K = 1:10
    max_prob = 0 # holds max probability per img
    for i in range(0,192-96,stride):
        for j in range(0,192-32,stride):
            img = imgs_test[K-1]
            window = img[i:i+96, j:j+32]
            window = np.expand_dims(window, axis=0)

            prob = model.predict(window, verbose=0)[0][0]

            if prob > max_prob:
                max_prob = prob
                i_max, j_max = i,j
    
    if(max_prob > thres):
        img[i_max:i_max+96,j_max:j_max+32] = blue
    img = np.clip(img, 0, 255).astype(np.uint8)
    matimg.imsave(f'/workspaces/python/week7/output/ps7-2-c-{K}.png',img) # saves image with window of highest probability of car

