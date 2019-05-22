# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

# tensorflow and td.keras
import tensorflow as tf
# from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# explore the data
print("train images' shape:", train_images.shape)
print('train labels count:', len(train_labels))
print('train labels:')
print(train_labels)

print("test images' shape:", test_images.shape)
print('test labels count:', len(test_labels))

# preprocess the data
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.figure()
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap = plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# setup the layers
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(128, activation = tf.nn.relu),
    tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])

# compile the model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# train the model
model.fit(train_images, train_labels, epochs = 5)

# evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test loss: {}, test accuracy: {}'.format(test_acc, test_acc))

# make predictions
predictions = model.predict(test_images)
print('predicions 0:')
print(predictions[0])
print('predicted label:', np.argmax(predictions[0]))
print('true label:', test_labels[0])

def plot_image(i, predictions_arrays, true_labels, imgs):
    predictions_array, true_label, img = predictions_arrays[i], true_labels[i], imgs[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap = plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'
    plt.xlabel('{}  {:2.0f}% ({})'.format(
        class_names[predicted_label],
        100 * np.max(predictions_array),
        class_names[true_label],
        color = color
    ))


def plot_value_array(i, predictions_arrays, true_labels):
    predictions_array, true_label = predictions_arrays[i], true_labels[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color = '#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# i = 12
# plt.figure(figsize = (6,3))
# plt.subplot(1, 2, 1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions,  test_labels)
# plt.show()

img = test_images[0]
print('test image 0 shape:', img.shape)

img = np.expand_dims(img, 0)
print('after expand dim test image 0 shape:', img.shape)

predictions_single = model.predict(img)
print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation = 45)
plt.show()

print(np.argmax(predictions_single[0]))
