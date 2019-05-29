# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import contrib

tf.enable_eager_execution()
print('Tensorflow verison:', tf.__version__)
print('Eager execution:', tf.executing_eagerly())

TRAIN_DATASET_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv'

train_dataset_fp = tf.keras.utils.get_file(fname = os.path.basename(TRAIN_DATASET_URL), origin = TRAIN_DATASET_URL)
print('Local copy of the dataset file:', train_dataset_fp)

# column order in csv file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
feature_names = column_names[:-1]
label_name = column_names[-1]
print('Features:', feature_names)
print('Label:', label_name)

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

batch_size = 32
train_dataset = tf.contrib.data.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names = column_names,
    label_name = label_name,
    num_epochs = 1
)

# # inspect a batch of features
# features, labels = next(iter(train_dataset))
# print(features)

# plt.scatter(
#     features['petal_length'].numpy(),
#     features['sepal_length'].numpy(),
#     c = labels.numpy(),
#     cmap = 'viridis'
# )
# plt.xlabel("Petal length")
# plt.ylabel("Sepal length")
# plt.show()


def pack_features_vector(features, labels):
    '''Pack the features into a single array.'''
    features = tf.stack(list(features.values()), axis = 1)
    return features, labels


train_dataset = train_dataset.map(pack_features_vector)
features, labels = next(iter(train_dataset))
print(features[:5])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation = tf.nn.relu, input_shape = (4,)),
    tf.keras.layers.Dense(10, activation = tf.nn.relu),
    tf.keras.layers.Dense(3)
])

predictions = model(features)
print(tf.nn.softmax(predictions[:5]))
print('Prediction: {}'.format(tf.argmax(predictions, axis = 1)))
print('Labels:     {}'.format(labels))


def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels = y, logits = y_)


l = loss(model, features, labels)
print('Loss test: {}'.format(l))


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)

global_step = tf.Variable(0)
loss_value, grads = grad(model, features, labels)
print('Step: {}, initial loss: {}'.format(global_step.numpy(), loss_value.numpy()))
optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)
print('Step: {},         loss: {}'.format(global_step.numpy(), loss(model, features, labels).numpy()))

tfe = contrib.eager
# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201
for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()
    
    # train loop
    for sample, label in train_dataset:
        # optimize the model
        loss_value, grads = grad(model, sample, label)
        optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)

        # track progress
        epoch_loss_avg(loss_value)    # add current batch loss
        # compare predicted label to actual label
        epoch_accuracy(tf.argmax(model(sample), axis = 1, output_type = tf.int32), label)
    
    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())
    if epoch % 10 == 0:
        print('Epoch {:03d}: loss: {:.6f}, acc: {:.6f}'.format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))

fig, axes = plt.subplots(2, sharex = True, figsize = (12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel('Loss', fontsize = 14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel('Accuracy', fontsize = 14)
axes[1].set_xlabel('Epoch', fontsize = 14)
axes[1].plot(train_accuracy_results)
plt.show()

test_url = 'https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv'
test_fp = tf.keras.utils.get_file(fname = os.path.basename(test_url), origin = test_url)
test_dataset = tf.contrib.data.make_csv_dataset(
    test_fp,
    batch_size,
    column_names = column_names,
    label_name = 'species',
    num_epochs = 1,
    shuffle = False
)
test_dataset = test_dataset.map(pack_features_vector)

test_accuracy = tfe.metrics.Accuracy()
for x, y in test_dataset:
    logits = model(x)
    prediction = tf.argmax(logits, axis = 1, output_type = tf.int32)
    test_accuracy(prediction, y)
print('Test accuracy: {:.6f}'.format(test_accuracy.result()))
