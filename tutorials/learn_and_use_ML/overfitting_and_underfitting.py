# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

print(tf.__version__)

NUM_WORDS = 10000
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(num_words = NUM_WORDS)


def multi_hot_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, word_idx in enumerate(sequences):
        results[i, word_idx] = 1.0
    return results


# print(train_data[0])
print('before train data shape:', train_data.shape)
train_data = multi_hot_sequences(train_data, dimension = NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension = NUM_WORDS)
# print(train_data[0])
print('after train data shape:', train_data.shape)

# plt.plot(train_data[0])
# plt.show()

# baseline model
baseline_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation = tf.nn.relu, input_shape = (NUM_WORDS,)),
    tf.keras.layers.Dense(16, activation = tf.nn.relu),
    tf.keras.layers.Dense(1, activation = tf.nn.sigmoid)
])
baseline_model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy', 'binary_crossentropy']
)
print(baseline_model.summary())

baseline_history = baseline_model.fit(
    train_data,
    train_labels,
    epochs = 20,
    batch_size = 512,
    validation_data = (test_data, test_labels),
    verbose = 2
)

# # smaller model
# smaller_model = tf.keras.Sequential([
#     tf.keras.layers.Dense(4, activation = tf.nn.relu, input_shape = (NUM_WORDS,)),
#     tf.keras.layers.Dense(4, activation = tf.nn.relu),
#     tf.keras.layers.Dense(1, activation = tf.nn.sigmoid)
# ])
# smaller_model.compile(
#     optimizer = 'adam',
#     loss = 'binary_crossentropy',
#     metrics = ['accuracy', 'binary_crossentropy']
# )
# print(smaller_model.summary())

# smaller_history = smaller_model.fit(
#     train_data,
#     train_labels,
#     epochs = 20,
#     batch_size = 512,
#     validation_data = (test_data, test_labels),
#     verbose = 2
# )

# # bigger model
# bigger_model = tf.keras.Sequential([
#     tf.keras.layers.Dense(512, activation = tf.nn.relu, input_shape = (NUM_WORDS,)),
#     tf.keras.layers.Dense(512, activation = tf.nn.relu),
#     tf.keras.layers.Dense(1, activation = tf.nn.sigmoid)
# ])
# bigger_model.compile(
#     optimizer = 'adam',
#     loss = 'binary_crossentropy',
#     metrics = ['accuracy', 'binary_crossentropy']
# )
# print(bigger_model.summary())

# bigger_history = bigger_model.fit(
#     train_data,
#     train_labels,
#     epochs = 20,
#     batch_size = 512,
#     validation_data = (test_data, test_labels),
#     verbose = 2
# )


def plot_history(histories, key = 'binary_crossentropy'):
    plt.figure(figsize = (16, 10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key], '--', label = name.title() + ' Val')
        plt.plot(history.epoch, history.history[key], color = val[0].get_color(), label = name.title() + ' Train')
    
    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()
    plt.xlim([0, max(history.epoch)])
    plt.show()


# plot_history([
#     ('baseline', baseline_history),
#     ('smaller', smaller_history),
#     ('bigger', bigger_history)
# ])

# # l2 regularization
# l2_model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(16, kernel_regularizer = tf.keras.regularizers.l2(0.001), activation = tf.nn.relu, input_shape = (NUM_WORDS,)),
#     tf.keras.layers.Dense(16, kernel_regularizer = tf.keras.regularizers.l2(0.001), activation = tf.nn.relu),
#     tf.keras.layers.Dense(1, activation = tf.nn.sigmoid)
# ])
# l2_model.compile(
#     optimizer = 'adam',
#     loss = 'binary_crossentropy',
#     metrics = ['accuracy', 'binary_crossentropy']
# )
# l2_model_history = l2_model.fit(
#     train_data,
#     train_labels,
#     epochs = 20,
#     batch_size = 512,
#     validation_data = (test_data, test_labels),
#     verbose = 2
# )

# plot_history([
#     ('baseline', baseline_history),
#     ('l2', l2_model_history)
# ])

# dropout
dropout_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation = tf.nn.relu, input_shape = (NUM_WORDS,)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(16, activation = tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation = tf.nn.sigmoid)
])
dropout_model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy', 'binary_crossentropy']
)
dropout_model_history = dropout_model.fit(
    train_data,
    train_labels,
    epochs = 20,
    batch_size = 512,
    validation_data = (test_data, test_labels),
    verbose = 2
)

plot_history([
    ('baseline', baseline_history),
    ('dropout', dropout_model_history)
])
