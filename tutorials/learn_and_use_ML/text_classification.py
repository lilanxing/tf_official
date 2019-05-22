# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# dataset
imdb = tf.keras.datasets.imdb
(train_data, train_lebels), (test_data, test_labels) = imdb.load_data(num_words = 10000)
print('training entries: {}, lebels: {}'.format(len(train_data), len(train_lebels)))
print(train_data[0])
print(len(train_data[0]), len(train_data[1]))

# a dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# the first indices are reserved
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2    # unknown
word_index['<UNUSED'] = 3
reversed_word_index = {v: k for k, v in word_index.items()}


def decode_review(text):
    return ' '.join([reversed_word_index.get(i, '?') for i in text])


print(decode_review(train_data[0]))

# prepare the data
train_data = tf.keras.preprocessing.sequence.pad_sequences(
    train_data, value = word_index['<PAD>'], padding = 'post', maxlen = 256
)
test_data = tf.keras.preprocessing.sequence.pad_sequences(
    test_data, value = word_index['<PAD>'], padding = 'post', maxlen = 256
)
print('after preprocessing...')
print(len(train_data[0]), len(train_data[1]))
print(train_data[0])

# input shape is the vocabulary count userd for the movie reviews (10000 words)
vocab_size = 10000

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 16))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(16, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation = tf.nn.sigmoid))

model.summary()

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])

# validation set
x_val, partial_x_train = train_data[:10000], train_data[10000:]
y_val, partial_y_train = train_lebels[:10000], train_lebels[10000:]

# train the model
history = model.fit(
    partial_x_train, 
    partial_y_train, 
    epochs = 40, 
    batch_size = 512,
    validation_data = (x_val, y_val),
    verbose = 1
)

# evaluate the model
results = model.evaluate(test_data, test_labels)
print(results)

history_dict = history.history
print(history_dict.keys())

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Trainign and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()