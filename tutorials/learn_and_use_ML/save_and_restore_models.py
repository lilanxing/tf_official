# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

CHECKPOINT_DIR = '/tmp/tf_official/checkpoints'
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
checkpoint_path = os.path.join(CHECKPOINT_DIR, 'cp.ckpt')

# load data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_labels, test_labels = train_labels[:1000], test_labels[:1000]
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


def create_model():
    '''create a short sequential model'''
    model = keras.models.Sequential([
        keras.layers.Dense(512, activation = keras.activations.relu, input_shape = (784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation = keras.activations.softmax)
    ])
    model.compile(
        optimizer = keras.optimizers.Adam(),
        loss = keras.losses.sparse_categorical_crossentropy,
        metrics = ['accuracy']
    )
    return model


# # create checkpoint callback
# cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only = True, verbose = 1)

# # create a basic model instance
# model = create_model()
# print(model.summary())
# model.fit(
#     train_images,
#     train_labels,
#     epochs = 10,
#     validation_data = (test_images, test_labels),
#     callbacks = [cp_callback]
# )

# new model
model = create_model()
loss, acc = model.evaluate(test_images, test_labels)
print('Untrained model, acc: {:.2f}%'.format(acc))

model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print('Restored model, acc: {:.2f}%'.format(acc))
