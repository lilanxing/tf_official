# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf

column_names = [
    'MPG', 'Cylinders', 'Displacement', 'Horsepower',
    'Weight', 'Acceleration', 'Model Year', 'Origin'
]
dataset_path = 'auto-mpg.data'
raw_dataset = pd.read_csv(
    dataset_path,
    names = column_names,
    na_values = '?',
    comment = '\t',
    sep = ' ',
    skipinitialspace = True
)
dataset = raw_dataset.copy()
dataset.tail()

# clean the data
print(dataset.isna().sum())
dataset = dataset.dropna()

origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0
dataset.tail()

train_dataset = dataset.sample(frac = 0.8, random_state = 0)
test_dataset = dataset.drop(train_dataset.index)

# inspect the data
# sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind = "kde")
# plt.show()

train_stats = train_dataset.describe()
train_stats.pop('MPG')
train_stats = train_stats.transpose()
print(train_stats)

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


def norm(x):
    '''normalize the data'''
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation = tf.nn.relu, input_shape = [len(train_dataset.keys())]),
        tf.keras.layers.Dense(64, activation = tf.nn.relu),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(
        loss = 'mean_squared_error',
        optimizer = optimizer,
        metrics = ['mean_absolute_error', 'mean_squared_error']
    )
    return model


model = build_model()
print(model.summary())


class PrintDot(tf.keras.callbacks.Callback):
    '''display training progress by printing a single dot for each completed epoch'''

    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end = '')


EPOCHS = 1000

history = model.fit(
    normed_train_data,
    train_labels,
    epochs = EPOCHS,
    validation_split = 0.2,
    verbose = 0,
    callbacks = [PrintDot()]
)