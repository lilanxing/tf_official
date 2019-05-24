# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt
import tensorflow as tf

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

# inspect a batch of features
features, labels = next(iter(train_dataset))
print(features)

plt.scatter(
    features['petal_length'].numpy(),
    features['sepal_length'].numpy(),
    c = labels.numpy(),
    cmap = 'viridis'
)
plt.show()


