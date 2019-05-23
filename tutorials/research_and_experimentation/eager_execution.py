# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import tempfile

tf.enable_eager_execution()

# Tensors
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))
print(tf.encode_base64("hello world"))

# Operator overloading is also supported
print(tf.square(2) + tf.square(3))

x = tf.matmul([[1]], [[2, 3]])
print(x.shape)
print(x.dtype)

# Numpy compatibility
ndarray = np.ones([3, 3])

print('Tensorflow operations convert numpy arrays to Tensors automatically')
tensor = tf.multiply(ndarray, 42)
print(tensor)

print('And Numpy operations convert Tensors to numpy array automatically')
print(np.add(tensor, 1))

print('The .numpy() method explicitly converts a Tensor to a numpy array')
print(tensor.numpy())

x = tf.random_uniform([3, 3])
print('gpu available:', tf.test.is_gpu_available())
print('tensor on GPU:', 'GPU' in x.device)
print(x.device)


def time_matmul(x):
    start = time.time()
    for _ in range(100):
        tf.matmul(x, x)
    spent = time.time() - start
    print('100 loops: {:.4f}ms'.format(1000 * spent))


# force execution on cpu
print('On cpu:')
with tf.device('CPU'):
    x = tf.random_uniform([1000, 1000])
    print(x.device)
    assert 'CPU' in x.device
    time_matmul(x)

# force execution on GPU if available
if tf.test.is_gpu_available():
    with tf.device('GPU'):
        x = tf.random_uniform([1000, 1000])
        print(x.device)
        assert 'GPU' in x.device
        time_matmul(x)

# Datasets
# create a source dataset
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

_, filename = tempfile.mkstemp()
with open(filename, 'w') as f:
    f.write('''Line 1
    Line 2
    Line 3'''
    )
ds_file = tf.data.TextLineDataset(filename)

ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)

print('Elements of ds_tensors:')
for x in ds_tensors:
    print(x)

print('Elements of ds_file:')
for x in ds_file:
    print(x)
