# -*- coding: utf-8 -*-

import tensorflow as tf

tf.enable_eager_execution()

x = tf.ones((2, 2))

with tf.GradientTape(persistent = True) as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

dz_dx = t.gradient(z, x)
print(dz_dx)
for i in [0, 1]:
    for j in [0, 1]:
        print(dz_dx[i][j].numpy())

dz_dy = t.gradient(z, y)
print(dz_dy)

del t