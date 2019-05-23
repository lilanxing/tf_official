# -*- coding: utf-8 -*-

import tensorflow as tf

tf.enable_eager_execution()


class MyDenseLayer(tf.keras.layers.Layer):

    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs
    
    def build(self, input_shape):
        self.kernel = self.add_variable(
            'kernel',
            shape = [int(input_shape[-1]), self.num_outputs]
        )
    
    def call(self, input):
        return tf.matmul(input, self.kernel)


layer = MyDenseLayer(10)
print(layer(tf.zeros([10, 5])))
print(layer.trainable_variables)


class ResnetIdentityBlock(tf.keras.Model):

    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name = '')
        filter1, filter2, filter3 = filters

        self.conv1 = tf.keras.layers.Conv2D(filter1, (1, 1))
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filter2, kernel_size, padding = 'same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2D(filter3, (1, 1))
        self.bn3 = tf.keras.layers.BatchNormalization()
    
    def call(self, input_tensor, training = False):
        x = self.conv1(input_tensor)
        x = self.bn1(x, training = training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training = training)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training = training)
        
        x += input_tensor
        x = tf.nn.relu(x)

        return x


block = ResnetIdentityBlock(1, [1, 2, 3])
print(block(tf.zeros([1, 2, 3, 3])))
print([x.name for x in block.trainable_variables])
