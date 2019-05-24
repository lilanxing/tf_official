# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt

tf.enable_eager_execution()

# Variables
# using python state
# x = tf.zeros((10, 10))
# x += 2
# print(x)

v = tf.Variable(1.0)
assert v.numpy() == 1.0
print(v)

v.assign(3.0)
assert v.numpy() == 3.0
print(v)

v.assign(tf.square(v))
assert v.numpy() == 9.0
print(v)

############ Example: Fitting a linear model ############
# 1. Define the model
# 2. Define a loss function
# 3. Obtain training data
# 4. Run through the training data and use an optimizer 
#    to adjust the variables to fit the data
#########################################################


# define the model
class Model(object):

    def __init__(self):
        # initialize variable to (5.0, 0.0), in practice these
        # should be initialized to random values.
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)
    
    def __call__(self, x):
        return self.W * x + self.b


model = Model()
assert model(3.0).numpy() == 15.0


# define a loss function
def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))


# obtain training data
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs = tf.random_normal(shape = [NUM_EXAMPLES])
noise = tf.random_normal(shape = [NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

plt.scatter(inputs, outputs, c = 'b')
plt.scatter(inputs, model(inputs), c = 'r')
plt.show()

print('current loss:', loss(model(inputs), outputs).numpy())


# define a training loop
def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)


Ws, bs = [], []
epochs = range(100)
for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(model(inputs), outputs)

    train(model, inputs, outputs, learning_rate = 0.1)
    
    print('Epoch {:2d}: W = {:.2f}, b = {:.2f}, loss = {:.5f}'.format(
        epoch, Ws[-1], bs[-1], current_loss
    ))

plt.plot(epochs, Ws, 'r', epochs, bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--', [TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'true W', 'true b'])
plt.show()
