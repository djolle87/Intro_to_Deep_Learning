import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Step 1 load data

df = pd.read_csv('data/data.csv')
df = df.drop(['index', 'price', 'sq_price'], axis=1)
df = df[0:10]

# Step 2 - add labels

# 1 is good buy and 0 is bad buy
df.loc[:, ('y1')] = [1, 1, 1, 0, 0, 1, 0, 1, 1, 1]
# y2 is a negation of y1, opposite
df.loc[:, ('y2')] = df['y1'] == 0
# turn True/False values to 1s and 0s
df.loc[:, ('y2')] = df['y2'].astype(int)

# Step 3 - prepare data for tensorflow

# tensors are a generic version of vectors and matrices
# vector is a list of numbers (1D Tensor)
# matrix is a list of list of numbers (2D Tensor)
# list of list of list of numbers (3D Tensor)
# .....
# Convert features to input tensor
inputX = df.loc[:, ['area', 'bathrooms']].as_matrix()
# Convert labels to input tensors
inputY = df.loc[:, ['y1', 'y2']].as_matrix()

# Step 4 - write out our hyperparameters
learning_rate = 0.000001
training_epochs = 2000
display_step = 50
n_samples = inputY.size

# Step 5 - Create our computation

# for feature input tensors, none means any numbers of examples
# placeholders are gateways for data into our computation graph
x = tf.placeholder(tf.float32, [None, 2])

# create weights
# 2x2 float matrix, that we'll keep updating through the training process
# variables in th hold and update parameters
# in memory buffers containing tensors
W = tf.Variable(tf.zeros([2, 2]))

# add biases
b = tf.Variable(tf.zeros([2]))

# multiply our weights by our inputs, first calculation
# weights are how we govern how data flows in our computation graph
# multiply input by weights and add biasis
y_values = tf.add(tf.matmul(x, W), b)

# apply softmax to value we just created
# softmax is our activation function
y = tf.nn.softmax(y_values)

# feed a matrix of labels
y_ = tf.placeholder(tf.float32, [None, 2])

# Step 6 - perform training

# create our cost function, mean squared error
# reduce sum computes the sum of elements across dimension of a tensor
cost = tf.reduce_sum(tf.pow(y_ - y, 2)) / (2 * n_samples)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# initialize variables and tensorflow session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# training loop
for i in range(training_epochs):
    sess.run(optimizer, feed_dict={x: inputX, y_: inputY})
    # write out logs of training
    if (i) % display_step == 0:
        cc = sess.run(cost, feed_dict={x: inputX, y_: inputY})
        print("Training step:", '%04d' % (i), "cost=", "{:.9f}".format(cc))

print("Optimization Finished!")
training_cost = sess.run(cost, feed_dict={x: inputX, y_: inputY})
print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')


print('Test with train data:')
print(sess.run(y, feed_dict={x: inputX}))
# it is saying that all houses are a good buy (this is 7/10 correct)
# how to improve? add a hidden layer