"""
" recognize_digits.py
"
" This file contains Artificial Neural Network (ANN) implementation
" to recognize handwritten digits using Tensorflow.
"
:Copyright: Mikko Majamaa
:Author: Mikko Majamaa
:Date: 23 July, 2020
:Version: 1.0.0
"""


# built-in package imports
import sys
# third party package imports
import tensorflow as tf
import numpy as np


training_set, test_set = tf.keras.datasets.mnist.load_data()
# training labels and training set
training_labels = training_set[1]
training_labels = np.reshape(training_labels, (60000, 1))
# modify the training labels such that each label is a vector of length 10 and set
# the index of the label to 1 
training_labels_modified = np.zeros(shape=(60000, 10), dtype=np.float32)
for i in range(0, 60000):
    training_labels_modified[i][int(training_labels[i])] = np.float32(1)
# modify the training set such that each row contains a vector that is a flattened version of the image
training_set = training_set[0]
training_set = np.reshape(training_set, (60000, 784))

# define mini-batch size
batch_size = 100

# define the s.t.d. of the variable initialization distributions
init_std = 0.5

# switch to the v1 routines
tf.compat.v1.disable_eager_execution()

### define the computational graphs

# define placeholders for the input data
x = tf.compat.v1.placeholder(tf.float32, [None, 784])
labels = tf.compat.v1.placeholder(tf.float32, [None, 10])

# first networt layer
W1 = tf.Variable(tf.random.truncated_normal([784, 100], stddev=init_std))
b1 = tf.Variable(tf.random.truncated_normal([100], stddev=init_std))
y1 = tf.tanh(tf.matmul(x, W1) + b1)

# second network layer
W2 = tf.Variable(tf.random.truncated_normal([100, 30], stddev=init_std))
b2 = tf.Variable(tf.random.truncated_normal([30], stddev=init_std))
y2 = tf.tanh(tf.matmul(y1, W2) + b2)

# output network layer
W3 = tf.Variable(tf.random.truncated_normal([30, 10], stddev=init_std))
b3 = tf.Variable(tf.random.truncated_normal([10], stddev=init_std))
output = tf.tanh(tf.matmul(y2, W3) + b3)

# loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=labels))

# optimizer
train_step = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.cast(tf.argmax(output, 1), tf.float32), tf.cast(tf.argmax(labels, 1), tf.float32))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

### run the session

# create a new session
sess = tf.compat.v1.Session()

# initialize the variables (weights and biases)
sess.run(tf.compat.v1.global_variables_initializer())

# train the ANN
for i in range(0, 5000):
    # randomly select a subset of the training data
    batch_idxs = np.random.randint(training_set.shape[0], size=batch_size)
    batch_data = training_set[batch_idxs, :]
    batch_labels = training_labels_modified[batch_idxs, :]

    # run the optimization step
    sess.run(train_step, feed_dict={x: batch_data, labels: batch_labels})

    # evaluate the accuracy
    if (i % 100 == 0):
        result = sess.run(accuracy, feed_dict={x: batch_data, labels: batch_labels})
        print('Epoch {}, accuracy = {}%'.format(i, result * 100))

# evaluate the ANN with the test data
test_labels = test_set[1]
test_set = test_set[0]
batch_data = test_set.reshape(10000, 784)
batch_labels = test_labels.reshape(10000, 1)
batch_labels_modified = np.zeros(shape=(10000, 10), dtype=np.float32)
for i in range(0, 10000):
    batch_labels_modified[i][int(batch_labels[i])] = np.float32(1)
result = sess.run(accuracy, feed_dict={x: batch_data, labels: batch_labels_modified})
print('Testing set evaluation, accuracy = {}%'.format(result * 100))