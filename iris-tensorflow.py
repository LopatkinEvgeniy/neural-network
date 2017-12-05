import csv
import random
import numpy as np
import tensorflow as tf

# data prepration
iris_codes = {
    'setosa': [1, 0, 0],
    'versicolor': [0, 1, 0],
    'virginica': [0, 0, 1]
}

iris_dataset = []

for i, row in enumerate(csv.reader(open('data/iris.csv', 'r'))):
    if i == 0:
        continue
    iris_name = row[4]
    iris_data = list(map(lambda n: float(n), row[0:4]))
    iris_dataset.append([iris_data, iris_codes[iris_name]])

random.shuffle(iris_dataset)

x = []
y = []

for iris in iris_dataset:
    x.append(iris[0])
    y.append(iris[1])

x = np.array(x, dtype=np.float32)
y = np.array(y, dtype=np.float32)

x_validation = x[:15]
y_validation = y[:15]

x_train = x[15:]
y_train = y[15:]

# settings
INPUT_NODES = 4
OUTPUT_NODES = 3
HIDDEN_NODES = 12

LEARNING_RATE = 0.05
STEPS = 100000

# model
input = tf.placeholder(tf.float32, shape=[None, INPUT_NODES], name="input")
output = tf.placeholder(tf.float32, shape=[None, OUTPUT_NODES], name="output")

weights_hidden = tf.Variable(tf.random_uniform([INPUT_NODES, HIDDEN_NODES], -1, 1), name="hidden_weights")
weights_output = tf.Variable(tf.random_uniform([HIDDEN_NODES, OUTPUT_NODES], -1, 1), name="output_weights")

biases_hidden = tf.Variable(tf.random_uniform([HIDDEN_NODES], -1, 1), name="hidden_biases")
biases_output = tf.Variable(tf.random_uniform([OUTPUT_NODES], -1, 1), name="output_biases")

layer_hidden = tf.sigmoid(tf.matmul(input, weights_hidden) + biases_hidden)
layer_output = tf.sigmoid(tf.matmul(layer_hidden, weights_output) + biases_output)

# initialization
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# train
cost = tf.reduce_mean(tf.square(output - layer_output))
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

for i in range(STEPS):
    sess.run(train_step, feed_dict={input: x_train, output: y_train})

# check
results = sess.run(layer_output, feed_dict={input: x_validation, output: y_validation})

for i in range(len(results)):
    print(y_validation[i], " -> ", results[i])

print('Train cost: ', sess.run(cost, feed_dict={input: x_train, output: y_train}))
print('Cost: ', sess.run(cost, feed_dict={input: x_validation, output: y_validation}))
