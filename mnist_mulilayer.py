import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
picture_size = 784

# layer 1
x = tf.placeholder(tf.float32, [None, picture_size])
W = tf.Variable(tf.random_uniform(shape=[picture_size, picture_size//2], minval=-0.1, maxval=0.1))
b = tf.Variable(tf.zeros([picture_size/2]))
y1 = tf.nn.relu(tf.matmul(x, W) + b)

# layer 2
W2 = tf.Variable(tf.random_uniform(shape=[picture_size//2, picture_size//8], minval=-0.1, maxval=0.1))
b2 = tf.Variable(tf.zeros([picture_size//8]))
y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)

# layer 3
W3 = tf.Variable(tf.random_uniform(shape=[picture_size//8, picture_size//32], minval=-0.1, maxval=0.1))
b3 = tf.Variable(tf.zeros([picture_size//32]))
y3 = tf.nn.relu(tf.matmul(y2, W3) + b3)

# layer 4
W4 = tf.Variable(tf.random_uniform(shape=[picture_size//32, 10], minval=-0.1, maxval=0.1))
b4 = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(y3, W4) + b4)

# implement loss func
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# training
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

statictic = {
    'iteration': [],
    'accuracy': [],
}

for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    if i % 100 == 0:
        # get current accuracy
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_val = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

        statictic['iteration'].append(i)
        statictic['accuracy'].append(accuracy_val)

plt.plot(statictic['iteration'], statictic['accuracy'])
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.show()

# get result accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels}))