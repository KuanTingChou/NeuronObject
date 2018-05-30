import tensorflow as tf
import numpy as np

# prepare data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# set parameters
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# loss function
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(loss)

# initialization
init = tf.global_variables_initializer()

# session
sess = tf.Session()
sess.run(init)

# train
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

# close
sess.close()
