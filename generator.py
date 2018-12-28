import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def train_data(batchsize):
    x = np.random.normal(0, 1, [batchsize, 100])
    return x

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0, 1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def generate_graph(x, batch_size):
    # dense fully connected layer
    w_fc1 = weight_variable([100, 128*16*16])
    b_fc1 = bias_variable([128*16*16])
    fc1 = tf.nn.relu(tf.matmul(x, w_fc1) + b_fc1)

    # upsampling1
    fc1 = tf.reshape(fc1, [-1, 16, 16, 128])
    # ?????
    ups1 = tf.nn.conv2d_transpose(fc1, weight_variable([2, 2, 128, 128]), output_shape=[batch_size, 32, 32, 128], strides=[1,2,2,1],padding="SAME")

    # conv1
    w_conv1 = weight_variable([4, 4, 128, 128])
    b_conv1 = bias_variable([128])
    conv1 = tf.nn.relu(conv2d(ups1, w_conv1) + b_conv1)

    # upsampling2
    ups2 = tf.nn.conv2d_transpose(conv1, weight_variable([2, 2, 128, 128]), output_shape=[batch_size, 64, 64, 128], strides=[1,2,2,1],padding="SAME")

    # conv2
    w_conv2 = weight_variable([4, 4, 128, 64])
    b_conv2 = bias_variable([64])
    conv2 = tf.nn.relu(conv2d(ups2, w_conv2) + b_conv2)

    # conv3
    w_conv3 = weight_variable([4, 4, 64, 3])
    b_conv3 = bias_variable([3])
    conv3 = tf.nn.sigmoid(conv2d(conv2, w_conv3) + b_conv3)

    return conv3

def text():
    if __name__ == "__main__":
        lr = 0.0002
        beta = 0.5
        score = 0.5 # ?disc???score
        #g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=[1.0], logits=[2.0]))
        batch_size = 1

        x = tf.placeholder(tf.float32, [None, 100])
        #train_step= tf.train.AdamOptimizer(learning_rate=lr, beta1=beta).minimize(g_loss)
        generate_graph_output = generate_graph(x, batch_size)
    	
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            generator_image = sess.run(generate_graph_output, feed_dict={x:train_data(batch_size)})
            generator_image = generator_image.reshape([64, 64, 3])
            plt.imshow(generator_image)
            plt.show()




