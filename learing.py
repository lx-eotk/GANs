import tensorflow as tf 
import numpy as np 

def train_data(batchsize):
    x = np.random.normal(0, 1, [batchsize, 100])
    return x

if __name__ == "__main__":
    noise_place = tf.placeholder(tf.float32, [None, 100])
    with tf.Session() as sess:
        print(noise_place.shape)
        result = sess.run(noise_place, feed_dict={noise_place:train_data(10)})
        print(tf.reshape(noise_place, [-1, 100]).shape)

