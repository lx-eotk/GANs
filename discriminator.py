import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob

def read_images_path(data_dir):
    #method 1
    fpaths = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)
    return fpaths

    #method 2
    # images_paths = glob.glob(images_dir+'*.jpg')
    # images_paths += glob.glob(images_dir+'*.jpeg')
    # images_paths += glob.glob(images_dir+'*.png')
    # return images_paths

# 使用tf中的函数处理图像
def parse_data(filename):
    # 读取图像
    image = tf.read_file(filename)
    # 解码图片
    image = tf.image.decode_image(image)
    # 数据预处理，或者数据增强，这一步根据需要自由发挥
    # 随机提取patch
    # image = tf.random_crop(image, size=(100,100, 3))
    # 数据增强，随机水平翻转图像
    # image = tf.image.random_flip_left_right(image)
    # 图像归一化
    image = tf.cast(image, tf.float32) / 255.0
    # return image+类别
    return image

#生成器，生成图片的训练batch
def train_real_images(batch_size, shuffle=True):
    data_dir = "../extra_data/images/"
    with tf.Session() as sess:
        # 创建数据库
        train_dataset = tf.data.Dataset().from_tensor_slices((read_images_path(data_dir)))
        # 预处理数据
        train_dataset = train_dataset.map(parse_data)
        # 设置 batch
        train_dataset = train_dataset.batch(batch_size)
        # 无限重复数据
        # train_dataset = train_dataset.repeat()
        # 打乱
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=4)

        # 创建迭代器
        train_iterator = train_dataset.make_initializable_iterator()
        sess.run(train_iterator.initializer)
        train_batch = train_iterator.get_next()

        # 开始生成数据
        while True:
            try:
                x_batch = sess.run(train_batch)
                yield x_batch
            except:
                return False

def train_real_generator(batchsize, shuffle=True):
    data_dir = "../extra_data/images/"
    batchsize = 10
    with tf.Session() as sess:
        # 创建数据库
        train_dataset = tf.data.Dataset().from_tensor_slices(read_images_path(data_dir))
        # 预处理数据
        train_dataset = train_dataset.map(parse_data)
        # 设置 batch size
        train_dataset = train_dataset.batch(batchsize)
        # 无限重复数据
        # train_dataset = train_dataset.repeat()
        # 打乱
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=4)

        # 创建迭代器
        train_iterator = train_dataset.make_initializable_iterator()
        sess.run(train_iterator.initializer)
        train_batch = train_iterator.get_next()

        # 开始生成数据
        while True:
            try:
                x_batch, y_batch = sess.run(train_batch)
                yield (x_batch, y_batch)
            except:
                return False
                # 如果没有  train_dataset = train_dataset.repeat()
                # 数据遍历完就到end了，就会抛出异常
                # train_iterator = train_dataset.make_initializable_iterator()
                # sess.run(train_iterator.initializer)
                # train_batch = train_iterator.get_next()
                # x_batch, y_batch = sess.run(train_batch)
                # yield (x_batch, y_batch)

#Weight Initialization
def weight_variable(shape, name='www'):
    return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=0.02))
    # initial = tf.truncated_normal(shape, stddev=0.1)
    # return tf.Variable(initial)

def bias_variable(shape, name='xxx'):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0))
    # initial = tf.constant(0.1, shape=shape)
    # return tf.Variable(initial)

#Convolution and Pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def discriminator_graph(x, reuse=False):
    x_image = tf.reshape(x, [-1, 64, 64, 3])

    #first convolution layer
    w_conv1 = weight_variable([4, 4, 3, 32], 'w_conv1')
    b_conv1 = bias_variable([32], 'b_conv1')
    conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)

    #second
    w_conv2 = weight_variable([4, 4, 32, 64], 'w_conv2')
    b_conv2 = bias_variable([64], 'b_conv2')
    conv2 = tf.nn.relu(conv2d(conv1, w_conv2) + b_conv2)

    #third
    w_conv3 = weight_variable([4, 4, 64, 128])
    b_conv3 = bias_variable([128])
    conv3 = tf.nn.relu(conv2d(conv2, w_conv3) + b_conv3)

    #forth
    w_conv4 = weight_variable([4, 4, 128, 256])
    b_conv4 = bias_variable([256])
    conv4 = tf.nn.relu(conv2d(conv3, w_conv4) + b_conv4)

    #fully connected layer
    w_fc1 = weight_variable([64*64*256, 1])
    b_fc1 = bias_variable([1])

    conv4_flat = tf.reshape(conv4, [-1, 64*64*256])
    fc1 = tf.nn.softmax(tf.matmul(conv4_flat, w_fc1) + b_fc1)
    return fc1
    #dropout 可以防止过拟合
    #keep_prob = tf.placeholder(tf.float32)
    #h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

def test():
    if __name__ == "__main__":
        #train and evaluate
        lr = 10e-3
        STEPS = 100
        batchsize = 10
        model_path = './save'
        x = tf.placeholder(tf.float32,[None, 12288])
        y_ = tf.placeholder(tf.float32, [None, 1])

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=fc1)
        train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
        accuracy = 1 - tf.reduce_mean(tf.abs(y_ - fc1))

        # 用于保存和载入模型
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_feed_dict = {
                x:1
            }
            for step in range(STEPS):
                _, loss_val = sess.run([train_step, loss], feed_dict=train_feed_dict)
                if step % 10 == 0:
                    print("step = {}\tloss = {}".format(step, loss_val))
            saver.save(sess, model_path)
            print("训练结束，保存模型到{}".format(model_path))








#多类别
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
#train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
#accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1)), tf.float32))

#tf.cast()
#cast(x, dtype, name=None)
#将x的数据格式转化成dtype.例如，原来x的数据格式是bool， 
#那么将其转化成float以后，就能够将其转化成0和1的序列
#tf.equal()
# a = [[1,2,3],[4,5,6]]
# b = [[1,0,3],[1,5,1]]
# with tf.Session() as sess:
#     print(sess.run(tf.equal(a,b)))
# 结果：
# [[ True False  True]
#  [False  True False]]



'''
v1=tf.Variable(tf.random_normal(shape=[4,3],mean=0,stddev=1),name='v1')
v2=tf.Variable(tf.constant(2),name='v2')
v3=tf.Variable(tf.ones([4,3]),name='v3')

print(v1,'\n', v2, '\n', v3)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(v1))
    print(sess.run(v2))
    print(sess.run(v3))
'''









#tf.truncated_normal()
#从截断的正态分布中输出随机值。 
#生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值
#2个标准偏差的值则丢弃重新选择

#tf.random_normal()
#从正态分布中输出随机值

# tf.fill(dims, value, name=None)
# tf.zeros(shape, dtype=tf.float32, name=None)
# tf.zeros_like(tensor, dtype=None, name=None)
# tf.ones(shape, dtype=tf.float32, name=None)
# tf.ones_like(tensor, dtype=None, name=None)

#tf.constant(value,dtype=None,shape=None,name=’Const’) 
#创建一个常量tensor，按照给出value来赋值。value可以是一个数，也可以是一个list。
#list,那么len(value)一定要小于等于shape展开后的长度。
#赋值时，先将value中的值逐个存入。不够的部分，则全部存入value的最后一个值。

#tf.Variable()
#eg:tf.Variable(tf.random_normal(shape=[4,3],mean=0,stddev=1),name='v1')

'''
方法
tf.layers 模块提供的方法有：

Input(…): 用于实例化一个输入 Tensor，作为神经网络的输入。
average_pooling1d(…): 一维平均池化层
average_pooling2d(…): 二维平均池化层
average_pooling3d(…): 三维平均池化层
batch_normalization(…): 批量标准化层
conv1d(…): 一维卷积层
conv2d(…): 二维卷积层
conv2d_transpose(…): 二维反卷积层
conv3d(…): 三维卷积层
conv3d_transpose(…): 三维反卷积层
dense(…): 全连接层
dropout(…): Dropout层
flatten(…): Flatten层，即把一个 Tensor 展平
max_pooling1d(…): 一维最大池化层
max_pooling2d(…): 二维最大池化层
max_pooling3d(…): 三维最大池化层
separable_conv2d(…): 二维深度可分离卷积层
'''

'''
下面是对三个模块的简述：

tf.nn ：提供神经网络相关操作的支持，包括卷积操作（conv）、池化操作（pooling）、归一化、loss、分类操作、embedding、RNN、Evaluation。
tf.layers：主要提供的高层的神经网络，主要和卷积相关的，tf.nn会更底层一些。
tf.contrib：tf.contrib.layers提供够将计算图中的 网络层、正则化、摘要操作、是构建计算图的高级操作，但是tf.contrib包含不稳定和实验代码，有可能以后API会改变。

Activation Functions 
tf.nn.relu(features, name=None)
tf.nn.relu6(features, name=None)
tf.nn.softplus(features, name=None)
tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)
tf.nn.bias_add(value, bias, name=None)
tf.sigmoid(x, name=None)
tf.tanh(x, name=None)
Convolution 
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
tf.nn.depthwise_conv2d(input, filter, strides, padding, name=None)
tf.nn.separable_conv2d(input, depthwise_filter, pointwise_filter, strides, padding, name=None)
Pooling 
tf.nn.avg_pool(value, ksize, strides, padding, name=None)
tf.nn.max_pool(value, ksize, strides, padding, name=None)
tf.nn.max_pool_with_argmax(input, ksize, strides, padding, Targmax=None, name=None)
Normalization 
tf.nn.l2_normalize(x, dim, epsilon=1e-12, name=None)
tf.nn.local_response_normalization(input, depth_radius=None, bias=None, alpha=None, beta=None, name=None)
tf.nn.moments(x, axes, name=None)
Losses 
tf.nn.l2_loss(t, name=None)
Classification 
tf.nn.sigmoid_cross_entropy_with_logits(logits, targets, name=None)
tf.nn.softmax(logits, name=None)
tf.nn.softmax_cross_entropy_with_logits(logits, labels, name=None)
Embeddings 
tf.nn.embedding_lookup(params, ids, name=None)
Evaluation

tf.nn.top_k(input, k, name=None)
tf.nn.in_top_k(predictions, targets, k, name=None)
Candidate Sampling（包含Sampled Loss Functions和Miscellaneous candidate sampling utilities）

Sampled Loss Functions

tf.nn.nce_loss(weights, biases, inputs, labels, num_sampled, num_classes, num_true=1, sampled_values=None, remove_accidental_hits=False, name=’nce_loss’)
tf.nn.sampled_softmax_loss(weights, biases, inputs, labels, num_sampled, num_classes, num_true=1, sampled_values=None, remove_accidental_hits=True, name=’sampled_softmax_loss’)
Candidate Samplers 
tf.nn.uniform_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, seed=None, name=None)
tf.nn.log_uniform_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, seed=None, name=None)
tf.nn.learned_unigram_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, seed=None, name=None)
tf.nn.fixed_unigram_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, vocab_file=”, distortion=0.0, num_reserved_ids=0, num_shards=1, shard=0, unigrams=[], seed=None, name=None)
Miscellaneous candidate sampling utilities 
tf.nn.compute_accidental_hits(true_classes, sampled_candidates, num_true, seed=None, name=None)
--------------------- 
作者：呆呆的猫 
来源：CSDN 
原文：https://blog.csdn.net/jiaoyangwm/article/details/79247371
'''

'''
1. 优化器（optimizer）
优化器的基类（Optimizer base class）主要实现了两个接口，一是计算损失函数的梯度，二是将梯度作用于变量。tf.train 主要提供了如下的优化函数：

tf.train.Optimizer
tf.train.GradientDescentOptimizer
tf.train.AdadeltaOpzimizer 
Ada delta
tf.train.AdagradDAOptimizer
tf.train.MomentumOptimizer
tf.train.AdamOptimizer
tf.train.FtrlOptimizer
tf.train.ProximalGradientDescentOptimizer
tf.train.ProximalAdagradOptimizer
tf.train.RMSPropOptimizer
2. 梯度计算
TensorFlow 同时也提供了给定 TensorFlow 计算图（computation graph）的导数。上节提到的优化器类（optimizer classes）会自动计算 computation graph 的导数，但用户自定义优化器时，可以使用如下低级别的函数：

tf.gradients
tf.AggregationMethod
tf.stop_gradient
tf.hessians
2. 学习率衰减（decaying the learning rate）
tf.train.exponential_decay


# 实现的是如下的操作

decayed_lr = lr * decay_rate ^ (global_step/decay_steps)
1
2
3
4
在其 tf 下的使用为：
1
lr = tf.train.exponential_decay(0.1, global_step, 100, .96, staircase=True)
1
tf.train.inverse_time_decay

tf.train.natural_exp_decay
tf.train.piecewise_constant
tf.train.polynomial_decay
--------------------- 
作者：Inside_Zhang 
来源：CSDN 
原文：https://blog.csdn.net/lanchunhui/article/details/61414425 
版权声明：本文为博主原创文章，转载请附上博文链接！
'''
