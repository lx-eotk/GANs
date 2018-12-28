import generator
import discriminator
import tensorflow as tf

if __name__ == '__main__':
    # loss etc. of discriminator
    d_lr = 0.002
    d_batch_size = 10
    noise_place = tf.placeholder(tf.float32, [None, 100])
    real_images_place = tf.placeholder(tf.float32, [None, 64, 64, 3])
    fake_images = generator.generate_graph(noise_place, d_batch_size)
    real_predicts = discriminator.discriminator_graph(real_images_place)
    fake_predicts = discriminator.discriminator_graph(fake_images)

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_predicts, labels=tf.ones_like(real_predicts)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_predicts, labels=tf.zeros_like(fake_predicts)))
    d_loss = d_loss_fake + d_loss_real

    d_trainer = tf.train.AdamOptimizer(d_lr).minimize(d_loss)

    tvars = tf.trainable_variables()
    print(tvars)
    # loss etc. of generator

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # train disc
        for i in range (300):
            tmp = discriminator.train_real_images(d_batch_size)
            real_images = next(tmp)
            noise = generator.train_data(d_batch_size)

            sess.run(d_trainer, feed_dict={noise_place:noise, real_images_place:real_images})
            print("d_loss: ", sess.run(d_loss, feed_dict={noise_place:noise, real_images_place:real_images}))


        


        # train generator