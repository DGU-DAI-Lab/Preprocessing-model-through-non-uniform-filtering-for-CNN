import tensorflow as tf

def CSDN(x_image,is_training) :

    c_pyramid0 = tf.nn.conv2d(x_image,56, [3,3],stride=(1,1), padding='SAME')  # 28
    c_pyramid0= tf.nn.relu(tf.layers.batch_normalization(c_pyramid0,training=is_training))
    c_pyramid0 = tf.nn.conv2d(c_pyramid0, 1, [1, 1], stride=(1, 1), padding='SAME')  # 28
    c_pyramid0 = tf.nn.relu(tf.layers.batch_normalization(c_pyramid0, training=is_training))


    c_pyramid1 = tf.layers.conv2d_transpose(x_image, 56, [2, 2], strides=(2, 2), padding='SAME')  # 56
    c_pyramid1=tf.nn.relu(tf.layers.batch_normalization(c_pyramid1, training=is_training))
    c_pyramid1 = tf.nn.conv2d(c_pyramid1, 1, [2, 2], stride=(1, 1), padding='SAME')
    c_pyramid1 = tf.nn.relu(tf.layers.batch_normalization(c_pyramid1, training=is_training))

    c_pyramid2 = tf.layers.conv2d_transpose(x_image, 56, [3, 3], strides=(3, 3), padding='SAME')  # 84
    c_pyramid2 = tf.nn.relu(tf.layers.batch_normalization(c_pyramid2, training=is_training))
    c_pyramid2 = tf.nn.conv2d(c_pyramid2, 1, [3, 3], stride=(1, 1), padding='SAME')
    c_pyramid2 = tf.nn.relu(tf.layers.batch_normalization(c_pyramid2, training=is_training))

    c_pyramid3 = tf.layers.conv2d_transpose(x_image, 56, [4, 4], strides=(4, 4), padding='SAME')  # 112
    c_pyramid3 = tf.nn.relu(tf.layers.batch_normalization(c_pyramid3, training=is_training))
    c_pyramid3 = tf.nn.conv2d(c_pyramid3, 1, [4, 4], stride=(1, 1), padding='SAME')
    c_pyramid3 = tf.nn.relu(tf.layers.batch_normalization(c_pyramid3, training=is_training))

    # print(c_pyramid2.numpy())
    c_pool0 = tf.nn.max_pool(c_pyramid3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 56
    c_pool1 = tf.nn.max_pool(c_pyramid2, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')  # 28
    print(c_pool0)
    c_sub0 = tf.abs((tf.subtract(c_pyramid0, c_pool1)))  # |28-28|
    c_sub1 = tf.abs((tf.subtract(c_pyramid1, c_pool0)))  # |56-56|
    c_sub1 = tf.nn.max_pool(c_sub1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # |28-28|
    c_0 = tf.add(c_sub0, c_sub1)

    return c_0
