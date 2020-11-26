import tensorflow as tf

def edge_cha(x_image):
    out=tf.image.sobel_edges(x_image)
    return out

def intensity_cha(x_image):
    out=tf.image.rgb_to_grayscale(x_image)
    return out

def color_cha(x_image):
    R_filter = tf.constant([[[1, 0, 0]]])
    G_filter = tf.constant([[[0, 1, 0]]])
    B_filter = tf.constant([[[0, 0, 1]]])
    R = tf.nn.conv2d(x_image, R_filter, strides=[1, 1, 1, 1], padding='SAME')
    G = tf.nn.conv2d(x_image, G_filter, strides=[1, 1, 1, 1], padding='SAME')
    B = tf.nn.conv2d(x_image, R_filter, strides=[1, 1, 1, 1], padding='SAME')
    Y=((R+G)/2)-B
    #r-g
    RG=tf.abs(R-G)
    BY=tf.abs(B-Y)
    C_out=tf.add(RG,BY)
    return C_out

