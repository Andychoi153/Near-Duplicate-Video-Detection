import numpy as np
import os,math,cv2, h5py
import tensorflow as tf

from log import log

alex_net_path = os.path.join("tf_models/bvlc_alexnet.npy")
alex_net = np.load(alex_net_path, encoding='latin1').item()

vgg_net_path = os.path.join("tf_models/vgg16.npy")
vgg_net = np.load(vgg_net_path, encoding='latin1').item()


#Needed for creating feature descriptors
def max_pool(input_x, kernel_size, stride, padding='VALID'):
    ksize = [1, kernel_size, kernel_size, 1]
    strides = [1, stride, stride, 1]
    return tf.nn.max_pool(input_x, ksize=ksize, strides=strides, padding=padding)


#Here we already have pre-trained weights
def conv_2d(input_x, weights, stride, bias=None, padding='VALID'):
    stride_shape = [1, stride, stride, 1]
    c = tf.nn.conv2d(input_x, weights, stride_shape, padding=padding)
    if bias is not None:
        c += bias
    return c


def imgread(path):
    log.debug("Image:", path.split("/")[-1])
    # Read in the image using python opencv
    img = cv2.imread(path)
    img = img / 255.0
    log.info("Raw Image Shape: ", img.shape)

    # Center crop the image
    short_edge = min(img.shape[:2])
    W, H, C = img.shape
    to_crop = min(W, H)
    cent_w = int((img.shape[1] - short_edge) / 2)
    cent_h = int((img.shape[0] - short_edge) / 2)
    img_cropped = img[cent_h:cent_h + to_crop, cent_w:cent_w + to_crop]
    log.debug("Cropped Image Shape: ", img_cropped.shape)

    # Resize the cropped image to 224 by 224 for VGG16 network
    img_resized = cv2.resize(img_cropped, (224, 224), interpolation=cv2.INTER_LINEAR)
    log.debug("Resized Image Shape: ", img_resized.shape)
    return img_resized


def normalize(ip):
    m2 = np.min(ip)
    ip = ip - m2
    m1 = np.max(ip)
    ip = ip / m1
    return ip


def alex_net_graph(ip, weights, biases):
    w1, w2, w3, w4, w5 = weights
    b1, b2, b3, b4, b5 = biases
    with tf.variable_scope("alex_net"):
        # CONV 1
        c1 = conv_2d(ip, w1, 4, b1, padding='VALID')
        r1 = tf.nn.relu(c1)
        m1 = max_pool(r1, 3, 2, padding='VALID')
        # log.debug("M1", m1.get_shape)

        # CONV2
        m1 = tf.pad(m1, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")  # add 2 padding
        i1, i2 = tf.split(axis=3, num_or_size_splits=2, value=m1)
        w2_1, w2_2 = tf.split(axis=3, num_or_size_splits=2, value=w2)
        o1 = conv_2d(i1, w2_1, 1, bias=None, padding='SAME')
        o2 = conv_2d(i2, w2_2, 1, bias=None, padding='SAME')
        c2 = tf.concat(axis=3, values=[o1, o2])
        r2 = tf.nn.relu(c2)
        m2 = max_pool(r2, 3, 2, padding='VALID')
        # log.debug("M2",m2.get_shape)

        # CONV3
        c3 = conv_2d(m2, w3, 1, b3)
        r3 = tf.nn.relu(c3)
        # log.debug(r3.get_shape, "R3")

        # CONV4
        i1, i2 = tf.split(axis=3, num_or_size_splits=2, value=r3)
        w4_1, w4_2 = tf.split(axis=3, num_or_size_splits=2, value=w4)
        o1 = conv_2d(i1, w4_1, 1, bias=None, padding='SAME')
        o2 = conv_2d(i2, w4_2, 1, bias=None, padding='SAME')
        c4 = tf.concat(axis=3, values=[o1, o2])
        r4 = tf.nn.relu(c4)
        # log.debug(r4.get_shape, "R4")

        # CONV5
        i1, i2 = tf.split(axis=3, num_or_size_splits=2, value=r4)
        w5_1, w5_2 = tf.split(axis=3, num_or_size_splits=2, value=w5)
        o1 = conv_2d(i1, w5_1, 1, bias=None, padding='SAME')
        o2 = conv_2d(i2, w5_2, 1, bias=None, padding='SAME')
        c5 = tf.concat(axis=3, values=[o1, o2])
        r5 = tf.nn.relu(c5)
        m5 = max_pool(r5, 3, 2, padding='VALID')
        # log.debug(m5.get_shape, "M5")

        layers = [m1, m2, r3, r4, m5]
        return layers

        # c1 = conv_2d(ip, w1, 4, b1, padding='VALID')
        # log.debug("C1", c1.get_shape)
        # c2 = conv_2d(c1, w2, 1, b2)


# takes an input image and generates a feature descriptor from the image
def features_alex_net(inputs, alex_net):
    tf.reset_default_graph()
    H, W, D = 227, 227, 3

    w1, b1 = alex_net['conv1'][0], alex_net['conv1'][1]
    w2, b2 = alex_net['conv2'][0], alex_net['conv2'][1]
    w3, b3 = alex_net['conv3'][0], alex_net['conv3'][1]
    w4, b4 = alex_net['conv4'][0], alex_net['conv4'][1]
    w5, b5 = alex_net['conv5'][0], alex_net['conv5'][1]

    weights = [w1, w2, w3, w4, w5]
    biases = [b1, b2, b3, b4, b5]

    # log.debug(w1.shape, w2.shape, w3.shape, w4.shape, w5.shape)

    images = tf.placeholder(tf.float32, [None, H, W, D])
    input_layers = alex_net_graph(images, weights, biases)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        result = sess.run(input_layers, feed_dict={images: inputs})
        return result
