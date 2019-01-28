import numpy as np
import os,math,cv2, h5py
import tensorflow as tf
from collections import defaultdict
from scipy import misc

from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg
from random import shuffle
from sklearn.cluster import MiniBatchKMeans

alex_net_path = os.path.join("tf_models/bvlc_alexnet.npy")
alex_net = np.load(alex_net_path, encoding='latin1').item()

vgg_net_path = os.path.join("tf_models/vgg16.npy")
vgg_net = np.load(vgg_net_path, encoding='latin1').item()


a_c1 = alex_net['conv1']
w1 = a_c1[0]
b1 = a_c1[1]

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
    print("Image:", path.split("/")[-1])
    # Read in the image using python opencv
    img = cv2.imread(path)
    img = img / 255.0
    print("Raw Image Shape: ", img.shape)

    # Center crop the image
    short_edge = min(img.shape[:2])
    W, H, C = img.shape
    to_crop = min(W, H)
    cent_w = int((img.shape[1] - short_edge) / 2)
    cent_h = int((img.shape[0] - short_edge) / 2)
    img_cropped = img[cent_h:cent_h + to_crop, cent_w:cent_w + to_crop]
    print("Cropped Image Shape: ", img_cropped.shape)

    # Resize the cropped image to 224 by 224 for VGG16 network
    img_resized = cv2.resize(img_cropped, (224, 224), interpolation=cv2.INTER_LINEAR)
    print("Resized Image Shape: ", img_resized.shape)
    return img_resized


#center crop and resize
image_path = os.path.join("images/corgi.jpg")
img1 = imgread(image_path)

#basic cv resize
img2 = cv2.imread(image_path)
img_resized = cv2.resize(img2, (224, 224), interpolation=cv2.INTER_LINEAR)

#Testing Cell block
#For generating single descriptor from each convolution layer

tf.reset_default_graph()
images = tf.placeholder(tf.float32, [None, 224, 224, 3])
w = tf.get_variable('we', initializer=tf.to_float(w1))
print(images.get_shape(), w.get_shape())

c1 = conv_2d(images, w, 4, b1)
img1 = img1.reshape(1, 224, 224, 3) # convert to 4D tensor

#Testing Cell block
#tf.reset_default_graph()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # Usage passing the session explicitly.
    #print(op.eval(sess))
    #
    conv_op = sess.run([c1], feed_dict={images: img1})
    print(conv_op[0].shape)


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
        # print("M1", m1.get_shape)

        # CONV2
        m1 = tf.pad(m1, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")  # add 2 padding
        i1, i2 = tf.split(axis=3, num_or_size_splits=2, value=m1)
        w2_1, w2_2 = tf.split(axis=3, num_or_size_splits=2, value=w2)
        o1 = conv_2d(i1, w2_1, 1, bias=None, padding='SAME')
        o2 = conv_2d(i2, w2_2, 1, bias=None, padding='SAME')
        c2 = tf.concat(axis=3, values=[o1, o2])
        r2 = tf.nn.relu(c2)
        m2 = max_pool(r2, 3, 2, padding='VALID')
        # print("M2",m2.get_shape)

        # CONV3
        c3 = conv_2d(m2, w3, 1, b3)
        r3 = tf.nn.relu(c3)
        # print(r3.get_shape, "R3")

        # CONV4
        i1, i2 = tf.split(axis=3, num_or_size_splits=2, value=r3)
        w4_1, w4_2 = tf.split(axis=3, num_or_size_splits=2, value=w4)
        o1 = conv_2d(i1, w4_1, 1, bias=None, padding='SAME')
        o2 = conv_2d(i2, w4_2, 1, bias=None, padding='SAME')
        c4 = tf.concat(axis=3, values=[o1, o2])
        r4 = tf.nn.relu(c4)
        # print(r4.get_shape, "R4")

        # CONV5
        i1, i2 = tf.split(axis=3, num_or_size_splits=2, value=r4)
        w5_1, w5_2 = tf.split(axis=3, num_or_size_splits=2, value=w5)
        o1 = conv_2d(i1, w5_1, 1, bias=None, padding='SAME')
        o2 = conv_2d(i2, w5_2, 1, bias=None, padding='SAME')
        c5 = tf.concat(axis=3, values=[o1, o2])
        r5 = tf.nn.relu(c5)
        m5 = max_pool(r5, 3, 2, padding='VALID')
        # print(m5.get_shape, "M5")

        layers = [m1, m2, r3, r4, m5]
        return layers

        # c1 = conv_2d(ip, w1, 4, b1, padding='VALID')
        # print("C1", c1.get_shape)
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

    # print(w1.shape, w2.shape, w3.shape, w4.shape, w5.shape)

    images = tf.placeholder(tf.float32, [None, H, W, D])
    input_layers = alex_net_graph(images, weights, biases)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        result = sess.run(input_layers, feed_dict={images: inputs})
        return result


#testing cell block
H,W,D = 227, 227, 3
i1 = cv2.imread(os.path.join("images/corgi.jpg"))
i1 = cv2.resize(i1, (H, W), interpolation=cv2.INTER_LINEAR)
i2 = cv2.imread(os.path.join("images/dog.png"))
i2 = cv2.resize(i2, (H, W), interpolation=cv2.INTER_LINEAR)
i3 = cv2.imread(os.path.join("/home/chinmay/CODE/gumgum/NDVD/dataset/sample.jpg"))
i3 = cv2.resize(i3, (H, W), interpolation=cv2.INTER_LINEAR)
images = np.asarray([i1,i2,i3])

conv1, conv2, conv3, conv4, conv5 = features_alex_net(images, alex_net)


plt.imshow(i3)
plt.show()


#apply max pooling per channel
# m1.. m5 are layer level image feature descriptors
m1 = np.amax(conv1, axis=(1,2))
m2 = np.amax(conv2, axis=(1,2))
m3 = np.amax(conv3, axis=(1,2))
m4 = np.amax(conv4, axis=(1,2))
m5 = np.amax(conv5, axis=(1,2))


# as mentioned in paper, we get a vector of 1376
print(m1.shape, m2.shape)
r = np.concatenate((m1,m2,m3,m4,m5), axis=1)
print(r.shape)

##################### example end ###############


query_videos = set([1, 815, 1412, 1849, 2200, 2604, 3387, 3752, 4304, 4543, 4849, 5229, 6125, 6545, 6653, 8449, 8659, 9310, 9813,10382, 10580, 11047, 11466, 12818])

KEYFRAME_DATA_PATH = '/home/chinmay/CODE/gumgum/NDVD/repo/Near-Duplicate-Video-Detection/dataset/key_frames/'
KEYFRAME_META_PATH = './dataset/Shot_Info.txt'
TOTAL_FRAMES = 398008
TOTAL_VIDEOS = 13129
sample_size = 100000

H, W, D = 227, 227, 3
shot_data = defaultdict(int)
img_data = defaultdict(list)
sequence_data = [0 for i in range(TOTAL_FRAMES)]
video_data = defaultdict(list)
k = 0



with open(KEYFRAME_META_PATH) as f:
    for line in f:
        serial_id, key_frame, video_id, video_name = line.split('\t')
        video_id = int(video_id)
        if len(img_data[video_id]) == 0:
            img_data[video_id] = [0]
        else:
            img_data[video_id].append(0)

        if len(video_data[video_id]) == 0:
            video_data[video_id] = [key_frame]
        else:
            video_data[video_id].append(key_frame)

        shot_data[video_id] += 1
        sequence_data[k] = [video_id, key_frame]
        k += 1



video_data[1]
print(video_data[2])
print("OK")


def get_keyframes(path):
    for subdir, pdir, files in os.walk(path):
        for fname in files:
            filepath = os.path.join(os.sep, subdir, fname)
            if filepath.endswith('.jpg'):
                name,extension = os.path.splitext(fname)
                vid, seq, shot = name.split('_')
                vid, seq  = int(vid), int(seq)
                yield vid, seq, misc.imread(filepath)


def get_frame(vid, frame):
    ipath = KEYFRAME_DATA_PATH + str(vid) + "/" + frame + ".jpg"
    try:
        img = misc.imread(ipath)
    except FileNotFoundError:
        return None
    return img


def get_video_frames(vid, frames, resize = True):
    ipath = KEYFRAME_DATA_PATH + str(vid) + "/"
    images = []
    for f in frames:
        try:
            k = ipath + f + ".jpg"
            img = misc.imread(k)
            img = cv2.resize(img, (H, W), interpolation=cv2.INTER_LINEAR)
            images.append(img)
        except FileNotFoundError:
            continue
    return images


#CODEBOOK generation
# generate codebook from 100K sample frames


f = [i for i in range(TOTAL_FRAMES)]
shuffle(f)
code_frames = f[:sample_size]
#code_frames = f[:100000]
batch_size = 5000
clusters = 1000
code_size = 100

code_frames[10]
f = 10
d = get_frame(sequence_data[f][0], sequence_data[f][1])
d.shape

# Mini-batch k means to generate visual codebook

rng = np.random.RandomState(0)
kmeans = MiniBatchKMeans(n_clusters=clusters, random_state=rng, init='k-means++', verbose=True)

it = 0

batch_data = []
for f in code_frames:
    vid, frame = sequence_data[f][0], sequence_data[f][1]
    img = get_frame(vid, frame)
    if img is None:
        continue
    img = cv2.resize(img, (H, W), interpolation=cv2.INTER_LINEAR)
    if vid in query_videos:  # DO NOT add query videos
        continue

    batch_data.append(img)

    if len(batch_data) == batch_size:
        it += 1
        print("Iteration", it)

        '''
        batch_data = np.asarray(batch_data)
        #print(data.shape, data[0].shape)

        conv1, conv2, conv3, conv4, conv5 = features_alex_net(batch_data, alex_net)
        #Apply max pooling
        m1 = np.amax(conv1, axis=(1,2))
        m2 = np.amax(conv2, axis=(1,2))
        m3 = np.amax(conv3, axis=(1,2))
        m4 = np.amax(conv4, axis=(1,2))
        m5 = np.amax(conv5, axis=(1,2))

        r = np.concatenate((m1,m2,m3,m4,m5), axis=1)
        print(type(r[0][0]), r.shape)
        #zero-mean and unit normalize

        kmeans.partial_fit(r)

        batch_data = []
        '''

        batch_data = np.asarray(batch_data)
        batch_res = []
        # print(data.shape, data[0].shape)
        for k in range(0, len(batch_data) // code_size):
            conv1, conv2, conv3, conv4, conv5 = features_alex_net(batch_data[k * code_size:(k + 1) * code_size],
                                                                  alex_net)
            # Apply max pooling
            m1 = np.amax(conv1, axis=(1, 2))
            m2 = np.amax(conv2, axis=(1, 2))
            m3 = np.amax(conv3, axis=(1, 2))
            m4 = np.amax(conv4, axis=(1, 2))
            m5 = np.amax(conv5, axis=(1, 2))

            r = np.concatenate((m1, m2, m3, m4, m5), axis=1)

            # zero-mean and unit normalize

            if len(batch_res) == 0:
                batch_res = r
            else:
                batch_res = np.concatenate((batch_res, r), axis=0)
                # print(type(r), batch_res.shape, "IN", r.shape)

        # print(type(r[0][0]), batch_res.shape, "OUT")
        kmeans.partial_fit(batch_res)
        batch_data, batch_res = [], []


import pickle
pickle.dump(kmeans, open('kmeans++_clusters.pkl', 'wb'))

# Generate video level histograms
video_hist = [[0 for i in range(clusters)] for i in range(TOTAL_VIDEOS)]

dataset = range(1, TOTAL_VIDEOS + 1)
# dataset = range(1,10+1)
batch_data, vid_data, seq_data = [], [], []
it = 0
for d in dataset:
    vid = d
    frames = video_data[d]
    if vid in query_videos:  # DO NOT add query videos
        continue

    images = get_video_frames(vid, frames)
    it += 1
    if it % 100 == 0:
        print("Iteration", it)
    batch_data = np.asarray(images)
    # print(len(batch_data), vid)

    if len(batch_data) == 0:
        continue
    batch_res = []

    conv1, conv2, conv3, conv4, conv5 = features_alex_net(batch_data, alex_net)
    # Apply max pooling
    m1 = np.amax(conv1, axis=(1, 2))
    m2 = np.amax(conv2, axis=(1, 2))
    m3 = np.amax(conv3, axis=(1, 2))
    m4 = np.amax(conv4, axis=(1, 2))
    m5 = np.amax(conv5, axis=(1, 2))

    r = np.concatenate((m1, m2, m3, m4, m5), axis=1)

    # zero-mean and unit normalize
    kmeans.verbose = False
    y = kmeans.predict(r)
    for nearest_neighbour in y:
        video_hist[vid][nearest_neighbour] += 1
    # print(video_hist[vid])
    batch_data, r = [], []


pickle.dump(video_hist, open('video_hist.pkl', 'wb'))

# calculate tf-idf weights
# calculate inverted file index for words

import math

inverse_word_count = defaultdict(int)
total_word_count = 0
idf = [0 for i in range(clusters)]
inverted_index = {}

normal_vhist = [[]]
for i in range(1, TOTAL_VIDEOS):
    for index, val in enumerate(video_hist[i]):
        if val > 0:
            inverse_word_count[index] += val
            total_word_count += val

            s = inverted_index.get(index)
            if s is None:
                inverted_index[index] = set([i])
            else:
                inverted_index[index].add(i)

    k = np.array(video_hist[i], dtype=np.float32)
    # normalize TF
    x = sum(k)
    if x > 0:
        k = k / sum(k)
    normal_vhist.append(k)

# IDF weights
for i, v in inverse_word_count.items():
    if v > 0:
        idf[i] = 1 + math.log(total_word_count / v)
    else:
        idf[i] = 1


idf[1]


# Cosine Similarity(Query,Document1) = Dot product(Query, Document1) / ||Query|| * ||Document1||
def cos_similarity(query, doc):
    num = np.dot(query, doc)
    denom = np.sqrt(np.sum(np.square(query))) * np.sqrt(np.sum(np.square(doc)))
    return num / denom
# TODO: 코드 수정!

result = {}
# for all query videos, run this algorithm
for q in query_videos:
    vid = q
    frames = video_data[q]
    images = get_video_frames(vid, frames)
    batch_data = np.asarray(images)

    conv1, conv2, conv3, conv4, conv5 = features_alex_net(batch_data, alex_net)
    # Apply max pooling
    m1 = np.amax(conv1, axis=(1, 2))
    m2 = np.amax(conv2, axis=(1, 2))
    m3 = np.amax(conv3, axis=(1, 2))
    m4 = np.amax(conv4, axis=(1, 2))
    m5 = np.amax(conv5, axis=(1, 2))

    r = np.concatenate((m1, m2, m3, m4, m5), axis=1)
    kmeans.verbose = False
    # predict nearest neighbor
    y = kmeans.predict(r)
    query_tf = [0 for i in range(clusters)]
    test_vids = set()
    for nearest_neighbour in y:
        query_tf[nearest_neighbour] += 1
        test_vids = test_vids | inverted_index[nearest_neighbour]

    # print(q)
    # print(test_vids)

    # normalise tf
    query_tf = np.array(query_tf, dtype=np.float32)
    query_tf = query_tf / sum(query_tf)
    query_tf_idf = np.multiply(query_tf, idf)
    # print(query_tf_idf)

    # break
    for test in test_vids:
        doc_tf_idf = np.multiply(normal_vhist[test], idf)
        cos_sim = cos_similarity(query_tf_idf, doc_tf_idf)
        if q in result:
            result[q][test] = cos_sim
        else:
            result[q] = {test: cos_sim}
    # print(q)
    # print(result[q])
    # break
