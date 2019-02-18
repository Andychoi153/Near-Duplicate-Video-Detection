import numpy as np
import os, math, cv2, h5py
import tensorflow as tf
import pickle
from collections import defaultdict
from scipy import misc

from random import shuffle
from sklearn.cluster import MiniBatchKMeans

alex_net_path = os.path.join("tf_models/bvlc_alexnet.npy")
alex_net = np.load(alex_net_path, encoding='latin1').item()

vgg_net_path = os.path.join("tf_models/vgg16.npy")
vgg_net = np.load(vgg_net_path, encoding='latin1').item()


def max_pool(input_x, kernel_size, stride, padding='VALID'):
    ksize = [1, kernel_size, kernel_size, 1]
    strides = [1, stride, stride, 1]
    return tf.nn.max_pool(input_x, ksize=ksize, strides=strides, padding=padding)


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


# TODO: 수정
query_videos = set([
    13048,
    13052,
    13053,
    13054,
    13056,
    13057
])

KEYFRAME_DATA_PATH = '/Users/daebokchoi/Near-Duplicate-Video-Detection/dataset/Keyframes/'
KEYFRAME_META_PATH = '/Users/daebokchoi/Near-Duplicate-Video-Detection/dataset/shots.txt'
# KEYFRAME_META_PATH = '/Users/daebokchoi/Near-Duplicate-Video-Detection/dataset/shot_infos.csv'


def get_size():
    # TODO: length 정하는 알고리즘 만들어서 구현할 것!
    return TOTAL_FRAMES,


# TODO: 수정
TOTAL_FRAMES = 27594
TOTAL_VIDEOS = 13057
sample_size = 10000

# TODO: resize sampling 비율로 결정
batch_size = 5000
clusters = 1000
code_size = 100

H, W, D = 227, 227, 3
shot_data = defaultdict(int)
img_data = defaultdict(list)
sequence_data = [0 for i in range(TOTAL_FRAMES)]
video_data = defaultdict(list)
k = 0


with open(KEYFRAME_META_PATH) as f:
    for line in f:
        try:
            serial_id, key_frame, video_id, video_name = line.split('\t')
        except Exception:
            serial_id, key_frame, video_id, video_name = line.split(',')

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


# TODO: code 병신같이 짰네
def get_frame(vid, frame):
    ipath = KEYFRAME_DATA_PATH + str(vid) + frame
    try:
        if frame[-4:] != '.jpg':
            ipath = ipath+'.jpg'

        img = misc.imread(ipath)

    except FileNotFoundError:
        return None
    return img


def get_video_frames(vid, frames, resize = True):
    ipath = KEYFRAME_DATA_PATH + str(vid)
    images = []
    for f in frames:
        try:
            k = ipath + f
            if f[-4:] != '.jpg':
                k = k + '.jpg'
            img = misc.imread(k)
            img = cv2.resize(img, (H, W), interpolation=cv2.INTER_LINEAR)
            images.append(img)
        except FileNotFoundError:
            continue
        except TypeError:
            continue
    return images


#CODEBOOK generation
# generate codebook from 100K sample frames


f = [i for i in range(TOTAL_FRAMES)]

shuffle(f)
code_frames = f[:sample_size]
#code_frames = f[:100000]



f = 10
d = get_frame(sequence_data[f][0], sequence_data[f][1])

# Mini-batch k means to generate visual codebook

rng = np.random.RandomState(0)
kmeans = MiniBatchKMeans(n_clusters=clusters, random_state=rng, init='k-means++', verbose=True)

it = 0

batch_data = []
for f in code_frames:
    value = sequence_data[f]
    try:
        vid = value[0]
        frame = value[1]
        img = get_frame(vid, frame)

    except Exception as e:
        img = None
        print(str(vid)+'/' + str(f))

    if img is None:
        continue
    try:
        img = cv2.resize(img, (H, W), interpolation=cv2.INTER_LINEAR)
    except Exception:
        continue
    # if vid in query_videos:  # DO NOT add query videos
    #     continue

    batch_data.append(img)

    if len(batch_data) == batch_size:
        it += 1
        print("Iteration", it)

        batch_data = np.asarray(batch_data)
        batch_res = []
        # print(data.shape, data[0].shape)
        for k in range(0, len(batch_data) // code_size):
            conv1, conv2, conv3, conv4, conv5 = features_alex_net(batch_data[k * code_size:(k + 1) * code_size], alex_net)
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

pickle.dump(kmeans, open('kmeans++_clusters.pkl', 'wb'))
