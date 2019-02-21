import numpy as np
import os, cv2
from collections import defaultdict
from scipy import misc

from random import shuffle
from sklearn.cluster import MiniBatchKMeans
from log import log
import pickle
import math

from .model import features_alex_net, alex_net


class FileHandler:

    def get_file_dict(self, file_path):
        file_names = os.listdir(file_path)
        vids = []
        key_frames = []

        for i in file_names:
            vid = i.split('_')[0]
            vids.append(vid)

        vids = list(set(vids))
        vid_dict = {k: key_frames for k in vids}

        for file_name in file_names:
            vid = file_name.split('_')[0]
            temp = vid_dict.get(vid)
            temp.append(file_name)
            vid_dict.update({vid: temp})
        return vid_dict


def get_video_frames(vid, vid_dict):
    images = []
    key_frames = vid_dict.get(vid)
    for i in key_frames:
        img = misc.imread(i)
        images.append(img)
    return images


def train_kmeans(batch_size, code_size, clusters, H, W, vid_dict, sample_size):
    from itertools import chain
    from random import shuffle

    rng = np.random.RandomState(0)
    it = 0

    video_filenames = list(chain(*vid_dict.values()))
    shuffle(video_filenames)

    code_frames = video_filenames[:sample_size]

    kmeans = MiniBatchKMeans(n_clusters=clusters, random_state=rng, init='k-means++', verbose=True)
    batch_data = []

    for directory in code_frames:

        try:
            img = misc.imread(directory)
            img = cv2.resize(img, (H, W), interpolation=cv2.INTER_LINEAR)
        except Exception:
            log.error('file name: {directory}'.format(directory=directory))

        batch_data.append(img)

        if len(batch_data) == batch_size:
            it += 1
            log.debug("Iteration", it)

            batch_data = np.asarray(batch_data)
            batch_res = []

            for k in range(0, len(batch_data) // code_size):
                conv1, conv2, conv3, conv4, conv5 = features_alex_net(batch_data[k * code_size:(k + 1) * code_size], alex_net)
                # Apply max pooling
                m1 = np.amax(conv1, axis=(1, 2))
                m2 = np.amax(conv2, axis=(1, 2))
                m3 = np.amax(conv3, axis=(1, 2))
                m4 = np.amax(conv4, axis=(1, 2))
                m5 = np.amax(conv5, axis=(1, 2))

                r = np.concatenate((m1, m2, m3, m4, m5), axis=1)

                if len(batch_res) == 0:
                    batch_res = r
                else:
                    batch_res = np.concatenate((batch_res, r), axis=0)

            log.debug(type(r[0][0]), batch_res.shape, "OUT")
            kmeans.partial_fit(batch_res)
            batch_data, batch_res = [], []

    pickle.dump(kmeans, open('kmeans++_clusters.pkl', 'wb'))


# kmeans 처리된 데이터 떨구기
def kemans_set_source_video(query_videos, kmeans, clusters, total_videos, video_dict):
    dataset = video_dict.keys()
    it = 0
    video_hist = [[0 for i in range(clusters)] for i in range(total_videos)]

    for d in dataset:
        if d in query_videos:  # DO NOT add query videos
            continue

        images = get_video_frames(d, video_dict)
        it += 1
        if it % 100 == 0:
            log.debug("Iteration", it)
        batch_data = np.asarray(images)

        if len(batch_data) == 0:
            continue

        conv1, conv2, conv3, conv4, conv5 = features_alex_net(batch_data, alex_net)

        m1 = np.amax(conv1, axis=(1, 2))
        m2 = np.amax(conv2, axis=(1, 2))
        m3 = np.amax(conv3, axis=(1, 2))
        m4 = np.amax(conv4, axis=(1, 2))
        m5 = np.amax(conv5, axis=(1, 2))

        r = np.concatenate((m1, m2, m3, m4, m5), axis=1)

        kmeans.verbose = True
        y = kmeans.predict(r)
        for nearest_neighbour in y:
            video_hist[d][nearest_neighbour] += 1

    return video_hist


def normalize_video(clusters, video_num, video_hist):
    inverse_word_count = defaultdict(int)
    total_word_count = 0
    idf = [0 for i in range(clusters)]
    inverted_index = {}
    normal_vhist = [[]]
    for i in range(0, video_num):
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
    return normal_vhist, idf, inverted_index


# Cosine Similarity(Query,Document1) = Dot product(Query, Document1) / ||Query|| * ||Document1||
def cos_similarity(query, doc):
    num = np.dot(query, doc)
    denom = np.sqrt(np.sum(np.square(query))) * np.sqrt(np.sum(np.square(doc)))
    return num / denom


def get_similarity(query_videos, clusters, kmeans, idf, normal_vhist, inverted_index):
    result = {}
    for q in query_videos:

        images = get_video_frames(q)
        if images is None:
            continue
        batch_data = np.asarray(images)

        conv1, conv2, conv3, conv4, conv5 = features_alex_net(batch_data, alex_net)
        # Apply max pooling
        m1 = np.amax(conv1, axis=(1, 2))
        m2 = np.amax(conv2, axis=(1, 2))
        m3 = np.amax(conv3, axis=(1, 2))
        m4 = np.amax(conv4, axis=(1, 2))
        m5 = np.amax(conv5, axis=(1, 2))

        r = np.concatenate((m1, m2, m3, m4, m5), axis=1)
        kmeans.verbose = True
        # predict nearest neighbor
        y = kmeans.predict(r)
        query_tf = [0 for i in range(clusters)]
        test_vids = set()
        for nearest_neighbour in y:
            query_tf[nearest_neighbour] += 1
            try:
                test_vids = test_vids | inverted_index[nearest_neighbour]  # set 에서의 or 연산은 더하기임!
            except Exception as e:
                log.debug(e)
                log.debug(nearest_neighbour)
                pass

        # normalise tf
        query_tf = np.array(query_tf, dtype=np.float32)
        query_tf = query_tf / sum(query_tf)
        query_tf_idf = np.multiply(query_tf, idf)
        # log.debug(query_tf_idf)

        # break
        for test in test_vids:
            doc_tf_idf = np.multiply(normal_vhist[test], idf)
            cos_sim = cos_similarity(query_tf_idf, doc_tf_idf)
            if q in result:
                result[q][test] = cos_sim
            else:
                result[q] = {test: cos_sim}

    return result
