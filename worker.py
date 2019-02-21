from core.method import *
from setting import *

import pickle

class Worker:
    def __init__(self):
        file_dir = 'dataset/Keyframes'
        file_handler = FileHandler()
        self.vid_dict = file_handler.get_file_dict(file_dir)
        self.total_video = len(self.vid_dict.keys())

    def key_frame(self):
        pass

    def kmeans_learn(self):
        train_kmeans(BATCH_SIZE, CODE_SIZE, CLUSTERS, H, W, self.vid_dict, SAMPLE_SIZE)

    def kmeans_predict(self, kmeans):
        query_videos = []
        kemans_set_source_video(query_videos, kmeans, CLUSTERS, self.total_video, self.vid_dict)

    def run(self):

        f = open('kmeans++_clusters.pkl', 'wb')

        try:
            kmeans = pickle.load(f)
        except Exception:
            self.kmeans_learn()
            kmeans = pickle.load(f)

        self.kmeans_predict(kmeans)


if __name__ =='__main__':
    worker = Worker()
    worker.run()
