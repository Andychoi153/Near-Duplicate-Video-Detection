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
        video_hist = kemans_set_source_video(query_videos, kmeans, CLUSTERS, self.total_video, self.vid_dict)
        normal_hist, idf, inverted_index = normalize_video(CLUSTERS, self.total_video, video_hist)
        result = get_similarity(query_videos, CLUSTERS, kmeans, idf, normal_hist, inverted_index)
        return result

    def run(self):

        f = open('kmeans++_clusters.pkl', 'wb')

        try:
            kmeans = pickle.load(f)
            log.info('model fetch')
        except Exception:
            self.kmeans_learn()
            log.info('train complete')
            kmeans = pickle.load(f)

        result = self.kmeans_predict(kmeans)
        log.info(result)



if __name__ =='__main__':
    worker = Worker()
    worker.run()
