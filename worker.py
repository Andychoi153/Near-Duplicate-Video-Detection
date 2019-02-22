from core.method import *
from setting import *

import core.extract_key_frames
import pickle
import pandas as pd
from matplotlib import pyplot as plt

class Worker:
    def __init__(self):
        file_dir = 'dataset/Keyframes'
        file_handler = FileHandler()
        self.vid_dict = file_handler.get_file_dict(file_dir)
        self.total_video = max(list(map(int, self.vid_dict.keys()))) + 1

    def key_frame(self):
        core.extract_key_frames.kf_main()

    def kmeans_learn(self):
        train_kmeans(BATCH_SIZE, CODE_SIZE, CLUSTERS, H, W, self.vid_dict, SAMPLE_SIZE)

    def kmeans_predict(self, kmeans):
        query_videos = [# Ipman query
                        13501, # parody
                        13502, # image insert pirate
                        13503, # camera record pirate

                        # Dragon query
                        13505, # image insert pirate
                        13506] # camera record pirate
        try:
            video_hist = pickle.load(open('video_hist.pkl', 'rb'))

        except Exception:

            video_hist = kemans_set_source_video(query_videos, kmeans, CLUSTERS, self.total_video, self.vid_dict, H, W)
            pickle.dump(video_hist, open('video_hist.pkl', 'wb'))

        normal_hist, idf, inverted_index = normalize_video(CLUSTERS, self.total_video, video_hist)
        result = get_similarity(query_videos, self.vid_dict, CLUSTERS, kmeans, idf, normal_hist, inverted_index, H, W)
        return result

    def run(self):
        try:
            kmeans = pickle.load(open('kmeans++_clusters.pkl', 'rb'))
            log.debug('model fetch')
        except Exception :
            log.debug('no model, train start')
            self.kmeans_learn()
            log.debug('train complete')
            kmeans = pickle.load(open('kmeans++_clusters.pkl', 'rb'))

        result = self.kmeans_predict(kmeans)
        pd_result = pd.DataFrame.from_dict(result)
        fig, axs = plt.subplots(5, 1)
        arr = pd_result.as_matrix()
        arr = np.transpose(arr)
        x = [0, 500]
        y = [0.3, 0.3]
        plt.rcParams.update({'font.size': 5})

        axs[0].plot(arr[0])
        axs[0].plot(x, y, color='red')
        axs[0].set_title('IP man Parody')
        axs[0].set_ylim(0, 0.5)

        axs[1].plot(arr[1])
        axs[1].plot(x, y, color='red')
        axs[1].set_title('IP man Pirate 1')
        axs[1].set_ylim(0, 0.5)

        axs[2].plot(arr[2])
        axs[2].plot(x, y, color='red')
        axs[2].set_title('IP man Pirate 2')
        axs[2].set_ylim(0, 0.5)

        axs[3].plot(arr[3])
        axs[3].plot(x, y, color='red')
        axs[3].set_title('Dragon Train Pirate 1')
        axs[3].set_ylim(0, 0.5)

        axs[4].plot(arr[3])
        axs[4].plot(x, y, color='red')
        axs[4].set_title('Dragon Train Pirate 2')
        axs[4].set_ylim(0, 0.5)
        plt.show()


if __name__ =='__main__':
    worker = Worker()
    worker.run()
