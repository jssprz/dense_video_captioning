import os
import json
import pickle

import numpy as np
from sklearn.cluster import KMeans


class WindowsSizeModel:
    def __init__(self, configuration):
        self.config = configuration

    def train(self):
        annotations_path = os.path.join(self.config.data_dir, 'captions/train.json')
        with open(annotations_path) as f:
            data = json.load(f)

        total_counts = 0
        x = []
        for v in data.items():
            duration = v[1]['duration']
            total_counts += len(v[1]['timestamps'])
            for t in v[1]['timestamps']:
                fragment_len = t[1]-t[0]
                x.append(fragment_len/duration)

        self.cluster = KMeans(n_clusters=total_counts//len(data))
        self.cluster.fit(np.array(x).reshape(-1, 1))
        return self.cluster.cluster_centers_

    def get_windows_sizes(self, videos_data):
        centers = self.cluster.cluster_centers_
        result = [[c * v[1]['duration'] for c in centers] for v in videos_data.items()]
        return result

    def load_model(self, path):
        with open(path, 'rb') as f:
            self.cluster = pickle.load(f)

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.cluster, f)
