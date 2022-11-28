import itertools

import cv2
import dlib
import numpy as np

import config
from cluster import Cluster

MAX_CLUSTERED_FACES = 1
FACE_CLUSTERING_RATE = 1

class FaceClusteringNode:
    def __init__(self):
        super().__init__()
        self.cache = Redis()
        self.landmark_predictor = dlib.shape_predictor(config.LANDMARK_MODEL_PATH)
        self.face_encoder = dlib.face_recognition_model_v1(config.ENCODER_MODEL_PATH)
        self.clusters = {}
        self.cluster_model = saved_cluster()
        self.clusters = {cluster.id: cluster for cluster in self.cluster_model.load()}
        print("Loaded clusters: ", self.clusters)
        self.cluster_id_gen = itertools.count(int(max(self.clusters.keys(), default=0)) + 1)

    def run(self):
        for event in face_detector_sub(FACE_CLUSTERING_RATE)):
            faces = event.faces[:min(len(event.faces), MAX_CLUSTERED_FACES)]
            if len(faces) == 0:
                continue

            rgb = cv2.cvtColor(event.frame.buffer, cv2.COLOR_BGR2RGB)
            clustered_faces = [self.process_face(rgb, f) for f in faces]

    def dump_data(self):
        # print("Saving clusters: ", self.clusters)

        for cluster in self.clusters.values():
            self.cluster_model.save(cluster)

    def cluster_face(self, new_face):
        cluster = Cluster(next(self.cluster_id_gen), [new_face])

        self.clusters[cluster.id] = cluster

        similar_clusters = {id_: cluster for id_, cluster in self.clusters.items() if cluster.is_similar(new_face)}

        # print('similar clusters', similar_clusters)
        if similar_clusters:
            cluster = self.merge_clusters(similar_clusters)
            for id_ in similar_clusters:
                del self.clusters[id_]

            self.clusters[cluster.id] = cluster

        self.dump_data()

        return cluster

    def merge_clusters(self, clusters):
        # print(f"Merging {[cluster.id for cluster in clusters.values()]} clusters")
        merged_cluster = None
        for key in sorted(clusters.keys()):
            cluster = clusters[key]
            if not merged_cluster:
                merged_cluster = cluster
            else:
                merged_cluster.extend(cluster)

        return merged_cluster

    def encode_face(self, image, face):
        bbox = dlib.rectangle(face.x, face.y,
                              face.x + face.width, face.y + face.height)
        raw_landmark = self.landmark_predictor(image, bbox)
        encoding = np.array(self.face_encoder.compute_face_descriptor(
            image, raw_landmark, 1))
        return encoding

    def process_face(self, rgb, face):
        # print('got a face to cluster')
        encoded_face = self.encode_face(rgb, face.bbox)
        cluster = self.cluster_face(encoded_face)

        return ClusteredFace(face.id, cluster.id, cluster.count, face.bbox)
