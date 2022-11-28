import pickle
import time

import numpy as np

from framework import util

CLUSTER_THRESHOLD = 0.4


class Cluster:
    def __init__(self, cluster_id, core_faces, count=1, last_seen=None):
        self.last_seen = last_seen or time.time()
        self.id = cluster_id
        self.core_faces = core_faces
        self.count = count

    def __str__(self):
        return f"Cluster-{self.id}: {self.count}"

    def __repr__(self):
        return str(self)

    def to_tuple(self):
        return (
            util.timestamp_to_sql(self.last_seen),
            self.id,
            self.count,
            pickle.dumps(self.core_faces),
        )

    @classmethod
    def from_tuple(cls, tuple_):
        # gotta do this better
        return cls(int(tuple_[1]), pickle.loads(tuple_[3]), int(tuple_[2]), util.timestamp_from_sql(tuple_[0]))

    @staticmethod
    def _calc_distance(face_x, face_y):
        return np.linalg.norm(face_x - face_y)

    def is_similar(self, face):
        return any(self._calc_distance(face, cf) < CLUSTER_THRESHOLD
                   for cf in self.core_faces)

    def extend(self, cluster):
        self.core_faces.extend(cluster.core_faces)
        self.count += cluster.count
