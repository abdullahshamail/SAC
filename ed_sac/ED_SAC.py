from .IDC import IDC
from sklearn.cluster import DBSCAN
import numpy as np
from .utils import attr_align_from_indices
class ED_SAC(object):
    def __init__(self, eps, k, m, dThreshold, features, userToEmb, keyToAttr, attr_names):
        self.eps = eps
        self.k = k
        self.m = m
        self.dThreshold = dThreshold
        self.clf = DBSCAN(eps=self.eps, min_samples=m)
        self.features = features
        self.userToEmb= userToEmb

        self.keyToAttr = keyToAttr
        self.attr_names = attr_names

        self.user_to_key = dict(
            zip(
                self.userToEmb['userId'],
                self.userToEmb['embedding']
            )
        )

    def fit_predict(self, X, y=None, sample_weight=None):
        convoy_candidates = set()
        columns = len(X[0])
        column_iterator = range(columns)
        output_convoys = []

        for column in column_iterator:
            current_convoy_candidates = set()
            values = [
                row[column] if isinstance(row[column], (list, set)) else [row[column]]
                for row in X
            ]
            if len(values) < self.m:
                continue
            clusters = self.clf.fit_predict(values, y=y, sample_weight=sample_weight)
            unique_clusters = set(clusters)
            raw_cluster_indices = {c: set() for c in unique_clusters}
            for idx, lbl in enumerate(clusters):
                raw_cluster_indices[lbl].add(idx)

            clusters_indices = {}
            for lbl, indices in raw_cluster_indices.items():
                embedding_keys = [self.user_to_key[i+1] for i in indices]
                cand = IDC(
                    indices=indices,
                    is_assigned=False,
                    start_time=None,
                    end_time=None,
                    embeddingKeys=embedding_keys,
                    embeddings=self.features,
                    attr_names=self.attr_names
                )
                clusters_indices[lbl] = cand

            for convoy in convoy_candidates:
                convoy.is_assigned = False
                matched = False

                for lbl, cluster_cand in clusters_indices.items():
                    inter = cluster_cand.indices & convoy.indices

                    if len(inter) < self.m:
                        continue

                    joiners = inter - convoy.indices
                    leavers = convoy.indices - inter

                    new_n = convoy.n + len(joiners) - len(leavers)
                    new_S = convoy.S.copy()
                    new_Q = convoy.Q

                    for i in joiners:
                        emb_key = self.user_to_key[i+1]
                        raw_vec = self.features[emb_key]
                        norm = np.linalg.norm(raw_vec)
                        v = raw_vec/norm if norm>0 else raw_vec
                        new_S += v
                        new_Q += v.dot(v)

                    for i in leavers:
                        emb_key = self.user_to_key[i+1]
                        raw_vec = self.features[emb_key]
                        norm = np.linalg.norm(raw_vec)
                        v = raw_vec/norm if norm>0 else raw_vec
                        new_S -= v
                        new_Q -= v.dot(v)
                    if new_n > 1:
                        avg_sim = (new_S.dot(new_S) - new_Q) / (new_n * (new_n - 1))
                    
                        new_div = 1 - avg_sim
                    else:
                        new_div = 0.0

                    if new_div < self.dThreshold:
                        continue

                    convoy.r_c = (np.linalg.norm(convoy.S) / convoy.n) if convoy.n > 0 else 0.0

                    convoy.n         = new_n
                    convoy.S         = new_S
                    convoy.Q         = new_Q
                    convoy.diversity = new_div
                    convoy.indices   = inter
                    convoy.end_time  = column
                    convoy.is_assigned      = True
                    cluster_cand.is_assigned = True
                    current_convoy_candidates.add(convoy)
                    matched = True
                    break

                if not matched:
                    life = (convoy.end_time - convoy.start_time) + 1
                    if (not convoy.is_assigned or column == column_iterator[-1]) \
                    and life >= self.k and convoy.diversity >= self.dThreshold:
                        convoy.attr_align = attr_align_from_indices(
                            indices=convoy.indices,
                            user_to_key=self.user_to_key,
                            keyToAttr=self.keyToAttr,
                            features=self.features,
                            attr_names=self.attr_names
                        )
                        output_convoys.append(convoy)

            for lbl, cluster_cand in clusters_indices.items():
                if cluster_cand.is_assigned:
                    continue

                cluster_cand.start_time = column
                cluster_cand.end_time = column
                if cluster_cand.n >= self.m and cluster_cand.diversity >= self.dThreshold:
                    current_convoy_candidates.add(cluster_cand)


            convoy_candidates = current_convoy_candidates
        for convoy in convoy_candidates:
            life = (convoy.end_time - convoy.start_time) + 1
            if life >= self.k and convoy.diversity >= self.dThreshold:
                output_convoys.append(convoy)
                convoy.attr_align = attr_align_from_indices(
                    indices=convoy.indices,
                    user_to_key=self.user_to_key,
                    keyToAttr=self.keyToAttr,
                    features=self.features,
                    attr_names=self.attr_names
                )

        return output_convoys