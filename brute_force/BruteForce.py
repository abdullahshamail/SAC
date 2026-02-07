from .NDCC import NDCC
from sklearn.cluster import DBSCAN
from .utils import calculateDiversity_with_attr_win_tot
class BruteForce(object):
    """Coherence Moving Cluster (CMC) algorithm

    Attributes:
        k (int):  Min number of consecutive timestamps to be considered a convoy
        m (int):  Min number of elements to be considered a convoy
        dThreshold (float):  Diversity threshold
        features (list):  List of features for input data
    """
    def __init__(self, eps, k, m, dThreshold, features, userToEmb, keyToAttr):
        self.eps = eps
        self.k = k
        self.m = m
        self.dThreshold = dThreshold
        self.clf = DBSCAN(eps=self.eps, min_samples=m)
        self.features = features
        self.userToEmb= userToEmb
        self.keyToAttr = keyToAttr

    def fit_predict(self, X, y=None, sample_weight=None):
        convoy_candidates = set()
        columns = len(X[0])
        column_iterator = range(columns)
        output_convoys = []

        for column in column_iterator:
            current_convoy_candidates = set()
            values = [row[column] if isinstance(row[column], (list, set)) else [row[column]] for row in X]
            if len(values) < self.m:
                continue
            clusters = self.clf.fit_predict(values, y=y, sample_weight=sample_weight) 
            unique_clusters = set(clusters)
            clusters_indices = dict((cluster, NDCC(indices=set(), is_assigned=False, start_time=None, end_time=None, diversity=0.0)) for cluster in unique_clusters)

            for index, cluster_assignment in enumerate(clusters):
                clusters_indices[cluster_assignment].indices.add(index)



            for convoy_candidate in convoy_candidates:

                convoy_candidate.is_assigned = False
                for cluster in unique_clusters:
                    cluster_indices = clusters_indices[cluster].indices
                    cluster_candidate_intersection = cluster_indices & convoy_candidate.indices

                    if len(cluster_candidate_intersection) < self.m:
                        continue
                    neighbor_diversity, cohesion, attribute = calculateDiversity_with_attr_win_tot(cluster_candidate_intersection, self.userToEmb, self.features, self.keyToAttr)
                    if  neighbor_diversity < self.dThreshold:
                        continue
                    convoy_candidate.indices = cluster_candidate_intersection
                    convoy_candidate.diversity = neighbor_diversity
                    convoy_candidate.cohesion = cohesion
                    convoy_candidate.sattribute = attribute
                    current_convoy_candidates.add(convoy_candidate)
                    convoy_candidate.end_time = column
                    clusters_indices[cluster].is_assigned = convoy_candidate.is_assigned = True


                candidate_life_time = (convoy_candidate.end_time - convoy_candidate.start_time) + 1
                if (not convoy_candidate.is_assigned or column == column_iterator[-1]) and candidate_life_time >= self.k and convoy_candidate.diversity >= self.dThreshold:
                    output_convoys.append(convoy_candidate)


            for cluster in unique_clusters:
                cluster_data = clusters_indices[cluster]
                if cluster_data.is_assigned:
                    continue
                if len(cluster_data.indices) >= self.m:
                    cluster_data.diversity, cluster_data.cohesion, cluster_data.sattribute = calculateDiversity_with_attr_win_tot(cluster_data.indices, self.userToEmb, self.features, self.keyToAttr)
                    if cluster_data.diversity >= self.dThreshold:
                        cluster_data.start_time = cluster_data.end_time = column
                        current_convoy_candidates.add(cluster_data)
            convoy_candidates = current_convoy_candidates
        return output_convoys