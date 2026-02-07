import numpy as np

class IDC(object):
    """
    Attributes:
        indices (set): Object indices assigned to the convoy
        is_assigned (bool): Assignment status
        start_time (int): Start time index of the convoy
        end_time (int): Last time index of the convoy
        gap (int): Allowed gap (BARMC)
        totalGaps (int): Total gaps allowed (BARMC)
        diversity (float): Semantic diversity of the convoy
        embeddingKeys (set): Keys to retrieve embedding vectors
        n (int): Number of objects in the convoy
        S (np.array): Sum of semantic vectors (for incremental calculation)
        Q (float): Sum of squared norms of semantic vectors
    """
    __slots__ = ('indices', 'is_assigned', 'start_time', 'end_time', 
                 'gap', 'totalGaps', 'diversity', 'embeddingKeys', "group_stats", "r_c", "attr_align", 
                 'n', 'S', 'Q', 'tracked_members', 'S_trk', 'Q_trk', 'n_trk', 'coverage')

    def __init__(self, indices, is_assigned, start_time=0, end_time=0, gap=0, totalGaps=0,
                 diversity=0.0, embeddingKeys=None, embeddings=None, attr_names=[]):
        self.indices = set(indices)
        self.is_assigned = is_assigned
        self.start_time = start_time
        self.end_time = end_time
        self.gap = gap
        self.totalGaps = totalGaps
        self.embeddingKeys = embeddingKeys

        self.tracked_members = set()               
        self.S_trk = None            
        self.Q_trk = 0.0                      
        self.n_trk = 0


        self.group_stats = {a: {} for a in attr_names}
        self.attr_align = {a: 0.0 for a in attr_names}  
        self.r_c = 0.0
        self.coverage = 0.0

        if embeddings is not None and self.embeddingKeys:
            vectors = np.vstack([embeddings[k] for k in self.embeddingKeys])
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / norms

            self.n = len(vectors)
            self.S = np.sum(vectors, axis=0)
            self.Q = np.sum(np.linalg.norm(vectors, axis=1)**2)
            self.diversity = self._compute_incremental_diversity()
        else:
            self.n = 0
            self.S = None
            self.Q = 0.0
            self.diversity = 0.0

    def __repr__(self):
        return ('<{} indices={}, is_assigned={}, start_time={}, end_time={}, '
                'gap={}, totalGaps={}, diversity={:.4f}, r_c={}, attr_align={}>').format(
            self.__class__.__name__, self.indices, self.is_assigned,
            self.start_time, self.end_time, self.gap, self.totalGaps,
            self.diversity, self.r_c, self.attr_align)

    def add_object(self, obj_index, embedding_key, embedding_vector):
        if self.n == 0:
            self.S = np.zeros_like(embedding_vector)

        self.indices.add(obj_index)

        self.n += 1
        self.S += embedding_vector
        self.Q += np.dot(embedding_vector, embedding_vector)
        self.diversity = self._compute_incremental_diversity()

    def remove_object(self, obj_index, embedding_key, embedding_vector):
        self.indices.discard(obj_index)
        self.n -= 1
        self.S -= embedding_vector
        self.Q -= np.dot(embedding_vector, embedding_vector)

        if self.n > 1:
            self.diversity = self._compute_incremental_diversity()
        else:
            self.diversity = 0.0

    def _compute_incremental_diversity(self):
        if self.n < 2:
            return 0.0

        numerator = np.dot(self.S, self.S) - self.Q
        denominator = self.n * (self.n - 1)
        avg_cos_similarity = numerator / denominator
        diversity = 1 - avg_cos_similarity

        return diversity
