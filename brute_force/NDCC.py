class NDCC(object): #Naive diverse convoy candidate
    """
    Attributes:
        indices(set): The object indices assigned to the convoy
        is_assigned (bool):
        start_time (int):  The start index of the convoy
        end_time (int):  The last index of the convoy
        gap (int):  gpa allowed (BARMC)
        totalGaps (int):  total gaps allowed (BARMC)
        diversity (float):  diversity of the convoy
    """
    __slots__ = ('indices', 'is_assigned', 'start_time', 'end_time', 'gap', 'totalGaps', 'diversity', 'cohesion', "embeddingKeys", "sattribute")

    def __init__(self, indices, is_assigned, start_time = 0, end_time = 0, gap = 0, totalGaps = 0, diversity = 0.0, cohesion=0.0, embeddingKeys = None, sattribute=None):
        self.indices = indices
        self.is_assigned = is_assigned
        self.start_time = start_time
        self.end_time = end_time
        self.gap = gap
        self.totalGaps = totalGaps
        self.diversity = diversity
        self.cohesion = cohesion
        self.embeddingKeys = embeddingKeys
        self.sattribute = sattribute
        # self.lifeTime = self.end_time - self.start_time + 1

    def __repr__(self):
        return '<%r indices=%r, is_assigned=%r, start_time=%r, end_time=%r, gap=%r, totalGaps=%r, diversity=%r, cohesion=%r, sattribute=%r>' % (self.__class__.__name__, self.indices, self.is_assigned, self.start_time, self.end_time, self.gap, self.totalGaps, self.diversity, self.cohesion, self.sattribute)
