from typing import Tuple

from sklearn.cluster import KMeans

class ClusteringProcessor:
    def __init__(self, model, data: list[Tuple[int, int, int]]):
        self.data = data
        self.model = model

    def compute(self):
        self.model.fit(X=self.data)


def compute_kmeans(data: list[Tuple[int, int ,int]]) -> list[dict]:
    kmeans_model = KMeans()
    processor = ClusteringProcessor(kmeans_model, data)
    processor.compute()
    return processor.model.labels_