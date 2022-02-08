from cProfile import label
from inspect import _void
from re import X
from typing import Tuple
from matplotlib import projections, pyplot as plt
import numpy as np

from sklearn.cluster import KMeans

class ClusteringProcessor:
    def __init__(self, model, data: list[Tuple[int, int, int]]):
        self.data = data
        self.model = model
        self.y_kmeans = []
    def compute(self):
        self.model.fit(X=self.data)
        self.y_kmeans = self.model.fit_predict(X=self.data)

    def diagram_kmeans(self) -> None:
        # colors = ["#ff0000","#59ff00","#00ffe1","#0048ff","#8400ff","#ff00d9","#ffff00","#ff8000"]
        # fig = plt.figure()
        # ax = fig.add_subplot()
        # ax.scatter(labels, data, c=colors)

        print(self.data)
        plt.scatter(*zip(*self.data),self.y_kmeans, c = self.y_kmeans)
        # plt.legend()
        plt.title("Kmeans Diagram for recipes score")
        plt.xlabel("Score")
        plt.ylabel("Cluster")
        plt.savefig("kmeans_clustering.png")

def compute_kmeans(data: list[Tuple[int, int ,int]]) -> list[dict]:
    kmeans_model = KMeans()
    processor = ClusteringProcessor(kmeans_model, data)
    processor.compute()
    processor.diagram_kmeans()
    return processor.model.labels_
    


