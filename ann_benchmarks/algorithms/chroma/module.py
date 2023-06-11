import uuid

import chromadb

from ..base.module import BaseANN


class Chroma(BaseANN):

    def __init__(self):
        self._collection_name = "ann_benchmarks_test"
        self._client = chromadb.Client()

    def fit(self, X):
        collection = self._client.create_collection(self._collection_name)
        collection.add(
            X, 
            [{"i": i} for i in range(len(X))],
            [uuid.UUID(int=i) for i in range(len(X))]
        )

    def query(self, v, n):
        return self.get_collection(self._collection_name).query(
            query_texts=[v],
            n_results=n
        ).ids
