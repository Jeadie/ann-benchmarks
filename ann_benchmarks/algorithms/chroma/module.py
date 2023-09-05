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
            embeddings=X.tolist(), 
            metadatas=[{"i": i} for i in range(len(X))],
            ids=[str(uuid.UUID(int=i)) for i in range(len(X))]
        )

    def query(self, v, n):
        return self._client.get_collection(self._collection_name).query(
            query_embeddings=[v.tolist()],
            include=["embeddings"],
            n_results=n
        )["embeddings"][0]
    
    # def batch_query(self, X, n):
    #     self.res = self._client.get_collection(self._collection_name).query(
    #         query_embeddings=X.tolist(),
    #         include=["embeddings"],
    #         n_results=n
    #     )["embeddings"]

    # def get_batch_results(self):
    #     return self.res