import os
import faiss
import pickle
import numpy as np

class PersonDatabase:
    def __init__(self, db_path="reid_db.pkl", threshold=0.4, variance_warning=0.3):
        self.db_path          = db_path
        self.threshold        = threshold
        self.variance_warning = variance_warning
        self.identities       = {}
        self.faiss_index      = None
        self.idx_to_name      = {}
        self.load()

    def load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                self.identities = pickle.load(f)
        self.rebuild_index()

    def save(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.identities, f)

    def rebuild_index(self):
        if not self.identities:
            self.faiss_index = None
            self.idx_to_name = {}
            return

        first_name = next(iter(self.identities))
        dim = self.identities[first_name][0].shape[0]

        self.faiss_index = faiss.IndexFlatL2(dim)
        self.idx_to_name = {}

        for idx, (name, embs) in enumerate(self.identities.items()):
            mean_emb = np.mean(embs, axis=0)
            centroid = mean_emb / np.linalg.norm(mean_emb)
            self.faiss_index.add(np.array([centroid], dtype=np.float32))
            self.idx_to_name[idx] = name

    def search(self, emb):
        if self.faiss_index is None:
            return None, None
        distances, indices = self.faiss_index.search(
            np.array([emb], dtype=np.float32), k=1
        )
        dist = distances[0][0]
        if dist < self.threshold:
            return self.idx_to_name[indices[0][0]], dist
        return None, dist

    def add_embedding(self, name: str, emb: np.ndarray):
        if name not in self.identities:
            self.identities[name] = []
        self.identities[name].append(emb)
        self.save()
        self.rebuild_index()

    def get_names(self):
        return list(self.identities.keys())