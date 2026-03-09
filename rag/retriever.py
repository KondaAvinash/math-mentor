"""
RAG Retriever — queries FAISS index for relevant math formulas.
"""
import os
import pickle
import numpy as np
from config import FAISS_INDEX_PATH, EMBEDDING_MODEL, TOP_K_RESULTS

_retriever_instance = None


class Retriever:
    def __init__(self):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(EMBEDDING_MODEL)
        index_path = os.path.join(FAISS_INDEX_PATH, "index.faiss")
        chunks_path = os.path.join(FAISS_INDEX_PATH, "chunks.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. Run: python -m rag.ingest"
            )

        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.sources = data["sources"]
        print(f"[Retriever] Loaded index ({self.index.ntotal} vectors)")

    def retrieve(self, query: str, top_k: int = None) -> list:
        if top_k is None:
            top_k = TOP_K_RESULTS
        query_vec = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "text": self.chunks[idx],
                    "source": self.sources[idx],
                    "score": float(score),
                })
        return results

    def format_context(self, chunks: list) -> str:
        if not chunks:
            return "No relevant formulas found."
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(f"[{i}] Source: {c['source']}\n{c['text']}")
        return "\n\n".join(parts)


def get_retriever() -> Retriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance
