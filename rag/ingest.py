"""
RAG Ingest — chunks knowledge base files, embeds them, saves FAISS index.
Run once: python -m rag.ingest
"""
import os
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from config import KNOWLEDGE_BASE_PATH, FAISS_INDEX_PATH, EMBEDDING_MODEL


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def ingest():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    kb_path = Path(KNOWLEDGE_BASE_PATH)
    all_chunks = []
    all_sources = []

    for file in sorted(kb_path.glob("*.txt")):
        print(f"Processing: {file.name}")
        text = file.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_sources.append(file.name)

    print(f"Total chunks: {len(all_chunks)}")
    print("Generating embeddings...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity (normalized vectors)
    index.add(embeddings)

    # Save index and chunks
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    faiss.write_index(index, os.path.join(FAISS_INDEX_PATH, "index.faiss"))
    with open(os.path.join(FAISS_INDEX_PATH, "chunks.pkl"), "wb") as f:
        pickle.dump({"chunks": all_chunks, "sources": all_sources}, f)

    print(f"✅ FAISS index saved to {FAISS_INDEX_PATH}")
    print(f"   Vectors: {index.ntotal}, Dimensions: {dim}")


if __name__ == "__main__":
    ingest()
