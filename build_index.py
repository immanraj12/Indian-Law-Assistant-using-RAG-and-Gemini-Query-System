# build_index.py
# Read chunks.json, compute embeddings with sentence-transformers, build FAISS index

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "chunks.json"
INDEX_FILE = "legal.index"
META_FILE = "legal_meta.json"
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    print("Loading chunks...")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [d["text"] for d in data]
    print(f"Generating embeddings for {len(texts)} chunks using {MODEL_NAME}...")

    model = SentenceTransformer(MODEL_NAME)
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    dim = embs.shape[1]
    print(f"Embedding dimension: {dim}")

    index = faiss.IndexFlatL2(dim)
    index.add(np.asarray(embs, dtype=np.float32))

    faiss.write_index(index, INDEX_FILE)
    print(f"Saved FAISS index to {INDEX_FILE}")

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata to {META_FILE}")


if __name__ == "__main__":
    main()
