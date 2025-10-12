# main.py
# Provides: load_resources(), set_api_key(), search(), answer_query()
# Designed to be called from Streamlit or CLI

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

# Google Generative AI client
try:
    import google.generativeai as genai
except Exception:
    genai = None

# Paths
CHUNKS_FILE = "chunks.json"
INDEX_FILE = "legal.index"
META_FILE = "legal_meta.json"
MODEL_NAME = "all-MiniLM-L6-v2"

# Globals (populated by load_resources)
_model = None
_index = None
_data = None


def load_resources():
    global _model, _index, _data
    if _model is not None and _index is not None and _data is not None:
        return

    print("Loading model and index (this may take a minute)...")
    _model = SentenceTransformer(MODEL_NAME)

    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        raise FileNotFoundError(
            "Missing index or meta. Please run build_index.py locally "
            "and commit legal.index & legal_meta.json to the repo."
        )

    _index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        _data = json.load(f)

    print("Resources loaded.")


def set_api_key(api_key: str):
    """Call this before answer_query if you want to use Gemini."""
    if genai is None:
        raise RuntimeError("google-generativeai is not installed or importable")
    genai.configure(api_key=api_key)


def search(query: str, k: int = 5):
    """Retrieve top K chunks using embeddings and FAISS index."""
    load_resources()
    q_emb = _model.encode([query], convert_to_numpy=True)
    D, I = _index.search(np.asarray(q_emb, dtype=np.float32), k)
    results = [(_data[i], float(D[0][j])) for j, i in enumerate(I[0])]
    return results  # list of (meta, distance)


def _build_prompt(query: str, docs: list):
    """Builds the prompt for Gemini or for local debugging."""
    snippets = []
    for i, (meta, _) in enumerate(docs):
        snippets.append(f"[Source:{meta.get('source','unknown')} | chunk:{i}]\n{meta['text'][:800]}")

    context = "\n\n".join(snippets)

    prompt = f"""
Answer the question **using only the context below**. Provide a detailed answer with:

- Section number(s)
- Punishment or penalty
- Any special notes or conditions
- Cite the source PDF file for each section

Context:
{context}

Question:
{query}

If the answer cannot be found in the context, say you don't know. End with: "Not legal advice."
"""
    return prompt


def answer_query(query: str, k: int = 5, use_gemini: bool = True):
    """Performs retrieval and calls Gemini (if available).
    Returns either the detailed answer from Gemini or the prompt for debugging.
    """
    docs = search(query, k=k)
    prompt = _build_prompt(query, docs)

    if use_gemini:
        if genai is None:
            raise RuntimeError(
                "google-generativeai is not available. Install it or set use_gemini=False"
            )

        # Use a valid Gemini model
        model_name = "gemini-2.5-flash-lite"  # safe default for most API keys
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        return resp.text

    else:
        # return the prompt only for local debugging
        return prompt
