# indian-law-rag


Indian Law RAG assistant — Streamlit + FAISS + HuggingFace embeddings + Gemini (free-tier)


## Quickstart (local)


1. Put PDFs into `data/` (e.g., ipc.pdf, crpc.pdf)
2. `python preprocessing.py` # creates chunks.json
3. `python build_index.py` # builds legal.index and legal_meta.json
4. `pip install -r requirements.txt`
5. (Optional) set env var: `export GEMINI_API_KEY=...`
6. `streamlit run app.py`


## Deploy to Streamlit Cloud


1. Push this repo to GitHub.
2. Go to Streamlit Cloud, create a new app and connect the repo.
3. Add the secret `GEMINI_API_KEY` in the Streamlit Cloud Secrets UI (Settings → Secrets).
4. Make sure `legal.index` and `legal_meta.json` are committed to the repository (or generated at startup).


## Notes
- This tool is for informational purposes only and does not constitute legal advice.
- For large scale usage (many users / heavy traffic), consider running a separate backend (server with a GPU) and using a paid LLM or a local LLM on a hosted machine.