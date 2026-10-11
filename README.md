# PDF RAG Lab (100% free)

A simple Retrieval-Augmented Generation (RAG) app. Upload a PDF and ask questions about it.
Uses open-source Hugging Face models that run locally, so there are no API keys and no token limits.

## Files
- `rag.py` - RAG logic (PDF reader, chunking, embeddings, retrieval, answer generation). Run `python rag.py` for a terminal / Colab version.
- `app.py` - Streamlit web front end.
- `requirements.txt` - dependencies.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free on Streamlit Community Cloud
1. Push this repo to GitHub (public).
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
3. Choose this repo, branch `main`, main file `app.py`, then **Deploy**.
