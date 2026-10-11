"""
rag.py - Simple RAG core (free, no API key, no token limits).

Pipeline: PDF -> text -> chunks -> embeddings -> retrieve -> local LLM answer.
Works in 3 ways:
  * Colab / terminal:  python rag.py   (prompts you to upload / enter a PDF)
  * Imported by the Streamlit front end (app.py)
"""
import numpy as np

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"   # free & CPU friendly; use flan-t5-large on a GPU


def load_models():
    """Load the embedding model and the open-source LLM (downloaded once, free)."""
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBED_MODEL, device=device)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    llm = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL).to(device)
    return embedder, tokenizer, llm, device


def read_pdf(file):
    """Extract text from a PDF (file path or file-like object)."""
    from pypdf import PdfReader
    reader = PdfReader(file)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def chunk_text(text, chunk_words=200, overlap=40):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    step = chunk_words - overlap
    chunks = [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), step)]
    return [c for c in chunks if c]


def build_index(chunks, models):
    """Embed all chunks. This is the 'training'/indexing step of RAG."""
    embedder = models[0]
    return embedder.encode(chunks, normalize_embeddings=True)


def retrieve(question, chunks, embeddings, models, k=3):
    """Return the top-k chunks most similar to the question (cosine similarity)."""
    q = models[0].encode([question], normalize_embeddings=True)[0]
    top = np.argsort(embeddings @ q)[::-1][:k]
    return [chunks[i] for i in top]


def answer(question, chunks, embeddings, models, k=3):
    """Retrieve context and generate an answer from it. Returns (answer, context)."""
    import torch
    _, tokenizer, llm, device = models
    context = "\n\n".join(retrieve(question, chunks, embeddings, models, k))
    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say \"I don't know\".\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=256, num_beams=4)
    return tokenizer.decode(out[0], skip_special_tokens=True), context


def ask_for_pdf():
    """Prompt the user for a PDF: upload dialog in Colab, otherwise a file path."""
    try:
        from google.colab import files
        print("Please upload a PDF file...")
        return list(files.upload().keys())[0]
    except ImportError:
        return input("Enter the path to your PDF: ").strip().strip('"')


def main():
    print("Loading models (first run downloads them)...")
    models = load_models()

    pdf_path = ask_for_pdf()
    print("Reading PDF...")
    text = read_pdf(pdf_path)
    if not text.strip():
        raise SystemExit("No text found. The PDF may be a scanned image (needs OCR).")

    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks. Building index...")
    embeddings = build_index(chunks, models)
    print("RAG is ready! Type 'quit' to stop.\n")

    while True:
        q = input("Your question: ").strip()
        if q.lower() in {"quit", "exit", ""}:
            break
        reply, _ = answer(q, chunks, embeddings, models)
        print("Answer:", reply, "\n")


if __name__ == "__main__":
    main()
