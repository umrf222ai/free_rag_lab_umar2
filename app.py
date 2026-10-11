"""Streamlit front end for the free PDF RAG lab (uses rag.py).
Run locally:  streamlit run app.py
"""
import streamlit as st
import rag

st.set_page_config(page_title="PDF RAG Chat", page_icon="📄")
st.title("📄 Chat with your PDF (free RAG)")
st.caption("Open-source models running locally: no API key, no token limits.")


@st.cache_resource(show_spinner="Loading models (first run takes a minute)...")
def get_models():
    return rag.load_models()


models = get_models()

# ---- Sidebar: upload + settings ----
with st.sidebar:
    st.header("1. Upload")
    pdf = st.file_uploader("Upload a PDF", type="pdf")
    k = st.slider("Chunks to retrieve", 1, 6, 3)
    if st.button("Clear chat"):
        st.session_state.history = []

# ---- Build the index once per uploaded file ----
if pdf is not None and st.session_state.get("file_name") != pdf.name:
    with st.spinner("Reading and indexing your PDF..."):
        text = rag.read_pdf(pdf)
        if not text.strip():
            st.error("No text found. The PDF may be a scanned image (needs OCR).")
            st.stop()
        chunks = rag.chunk_text(text)
        embeddings = rag.build_index(chunks, models)
        st.session_state.update(file_name=pdf.name, chunks=chunks,
                                embeddings=embeddings, history=[])
    st.sidebar.success(f"Indexed {len(chunks)} chunks from {pdf.name}")

if "chunks" not in st.session_state:
    st.info("👈 Upload a PDF in the sidebar to get started.")
    st.stop()

# ---- Chat ----
for role, msg in st.session_state.get("history", []):
    st.chat_message(role).write(msg)

if question := st.chat_input("Ask something about your PDF"):
    st.chat_message("user").write(question)
    with st.spinner("Thinking..."):
        reply, ctx = rag.answer(question, st.session_state.chunks,
                                st.session_state.embeddings, models, k)
    st.chat_message("assistant").write(reply)
    with st.expander("Retrieved context"):
        st.write(ctx)
    st.session_state.history.append(("user", question))
    st.session_state.history.append(("assistant", reply))
