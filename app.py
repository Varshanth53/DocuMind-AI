import os
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq
import faiss

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide"
)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing. Please add it to the .env file.")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = None

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

def extract_pages(pdf_file):
    reader = PdfReader(pdf_file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append((i + 1, text))
    return pages

def split_pages_into_chunks(pages, chunk_size=180, chunk_overlap=40):
    chunks = []
    for page_num, text in pages:
        words = text.split()
        step = chunk_size - chunk_overlap
        for start in range(0, len(words), step):
            chunk_text = " ".join(words[start:start+chunk_size])
            if chunk_text.strip():
                chunks.append({"text": chunk_text, "page": page_num})
    return chunks

def create_chunk_embeddings(chunks, model):
    texts = [c["text"] for c in chunks]
    return model.encode_document(texts, convert_to_numpy=True, normalize_embeddings=True)

def create_faiss_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))
    return index

def retrieve_relevant_chunks(question, model, index, chunks, top_k=3):
    q_emb = model.encode_query(question, convert_to_numpy=True, normalize_embeddings=True)
    q_emb = q_emb.astype("float32").reshape(1, -1)
    scores, indices = index.search(q_emb, top_k)
    results = []
    for idx in indices[0]:
        if idx != -1:
            results.append(chunks[idx])
    return results

def generate_answer(question, relevant_chunks, client, model_name):
    context = "\n\n".join([c["text"] for c in relevant_chunks])
    system_prompt = """
You are DocuMind AI, a document question-answering assistant.
Answer using ONLY the context provided.
If the exact answer is not in the context, answer from the closest related context and clearly say "Based on related information in the document".
Do not invent facts outside the context.
"""
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Document context:\n\n{context}\n\nQuestion:\n\n{question}"}
        ],
        temperature=0
    )
    return response.choices[0].message.content

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 DocuMind AI")
    st.write("Ask questions from your PDFs using AI.")
    st.divider()
    uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.uploaded_names = None
        st.rerun()
    st.divider()
    st.caption("Built with Streamlit + LangChain + Groq + FAISS")

# --- MAIN ---
st.title("🧠 DocuMind AI")
st.write("Upload one or more PDFs and ask questions based on their content.")

if not uploaded_files:
    st.info("👈 Upload PDF(s) from the sidebar to start chatting.")
    st.stop()

current_names = tuple([f.name for f in uploaded_files])
if st.session_state.uploaded_names != current_names:
    st.session_state.messages = []
    st.session_state.uploaded_names = current_names

st.success(f"Uploaded: {', '.join(current_names)}")

all_pages = []
for f in uploaded_files:
    all_pages.extend(extract_pages(f))

if not all_pages:
    st.error("No readable text found in PDFs. They may be scanned/image-based.")
    st.stop()

document_chunks = split_pages_into_chunks(all_pages)

with st.spinner("Creating document embeddings..."):
    chunk_embeddings = create_chunk_embeddings(document_chunks, embedding_model)

faiss_index = create_faiss_index(chunk_embeddings)

# Show stats
col1, col2, col3 = st.columns(3)
col1.metric("PDFs", len(uploaded_files))
col2.metric("Pages", len(all_pages))
col3.metric("Chunks", len(document_chunks))

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Sources"):
                for s in message["sources"]:
                    st.caption(f"Page {s['page']}: {s['text'][:200]}...")

user_question = st.chat_input("Ask a question about your documents...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    relevant = retrieve_relevant_chunks(user_question, embedding_model, faiss_index, document_chunks, top_k=3)
    try:
        with st.chat_message("assistant"):
            with st.spinner("Searching the document..."):
                answer = generate_answer(user_question, relevant, groq_client, GROQ_MODEL)
            if not answer or not answer.strip():
                st.error("The model returned an empty response.")
                st.stop()
            st.markdown(answer)
            with st.expander("📚 Sources"):
                for s in relevant:
                    st.caption(f"Page {s['page']}: {s['text'][:200]}...")
    except Exception as e:
        st.error("Unable to generate an answer. Check API key and internet.")
        st.caption(str(e))
        st.stop()

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": relevant})