import streamlit as st
import tempfile
import json
import main
# Import all your functions exactly from your file
from main import (
    extract_unstructured,
    summarize_chunks,
    create_vector_store,
    search_with_reranking,
    produce_raw_chunks,
    generate_final_answer
)


st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("📄 Multimodal RAG PDF Chatbot")


# -----------------------------------------------
#   SESSION STATE INITIALIZATION
# -----------------------------------------------
if "db" not in st.session_state:
    st.session_state.db = None

if "processed" not in st.session_state:
    st.session_state.processed = False

if "history" not in st.session_state:
    st.session_state.history = []


# -----------------------------------------------
#   PDF UPLOAD
# -----------------------------------------------
uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_pdf and st.button("Process PDF"):
    with st.spinner("Extracting, summarizing & indexing PDF..."):
        
        # Save PDF to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            pdf_path = tmp.name

        # 1️⃣ Extract unstructured content
        chunks = extract_unstructured(pdf_path)

        # 2️⃣ Summarize each document chunk
        docs = summarize_chunks(chunks)

        # 3️⃣ Embed + Store in Vector DB
        db = create_vector_store(docs)

        st.session_state.db = db
        st.session_state.processed = True

    st.success("PDF processed successfully! You can now chat with the document.")


st.markdown("---")


# -----------------------------------------------
#   CHATBOT SECTION (AFTER PROCESSING)
# -----------------------------------------------
if st.session_state.processed:

    st.subheader("💬 Ask a Question")

    query = st.text_input("Enter your question:")

    if query:
        with st.spinner("Retrieving & generating answer..."):

            # 1️⃣ Retrieve relevant chunks using vector + reranker
            top_chunks = search_with_reranking(st.session_state.db,query, k=5)

            # 2️⃣ Convert summarized chunks → RAW text/tables/images
            raw_chunks = produce_raw_chunks(top_chunks)

            # 3️⃣ Final answer from RAW data
            answer = generate_final_answer(raw_chunks, query)

            # Save chat history
            st.session_state.history.append(("user", query))
            st.session_state.history.append(("assistant", answer))

        # Display answer
        st.write(answer)

    # -----------------------------------------------
    #   CHAT HISTORY
    # -----------------------------------------------
    if st.session_state.history:
        st.markdown("### Chat History")
        for role, msg in st.session_state.history:
            if role == "user":
                st.markdown(f"**You:** {msg}")
            else:
                st.markdown(f"** Bot:** {msg}")
