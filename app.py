import streamlit as st
import tempfile
import json
import main

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



if "db" not in st.session_state:
    st.session_state.db = None

if "processed" not in st.session_state:
    st.session_state.processed = False

if "history" not in st.session_state:
    st.session_state.history = []


uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_pdf and st.button("Process PDF"):
    with st.spinner("Extracting, summarizing & indexing PDF..."):
        
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            pdf_path = tmp.name

        chunks = extract_unstructured(pdf_path)

        
        docs = summarize_chunks(chunks)

       
        db = create_vector_store(docs)

        st.session_state.db = db
        st.session_state.processed = True

    st.success("PDF processed successfully! You can now chat with the document.")


st.markdown("---")



if st.session_state.processed:

    st.subheader("💬 Ask a Question")

    query = st.text_input("Enter your question:")

    if query:
        with st.spinner("Retrieving & generating answer..."):

            
            top_chunks = search_with_reranking(st.session_state.db,query, k=5)

            
            raw_chunks = produce_raw_chunks(top_chunks)

           
            answer = generate_final_answer(raw_chunks, query)

         
            st.session_state.history.append(("user", query))
            st.session_state.history.append(("assistant", answer))

       
        st.write(answer)

    
    if st.session_state.history:
        st.markdown("### Chat History")
        for role, msg in st.session_state.history:
            if role == "user":
                st.markdown(f"**You:** {msg}")
            else:
                st.markdown(f"** Bot:** {msg}")
