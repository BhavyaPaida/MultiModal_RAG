# Multimodal RAG PDF Chatbot

A RAG chatbot that processes PDFs containing text, tables, and images to answer questions with citations.

## Features

- Extracts text, tables, and images from PDFs
- AI-powered summarization using Google Gemini
- Vector search with ChromaDB
- Cross-encoder reranking for better accuracy
- Answers include and content references
- Streamlit web interface

## Technology Stack

- LLM: Google Gemini 2.5 Flash
- Embeddings: Google text-embedding-004
- Vector Store: ChromaDB
- Reranker: Cross-Encoder (ms-marco-MiniLM-L-6-v2)
- PDF Processing: Unstructured library
- Framework: LangChain
- UI: Streamlit

## Prerequisites

- Python 3.8 or higher
- Google API Key for Gemini

## Installation

1. Clone the repository
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables

Create a `.env` file:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage

Run the application:

```bash
streamlit run app.py
```

Steps:

1. Upload a PDF file
2. Click "Process PDF" to extract and index content
3. Ask questions in the text input
4. Receive answers with page and content citations

## Project Structure

```
.
├── app.py              # Streamlit UI application
├── main.py             # Core RAG pipeline functions
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create this)
└── dbv1/              # ChromaDB persistence directory (auto-created)
```

## Key Functions

### extract_unstructured(pdf_path)
Extracts text, tables, and images from PDFs.

### summarize_chunks(chunks)
Generates searchable descriptions for each chunk.

### create_vector_store(documents)
Creates ChromaDB vector store with embeddings.

### search_with_reranking(db, query, k)
Two-stage retrieval with vector search and cross-encoder reranking.

### generate_final_answer(chunks, query)
Generates answers using raw content with citations.

## Configuration

Chunking parameters in extract_unstructured():
- max_characters: 10000
- combine_text_under_n_chars: 2000
- new_after_n_chars: 6000

Retrieval parameters in search_with_reranking():
- k: Number of results (default: 3)
- candidate_multiplier: Pool size (default: 4)

LLM temperature in generate_final_answer():
- Default: 0

## How It Works

1. PDF is uploaded and temporarily stored
2. Text, tables, and images are extracted with metadata
3. AI summaries are created for chunks with tables/images
4. Summaries are embedded and stored in ChromaDB
5. User questions trigger semantic search
6. Cross-encoder reranks results
7. LLM generates answers using raw content with citations

## Troubleshooting

API Key Error: Ensure GOOGLE_API_KEY is set in .env file

PDF Processing Fails: Check PDF format and available memory

Slow Processing: Reduce max_characters or use smaller PDFs


## Notes

This application requires a Google API key with access to Gemini models and embedding services.
