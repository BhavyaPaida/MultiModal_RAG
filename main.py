import os
from unstructured.partition.pdf import partition_pdf
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

from langchain_google_genai import ChatGoogleGenerativeAI


import base64
import json
load_dotenv()
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
def extract_unstructured(pdf_path):
    image_dir = "unstructured"
    os.makedirs(image_dir, exist_ok=True)
    chunks=partition_pdf(
        filename=pdf_path,
        infer_table_structure=False,
        strategy="hi_res",
        extract_image_blocks=True,
        extract_image_block_types=["Image","Figure", "Table"],
        extract_image_block_to_payload=True,
        extract_image_block_to_file=False,
        hi_res_output_path=os.path.abspath(image_dir),
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )
    return chunks

#chunks=extract_unstructured("qatar_test_doc.pdf")
#print([type(c) for c in chunks])


def safe_coordinates(coords_obj):
    """Convert Unstructured CoordinatesMetadata to JSON-safe dict."""
    if coords_obj is None:
        return None

    try:
        # coords_obj.points is a list of coordinate tuples
        pts = coords_obj.points
        pts_list = [[float(x), float(y)] for (x, y) in pts]

        return {
            "points": pts_list,
            "system": str(coords_obj.system)
        }
    except:
        return None

def seperate_content_types(chunk):
    def safe_coordinates(coords_obj):
        if coords_obj is None:
            return None
        try:
            pts = coords_obj.points
            pts_list = [[float(x), float(y)] for (x, y) in pts]
            return {
                "points": pts_list,
                "system": str(coords_obj.system)
            }
        except:
            return None

    content_data = {
        "text": chunk.text,
        "tables": [],
        "images": [],
        "elements": [],
        "types": ["text"]
    }

    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for idx, element in enumerate(chunk.metadata.orig_elements):

            element_type = type(element).__name__

            coords = safe_coordinates(
                getattr(element.metadata, "coordinates", None)
            )

            element_meta = {
                "index": idx,
                "type": element_type,
                "page": getattr(element.metadata, "page_number", None),
                "coords": coords,
            }

            # TABLE
            if element_type == "Table":
                html = getattr(element.metadata, "text_as_html", element.text)
                content_data["tables"].append(html)
                element_meta["content"] = html
                content_data["types"].append("table")

            # IMAGE
            elif element_type in ["Image", "Figure"]:
                if hasattr(element.metadata, "image_base64"):
                    b64 = element.metadata.image_base64
                    content_data["images"].append(b64)
                    element_meta["content"] = b64
                    content_data["types"].append("image")

            content_data["elements"].append(element_meta)

    content_data["types"] = list(set(content_data["types"]))
    return content_data

#content_data=seperate_content_types(chunks[3])


gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

def create_AI_summary(text: str, tables: list[str], images: list[str]) -> str:
    try:
        if not text.strip():
            text = "(No text found — summarize tables/images only.)"

        prompt = f"""
You are creating a searchable description for document content retrieval.

TEXT CONTENT:
{text}

"""

        if tables:
            prompt += "\nTABLES:\n"
            for i, table in enumerate(tables):
                prompt += f"\nTABLE {i+1}:\n{table}\n"

        prompt += """
TASK:
Produce a detailed searchable description that includes:
- Key facts and statistics
- Main insights
- What questions this document answers
- Interpretations of any images or charts
- Search keywords
- Preserve important metadata like table numbers and page references if visible

Return ONLY the description.
"""

        # MULTIMODAL MESSAGE
        message_content = [{"type": "text", "text": prompt}]

        for img_b64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{img_b64}"
            })

        message = HumanMessage(content=message_content)
        response = gemini.invoke([message])

        return response.content

    except Exception as e:
        print(f"Summary generation failed due to: {e}")
        return None


def summarize_chunks(chunks):
    import json
    from langchain_core.documents import Document

    docs = []
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        print(f"processing chunk {i+1}/{total}")

        content = seperate_content_types(chunk)

        # ---- Generate multimodal summary ----
        summary = None
        if content["tables"] or content["images"]:
            try:
                summary = create_AI_summary(
                    content["text"],
                    content["tables"],
                    content["images"]
                )
                print("✓ AI summary created")
            except Exception as e:
                print("Summary failed:", e)
                summary = None

        # fallback
        if not summary:
            summary = content["text"]

        # ---- JSON encode complex metadata for Chroma ----
        metadata = {
            "raw_text": content["text"],

            "tables_html": json.dumps(content["tables"]),
            "images_base64": json.dumps(content["images"]),
            "elements": json.dumps(content["elements"]),
            "types": json.dumps(content["types"]),

            # NEW
            "chunk_page": content.get("chunk_page", None),
        }

        doc = Document(
            page_content=summary,
            metadata=metadata
        )

        docs.append(doc)

    return docs
#docs=summarize_chunks(chunks)



def create_vector_store(documents, persist_directory="dbv1/chroma_db"):
    embedder = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004"
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedder,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    return vectorstore
#db = create_vector_store(docs)



# ---------------------------------------
# Load cross encoder once
# ---------------------------------------



def search_with_reranking(db,query, k=3, candidate_multiplier=4):
    """
    Two-stage retrieval:
    1. Retrieve vector neighbors
    2. Rerank with cross-encoder
    """

    # ---------------------- Stage 1: Vector Retrieval ----------------------
    candidate_k = min(k * candidate_multiplier, 20)
    
    retriever = db.as_retriever(search_kwargs={"k": candidate_k})
    results = retriever.invoke(query)     # list of documents/chunks

    # If retriever returns Document objects, get page_content
    candidate_texts = [r.page_content for r in results]

    # ---------------------- Stage 2: Cross-Encoder Reranking ----------------
    # Create query-document pairs
    pairs = [[query, doc] for doc in candidate_texts]

    # Predict similarity for each pair
    scores = reranker.predict(pairs)

    # ---------------------- Sort & return Top-k ----------------------------
    # Combine results + scores
    reranked = sorted(
        list(zip(results, scores)),
        key=lambda x: x[1],
        reverse=True
    )[:k]

    # final list of only documents
    final_docs = [doc for (doc, score) in reranked]

    return final_docs


#query="exxplain table 3a. explain all the values mentioned in the picture"
#top_chunks = search_with_reranking(query, k=5)
def produce_raw_chunks(top_chunks):


    raw_chunks = []

    for c in top_chunks:
        raw_text = c.metadata.get("raw_text", "")
        raw_tables = json.loads(c.metadata.get("tables_html", "[]"))
        raw_images = json.loads(c.metadata.get("images_base64", "[]"))
        page = c.metadata.get("chunk_page", None)

        raw_chunks.append({
            "raw_text": raw_text,
            "raw_tables": raw_tables,
            "raw_images": raw_images,
            "page": page
        })
        return raw_chunks

    #print("\n================ RAW CHUNK ================\n")
    #print("PAGE:", page)
    #print("RAW TEXT:\n", raw_text[:600])   # print only raw PDF text
    #print("\nRAW TABLE COUNT:", len(raw_tables))
    #print("RAW IMAGE COUNT:", len(raw_images))
    #print("===========================================\n")


def generate_final_answer(chunks, query):
    """
    chunks = [
      {
        "raw_text": "...",
        "raw_tables": [...],
        "raw_images": [...],
        "page": 12
      },
      ...
    ]
    """

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2
        )

        # ---------------- HEADER PROMPT ----------------
        prompt_text = f"""
You are answering a user question based ONLY on the RAW PDF content.

USER QUESTION:
{query}

RULES:
1. Only use the RAW text, RAW tables, and RAW images provided.
2. Ignore summaries; they are for retrieval only.
3. Use page numbers when citing.
4. Never hallucinate.
INCLUDE THE CITATION: PAGE NUMBER(from metadat provided), TABLE NUMBER, IMAGE NUMBER WHEREVER POSSIBLE.(FROM THE RAW DATA WE ARE PROVIDING)
RAW CONTENT STARTS BELOW:
"""

        message_content = [{"type": "text", "text": prompt_text}]

        # ---------------- RAW CONTENT LOOP ----------------
        for i, chunk in enumerate(chunks):

            raw_text = chunk.get("raw_text", "")
            raw_tables = chunk.get("raw_tables", [])
            raw_images = chunk.get("raw_images", [])
            page = chunk.get("page")

            message_content.append({
                "type": "text",
                "text": f"\n\n====== RAW DOCUMENT {i+1} (Page {page}) ======\n"
            })

            # RAW TEXT
            if raw_text.strip():
                message_content.append({
                    "type": "text",
                    "text": f"\nRAW TEXT:\n{raw_text}\n"
                })

            # RAW TABLES
            for t_idx, table_html in enumerate(raw_tables):
                message_content.append({
                    "type": "text",
                    "text": f"\nTABLE {t_idx} (Page {page}):\n{table_html}\n"
                })

            # RAW IMAGES
            for img_idx, b64 in enumerate(raw_images):

                message_content.append({
                    "type": "text",
                    "text": f"[IMAGE {img_idx} — Page {page}]"
                })

                message_content.append({
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{b64}"
                })

        # ---------------- END + LLM CALL ----------------
        message_content.append({
            "type": "text",
            "text": "\nEND OF RAW CONTENT.\n\nFINAL ANSWER:\n"
        })

        response = llm.invoke([HumanMessage(content=message_content)])
        return response.content

    except Exception as e:
        print(f"Answer generation failed: {e}")
        return "Sorry, an error occurred."

#final_answer = generate_final_answer(raw_chunks, query)
#print(final_answer)
