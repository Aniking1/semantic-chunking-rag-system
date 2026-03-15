from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import os
import json
import uuid
from io import BytesIO
import dotenv

from chroma_db import store_chunks, query_chunks

import os
from dotenv import load_dotenv

load_dotenv()

CHUNK_LENGTH = int(os.getenv("CHUNK_LENGTH", 500))

# Vector DB
# Vector DB
import chromadb
from sentence_transformers import SentenceTransformer

# Set up Gemini
import google.generativeai as genai

# File parsing
try:
    import PyPDF2
except:
    PyPDF2 = None

try:
    import docx
except:
    docx = None

dotenv.load_dotenv()

# -------------------- Configuration --------------------
HF_API_KEY = os.environ.get("HF_API_KEY")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DB_HOST = os.environ.get("CHROMA_DB_HOST", "localhost")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gemini-2.5-flash")
DATA_DIR = os.environ.get("RAG_DATA_DIR", "./data")
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", 100))
PORT = int(os.getenv("PORT", 8000))

# -------------------- Vector Database Setup --------------------

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="rag_collection"
)

# -------------------- Embedding Model --------------------

embedding_model = SentenceTransformer(EMBED_MODEL_NAME)



# Create the data directory that holds the documents that form the vector data.
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="Demo RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# Embedding model
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# LLM model
genai.configure(api_key="AIzaSyCcLuKnqqloH5E4xstDyvhXnyxKW7x0ACc")
llm_model = genai.GenerativeModel(LLM_MODEL_NAME)



# ---- Helper functions ------ #

# Chunking the text
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, (start, end)))

        if end == length:
            break
        start = end - overlap if end - overlap > start else end
    return chunks


def extract_text(filename: str, content: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".txt") or lower.endswith(".md"):
        try:
            return content.decode("utf-8")
        except:
            return content.decode("latin-1", errors="ignore")


    if lower.endswith(".pdf"):
        if PyPDF2 is None:
            raise HTTPException(500, "PyPDF2 not installed")

        reader = PyPDF2.PdfReader(BytesIO(content))
        txt = []
        for p in reader.pages:
            try:
                txt.append(p.extract_text() or "")
            except:
                txt.append("")
        
        return "\n".join(txt)


    if lower.endswith(".docx"):
        if docx is None:
            raise HTTPException(500, "python-docx not installed")

        d = docx.Document(BytesIO(content))
        return "\n".join([p.text for p in d.paragraphs])


    try:
        return content.decode("utf-8")
    except:
        return content.decode("latin-1", errors="ignore")


# ---- Endpoints ------- #
@app.post("/upload")
def upload_files(file: UploadFile = File(...), context: Optional[str] = Form(None)):

    if context is None:
        context = f"ctx-{uuid.uuid4().hex[:8]}"

    ctx_dir = os.path.join(DATA_DIR, context)
    os.makedirs(ctx_dir, exist_ok=True)

    file_dir = os.path.join(ctx_dir, "files")
    os.makedirs(file_dir, exist_ok=True)

    metadata_path = os.path.join(ctx_dir, "metadata.json")

    metadata = []
    if os.path.exists(metadata_path):
        metadata = json.load(open(metadata_path))

    new_vectors = []

    f = file
    content = f.file.read()
    text = extract_text(f.filename, content)

    chunks = chunk_text(text)

    # save file
    dest = os.path.join(file_dir, f.filename)
    with open(dest, "wb") as out:
        out.write(content)

    # process chunks
    for chunk, (s, e) in chunks:

        vec = embed_model.encode(chunk).tolist()

        cid = uuid.uuid4().hex

        meta = {
            "id": cid,
            "context": context,
            "filename": f.filename,
            "offset_start": s,
            "offset_end": e,
            "text": chunk,
        }

        new_vectors.append((cid, vec, meta))
        metadata.append(meta)

    # send chunks to ChromaDB
    texts = [meta["text"] for _, _, meta in new_vectors]
    metas = [meta for _, _, meta in new_vectors]
    store_chunks(texts, metas)

    json.dump(metadata, open(metadata_path, "w"), indent=2)

    return {"context": context, "chunks": len(new_vectors)}



#@app.post("/chat")


@app.post("/chat")
def chat(context: str = Form(...), query: str = Form(...)):

    try:
        retrieved = query_chunks(query)

        if not retrieved:
            return {"answer": "No relevant document chunks found.", "context": []}

        context_block = "\n".join(retrieved)
        prompt = f"""
You are an AI assistant answering questions based only on the provided document context.

Instructions:
- Use ONLY the information in the context below.
- If the answer is not present in the context, say:
  "I cannot find the answer in the provided document."
- Be clear and concise.

Context:
{context_block}

Question:
{query}

Answer:
"""
        

#         prompt = f"""
# Context:
# {context_block}

# Question:
# {query}

# Answer the question using ONLY the context above.
# """
#         prompt = f"""
# You are an AI assistant answering questions based only on the provided document context.

# Instructions:
# - Use ONLY the information in the context below.
# - If the answer is not present in the context, say:
#   "I cannot find the answer in the provided document."
# - Be clear and concise.

# Context:
# {context_block}

# Question:
# {query}

# Answer:
# """


        response = llm_model.generate_content(prompt)

        return {
            "answer": response.text,
            "context": retrieved
        }

    except Exception as e:
        return {
            "error": str(e)
        }

# def chat(context: str = Form(...), query: str = Form(...)):
#     # embed query
   
#     retrieved = query_chunks(query)
#     if not retrieved:
#         raise HTTPException(status_code=404, detail="No relevant document chunks found")


#     # With LLM
#     prompt = f"""
# Context:
#     {context_block}

#     Question: {query}
    
#     Based on the context provided above, generate a succint answer to the query above.
# """
#     response = llm_model.generate_content(prompt)

#     return {"answer": response.text, "context": retrieved}


@app.get("/contexts")
def list_contexts():
    return [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]


@app.get("/context/{name}/metadata")
def get_metadata(name: str):
    p = os.path.join(DATA_DIR, name, "metadata.json")
    if not os.path.exists(p):
        raise HTTPException(404, "Context not found")
    return json.load(open(p))

#from utils import semantic_chunk_text

#test_text = "This is a simple test document to demonstrate semantic chunking in our RAG system."

#chunks = semantic_chunk_text(test_text, CHUNK_LENGTH)

#print(chunks)