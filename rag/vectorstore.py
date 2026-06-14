from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import hashlib
import os

CHROMA_PATH = "chroma_db"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "batch_size": 128,          # doubled from 64
            "normalize_embeddings": True,
        }
    )

def get_pdf_hash(uploaded_file):
    uploaded_file.seek(0)
    content = uploaded_file.read()
    uploaded_file.seek(0)
    return hashlib.md5(content).hexdigest()[:12]

def create_vectorstore(chunks, pdf_hash=None):
    cache_path = f"{CHROMA_PATH}_{pdf_hash}" if pdf_hash else CHROMA_PATH

    if pdf_hash and os.path.exists(cache_path):
        return Chroma(
            persist_directory=cache_path,
            embedding_function=get_embeddings()
        )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=cache_path
    )
    return vectorstore

def load_vectorstore(pdf_hash=None):
    cache_path = f"{CHROMA_PATH}_{pdf_hash}" if pdf_hash else CHROMA_PATH
    return Chroma(
        persist_directory=cache_path,
        embedding_function=get_embeddings()
    )