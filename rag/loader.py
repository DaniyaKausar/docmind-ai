from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os

def load_and_split_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    for i, page in enumerate(pages):
        page.metadata["page"] = i

    # Larger chunks = fewer embeddings = faster processing
    # Quality stays same because overlap keeps context
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,      # up from 800
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )

    chunks = splitter.split_documents(pages)
    os.unlink(tmp_path)
    return chunks