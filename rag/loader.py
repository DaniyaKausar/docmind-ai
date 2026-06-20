from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import tempfile, os

def get_llm():
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=150
    )

def get_document_context(pages):
    """Extract high-level context from first 2 pages."""
    llm = get_llm()
    sample_text = " ".join(p.page_content for p in pages[:2])[:2000]
    prompt = f"""In one sentence, describe this document:
- Document Type (e.g., Research Paper, Novel, Report, Resume)
- Main Topic
- Key Entity (company/person/subject)

Text: {sample_text}

Response format: [Type: X | Topic: Y | Subject: Z]"""
    response = llm.invoke(prompt)
    return response.content.strip()

def load_and_split_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    for i, page in enumerate(pages):
        page.metadata["page"] = i

    # Get document-level context ONCE
    doc_context = get_document_context(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )
    chunks = splitter.split_documents(pages)

    # Prepend context to every chunk — Contextual Retrieval pattern
    for chunk in chunks:
        chunk.page_content = f"{doc_context}\n\n{chunk.page_content}"

    os.unlink(tmp_path)
    return chunks
