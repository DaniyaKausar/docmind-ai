from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer, util
import tempfile, os, hashlib

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={"batch_size": 64, "normalize_embeddings": True}
    )

def get_llm():
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

def load_pdf_to_text(uploaded_file):
    """Extract raw text from uploaded PDF."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    uploaded_file.seek(0)

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    os.unlink(tmp_path)

    text = " ".join(p.page_content for p in pages)
    return text, len(pages)

def load_pdf_to_vectorstore(uploaded_file, pdf_name):
    """Load a single PDF into its own ChromaDB collection."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    uploaded_file.seek(0)

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    for i, page in enumerate(pages):
        page.metadata["page"] = i
        page.metadata["source"] = pdf_name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )
    chunks = splitter.split_documents(pages)
    os.unlink(tmp_path)

    # Unique collection per PDF
    safe_name = "".join(c for c in pdf_name if c.isalnum())[:20]
    collection_name = f"pdf_{safe_name}"
    persist_path = f"chroma_multi/{safe_name}"

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=persist_path,
        collection_name=collection_name
    )
    return vectorstore, len(chunks), len(pages)

def query_single_pdf(vectorstore, question, pdf_name):
    """Query one PDF's vectorstore and return answer with source."""
    llm = get_llm()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15}
    )
    docs = retriever.invoke(question)
    context = "\n\n".join(
        f"[Page {d.metadata.get('page',0)+1}]: {d.page_content}"
        for d in docs
    )
    prompt = PromptTemplate.from_template("""
Answer based ONLY on this document context.
If not found, say "Not found in {pdf_name}."

Context: {context}
Question: {question}
Answer:""")

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "question": question,
        "pdf_name": pdf_name
    })
    pages = sorted(set(f"Page {d.metadata.get('page',0)+1}" for d in docs))
    return answer, pages

def generate_pdf_summary(text, pdf_name):
    """Generate a structured summary for one PDF."""
    llm = get_llm()
    prompt = PromptTemplate.from_template("""
Generate a structured summary for this document called "{pdf_name}".

Document text (first 5000 chars):
{text}

Provide:
## 📄 Document Type
[What kind of document is this]

## 🎯 Main Topic
[Core subject in 2 sentences]

## 🔑 Key Points
[5 most important points as bullets]

## 📊 Notable Details
[Any numbers, dates, names, or specific facts]

## 💡 One-Line Summary
[Single sentence summary]
""")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"pdf_name": pdf_name, "text": text[:5000]})

def compare_two_pdfs(text1, name1, text2, name2):
    """Deep comparison between two PDFs using LLM."""
    llm = get_llm()
    prompt = PromptTemplate.from_template("""
You are an expert document analyst. Compare these two documents deeply.

Document 1 — "{name1}":
{text1}

Document 2 — "{name2}":
{text2}

Generate a thorough comparison:

## 🔍 Overview
[What each document is about in 1-2 lines each]

## ✅ Similarities
[What topics, themes, or content both documents share]

## ⚡ Key Differences
[Most important differences — be specific with examples]

## 📊 Topic Overlap
[Topics that appear in both vs topics unique to each]

## 🏆 Unique Strengths
**{name1}:** [What this doc covers better/uniquely]
**{name2}:** [What this doc covers better/uniquely]

## 💡 Recommendations
[When would you use document 1 vs document 2?]
[What's missing from both?]
""")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "name1": name1, "text1": text1[:3000],
        "name2": name2, "text2": text2[:3000]
    })

def compare_multiple_pdfs(pdf_texts: dict):
    """Find common themes across 3+ PDFs."""
    llm = get_llm()
    docs_section = "\n\n".join(
        f"--- {name} ---\n{text[:1500]}"
        for name, text in pdf_texts.items()
    )
    prompt = PromptTemplate.from_template("""
Analyze these {count} documents together:

{docs}

Generate:

## 🌐 Common Themes Across All Documents
[Themes that appear in ALL or most documents]

## 🗺️ Document Map
[For each document, its unique contribution to the collection]

## 🔗 How They Connect
[How these documents relate to each other]

## 📋 Master Summary
[If someone read all these docs, what's the key takeaway?]

## ❓ Gaps
[What important topic is missing across all documents?]
""")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"count": len(pdf_texts), "docs": docs_section})

def cross_pdf_query(vectorstores: dict, question: str):
    """Query all PDFs and compile answers from each."""
    results = {}
    for pdf_name, vs in vectorstores.items():
        answer, pages = query_single_pdf(vs, question, pdf_name)
        results[pdf_name] = {"answer": answer, "pages": pages}
    return results