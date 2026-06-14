from dotenv import load_dotenv
load_dotenv()

from rag.vectorstore import create_vectorstore
from rag.chain import build_qa_chain, get_answer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("C:/Users/Hp/OneDrive/Desktop/rag-pdf-chatbot/documents/A Thousand Splindid Suns.pdf")  # ← replace with any PDF
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(pages)

print(f"✅ PDF loaded: {len(pages)} pages, {len(chunks)} chunks")

vectorstore = create_vectorstore(chunks)
print("✅ Vector store created")

chain_and_retriever = build_qa_chain(vectorstore)
answer, sources = get_answer(chain_and_retriever, "What is this document about?")

print(f"\n💬 Answer: {answer}")
print(f"📄 Sources: {sources}")