from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from rag.hybrid_search import create_hybrid_retriever
import os

def get_llm(temperature=0):
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=temperature,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=1024
    )

def build_qa_chain(vectorstore, chunks=None):
    llm = get_llm()

    # Use hybrid if chunks available, else fallback to MMR
    if chunks:
        retriever = create_hybrid_retriever(vectorstore, chunks)
    else:
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7}
        )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert document analyst with deep reading comprehension.

STRICT RULES:
1. Answer ONLY from the document context provided
2. Be specific — quote details, names, numbers from the document
3. If partially found, answer what you can and note gaps
4. Never invent or assume information
5. End every answer with exactly:
   CONFIDENCE: [High/Medium/Low] — [reason in one sentence]

High = answer clearly in context
Medium = partially found or inferred
Low = barely found or not present

Document Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(
            f"[Page {doc.metadata.get('page', '?') + 1}]: {doc.page_content}"
            for doc in docs
        )

    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", [])
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def get_answer(chain_and_retriever, question, chat_history=None):
    chain, retriever = chain_and_retriever
    if chat_history is None:
        chat_history = []

    lc_history = []
    for msg in chat_history[-8:]:
        if msg["role"] == "user":
            lc_history.append(HumanMessage(content=msg["content"]))
        else:
            content = msg["content"].split("CONFIDENCE:")[0].strip()
            lc_history.append(AIMessage(content=content))

    raw = chain.invoke({"question": question, "chat_history": lc_history})

    confidence = "Medium"
    confidence_reason = ""
    answer = raw

    if "CONFIDENCE:" in raw:
        parts = raw.split("CONFIDENCE:")
        answer = parts[0].strip()
        conf_line = parts[1].strip()
        if conf_line.startswith("High"):
            confidence = "High"
        elif conf_line.startswith("Low"):
            confidence = "Low"
        confidence_reason = conf_line.split("—")[-1].strip() if "—" in conf_line else conf_line

    source_docs = retriever.invoke(question)
    sources = sorted(set(
        f"Page {doc.metadata.get('page', 0) + 1}"
        for doc in source_docs
    ))

    return answer, sources, confidence, confidence_reason


def generate_document_summary(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 12, "fetch_k": 35}
    )

    sample_docs = retriever.invoke(
        "main topics overview summary introduction conclusion key themes"
    )
    context = "\n\n".join(
        f"[Page {doc.metadata.get('page', '?') + 1}]: {doc.page_content}"
        for doc in sample_docs
    )

    prompt = f"""Analyze this document and provide a structured briefing.

Document excerpts:
{context}

Respond in this exact format:

**TYPE:** [Document type]
**ABOUT:** [2-3 sentence overview]
**KEY TOPICS:**
• [topic 1]
• [topic 2]
• [topic 3]
**BEST QUESTIONS TO ASK:**
1. [Specific question for this document]
2. [Specific question for this document]
3. [Specific question for this document]
4. [Specific question for this document]"""

    fresh_llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=1024
    )
    response = fresh_llm.invoke(prompt)
    return response.content