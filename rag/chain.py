from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import os

def get_llm(temperature=0):
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",  # upgraded model = better accuracy
        temperature=temperature,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=1024
    )

def build_qa_chain(vectorstore):
    llm = get_llm()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 25, "lambda_mult": 0.7}
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert document analyst with deep reading comprehension.

STRICT RULES:
1. Answer ONLY from the document context provided
2. Be specific — quote details, names, numbers from the document
3. If partially found, answer what you can and note gaps
4. Never invent or assume information
5. For narrative documents: describe events, characters, themes precisely
6. For technical documents: explain concepts clearly with examples from text
7. End every answer with exactly this line:
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
    for msg in chat_history[-8:]:  # last 4 exchanges
        if msg["role"] == "user":
            lc_history.append(HumanMessage(content=msg["content"]))
        else:
            # strip confidence line from history to keep it clean
            content = msg["content"].split("CONFIDENCE:")[0].strip()
            lc_history.append(AIMessage(content=content))

    raw = chain.invoke({"question": question, "chat_history": lc_history})

    # Parse confidence
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
    llm = get_llm(temperature=0.1)
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
• [topic 4]
**BEST QUESTIONS TO ASK:**
1. [Specific question for this document]
2. [Specific question for this document]
3. [Specific question for this document]
4. [Specific question for this document]"""

    response = llm.invoke(prompt)
    return response.content