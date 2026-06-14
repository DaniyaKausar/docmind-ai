from langchain_community.retrievers import BM25Retriever

def create_hybrid_retriever(vectorstore, chunks):
    """
    Combines BM25 (keyword) + ChromaDB (semantic) manually.
    No EnsembleRetriever needed!
    """
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4

    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.7}
    )

    class HybridRetriever:
        def invoke(self, query):
            # Get results from both
            bm25_docs = bm25_retriever.invoke(query)
            vector_docs = vector_retriever.invoke(query)

            # Combine and deduplicate
            seen = set()
            combined = []
            # 60% weight to vector, 40% to BM25
            for doc in vector_docs + bm25_docs:
                key = doc.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    combined.append(doc)

            return combined[:6]  # top 6 results

    return HybridRetriever()