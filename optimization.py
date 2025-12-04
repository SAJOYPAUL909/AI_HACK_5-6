# # # # from langchain.retrievers import ContextualCompressionRetriever
# # # # from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
# # # # from langchain_community.cross_encoders import CrossEncoderReranker
# # # # from langchain_community.document_transformers import CrossEncoderReranker
# # # # from langchain_community.document_transformers import CrossEncoderReranker
# # # # from langchain.retrievers import ContextualCompressionRetriever
# # # from langchain_community.document_transformers import CrossEncoderReranker


# # # # from langchain.retrievers import ContextualCompressionRetriever
# # # from langchain.retrievers.document_compressors import CrossEncoderReranker
# # # from rag_core.cross_encoder_reranker import CrossEncoderReranker

# # # from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# # # # from langchain_community.retrievers import ContextualCompressionRetriever
# # # # from langchain_community.document_transformers import CrossEncoderReranker
# # # # from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# # # # 1. Compression Retriever
# # # # from langchain.retrievers import ContextualCompressionRetriever

# # # # 2. Re-ranker (Community package)
# # # # from langchain_community.document_compressors import CrossEncoderReranker

# # # # 3. CrossEncoder Model (HuggingFace)
# # # from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# # # from langchain_community.document_compressors import CrossEncoderReranker
# # # from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# # # class Optimizer:
# # #     def __init__(self, vector_store):
# # #         self.vector_store = vector_store
# # #         # This model is specifically trained to grade "Question <-> Answer" relevance
# # #         self.model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

# # #     def get_optimized_retriever(self, top_k=10, final_top_n=3):
# # #         """
# # #         1. Fetches top_k (e.g., 10) docs using fast vector search.
# # #         2. Uses CrossEncoder to 'read' and score them.
# # #         3. Returns only the top_n (e.g., 3) best matches to the LLM.
# # #         """
# # #         base_retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        
# # #         compressor = CrossEncoderReranker(model=self.model, top_n=final_top_n)
        
# # #         # compression_retriever = ContextualCompressionRetriever(
# # #         #     base_compressor=compressor, 
# # #         #     base_retriever=base_retriever
# # #         # )
        
# # #         return base_retriever

# # # iptimation.py

# # from langchain_community.cross_encoders import HuggingFaceCrossEncoder
# # # from langchain_community.document_compressors import CrossEncoderReranker
# # # from langchain.retrievers import ContextualCompressionRetriever
# # from langchain_community.document_compressors import CrossEncoderReranker
# # from langchain_community.retrievers import ContextualCompressionRetriever

# # class Optimizer:
# #     def __init__(self, vector_store):
# #         self.vector_store = vector_store

# #         # Load HuggingFace Cross Encoder
# #         # Best for Question <-> Answer relevance scoring
# #         self.model = HuggingFaceCrossEncoder(
# #             model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
# #         )

# #     def get_optimized_retriever(self, top_k=10, final_top_n=3):
# #         """
# #         Workflow:
# #         1. Retrieve top_k vectors using fast similarity search.
# #         2. Re-rank them using CrossEncoder.
# #         3. Return a ContextualCompressionRetriever which outputs top_n reranked docs.
# #         """

# #         # Step 1: Base vector retriever
# #         base_retriever = self.vector_store.as_retriever(
# #             search_kwargs={"k": top_k}
# #         )

# #         # Step 2: CrossEncoder re-ranker
# #         compressor = CrossEncoderReranker(
# #             model=self.model,
# #             top_n=final_top_n
# #         )

# #         # Step 3: Contextual compression retriever
# #         compression_retriever = ContextualCompressionRetriever(
# #             base_compressor=compressor,
# #             base_retriever=base_retriever,
# #         )

# #         return compression_retriever

# # optimization.py

# from langchain_community.cross_encoders import HuggingFaceCrossEncoder


# class Optimizer:
#     def __init__(self, vector_store):
#         self.vector_store = vector_store

#         # Lightweight cross encoder for relevance scoring
#         self.model = HuggingFaceCrossEncoder(
#             model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
#         )

#     def rerank(self, query, docs, top_n=3):
#         """
#         Manually rerank the documents using CrossEncoder.
#         """

#         # Prepare pairs: (query, doc_text)
#         pairs = [(query, doc.page_content) for doc in docs]

#         # Cross-encoder scores
#         scores = self.model.predict(pairs)

#         # Attach scores to docs
#         scored_docs = list(zip(scores, docs))

#         # Sort high to low
#         scored_docs.sort(key=lambda x: x[0], reverse=True)

#         # Return top_n docs only
#         return [doc for _, doc in scored_docs[:top_n]]

#     def get_optimized_retriever(self, top_k=10, final_top_n=3):
#         """
#         Returns a callable function that performs:
#         1. Vector store retrieval
#         2. Manual cross-encoder reranking
#         """

#         base_retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})

#         def optimized(query):
#             # Step 1: retrieve top_k docs
#             docs = base_retriever.get_relevant_documents(query)

#             # Step 2: manually rerank using cross encoder
#             reranked_docs = self.rerank(query, docs, top_n=final_top_n)

#             return reranked_docs

#         return optimized

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

class Optimizer:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.reranker = HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(self, query, docs, top_n=3):
        """Manually rerank without LangChain community compressors."""
        pairs = [(query, d.page_content) for d in docs]
        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, score in ranked[:top_n]]

    def get_retriever(self, top_k=10):
        return self.vector_store.as_retriever(search_kwargs={"k": top_k})


def build_rag_chain(llm, optimizer: Optimizer):
    """Correct LCEL chain — no create_retrieval_chain, no with_config() """

    retriever = optimizer.get_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert assistant. Answer based on the context."),
        ("human", "Query: {query}\n\nContext:\n{context}")
    ])

    return (
        {
            "query": lambda x: x["query"],
            "context": lambda x: rerank_wrapper(x["query"], retriever)
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def rerank_wrapper(query, retriever):
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)
