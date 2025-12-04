from langchain_text_splitters  import TokenTextSplitter

class TokenChunker:
    @staticmethod
    def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
        """
        Splits text based on TOKEN count, not character count.
        Ensures chunks fit perfectly into LLM Context Context.
        """
        # "cl100k_base" is the standard tokenizer for GPT-4 and most modern LLMs
        splitter = TokenTextSplitter(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        return splitter.split_documents(documents)