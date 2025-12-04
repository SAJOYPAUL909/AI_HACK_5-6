from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class VectorDB:
    def __init__(self, persist_path="./chroma_db"):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.persist_path = persist_path

    def ingest(self, documents):
        Chroma.from_documents(
            documents=documents, 
            embedding=self.embeddings, 
            persist_directory=self.persist_path
        )

    def get_store(self):
        return Chroma(
            persist_directory=self.persist_path, 
            embedding_function=self.embeddings
        )