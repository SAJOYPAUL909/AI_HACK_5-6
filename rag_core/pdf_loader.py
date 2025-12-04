from langchain_community.document_loaders import PyPDFLoader
import os

class PDFLoader:
    @staticmethod
    def load_pdf(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        loader = PyPDFLoader(file_path)
        return loader.load()