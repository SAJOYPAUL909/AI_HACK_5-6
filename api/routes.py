from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os

# Import our Modular Core
from rag_core.pdf_loader import PDFLoader
from rag_core.chunker import TokenChunker
from rag_core.vector_store import VectorDB
from rag_core.llm_service import LLMService
from rag_core.optimization import Optimizer
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

api_bp = Blueprint('api', __name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@api_bp.route('/ingest', methods=['POST'])
def ingest():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files['file']
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    try:
        # 1. Load
        raw_docs = PDFLoader.load_pdf(path)
        
        # 2. Token-Based Chunking (The Optimization)
        chunked_docs = TokenChunker.chunk_documents(raw_docs, chunk_size=500)
        
        # 3. Store
        db = VectorDB()
        db.ingest(chunked_docs)
        
        return jsonify({
            "status": "success", 
            "chunks_created": len(chunked_docs),
            "message": "File embedded with TikToken chunking"
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api_bp.route('/generate', methods=['POST'])
def generate():
    data = request.json
    query = data.get('query')
    
    # Generic Configuration (Passed from Frontend)
    llm_config = data.get('llm_config', {}) 
    base_url = llm_config.get('base_url', 'https://openrouter.ai/api/v1') # Default to Ollama
    api_key = llm_config.get('api_key', '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')
    model_name = llm_config.get('model_name', 'meta-llama/llama-3.3-70b-instruct:free')
    print(f"LLM Config - URL: {base_url}, Model: {model_name}")

    try:
        # 1. Initialize Generic LLM
        service = LLMService(api_key, base_url, model_name)
        llm = service.get_llm()

        # 2. Setup Optimized Retrieval (Reranking)
        db = VectorDB()
        optimizer = Optimizer(db.get_store())
        retriever = optimizer.get_retriever(top_k=3)

        # 3. Build Chain
        prompt = ChatPromptTemplate.from_template(
            "Answer solely based on this context:\n\n{context}\n\nQuestion: {input}"
        )
        chain = create_retrieval_chain(
            retriever, 
            create_stuff_documents_chain(llm, prompt)
        )

        # 4. Execute
        response = chain.invoke({"input": query})
        
        return jsonify({
            "answer": response['answer'],
            "source_docs": [d.page_content[:100] + "..." for d in response['context']]
        })

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500