from flask import Blueprint, request, jsonify
from db import SessionLocal
from models import Document
from services.semantic_service import chat_with_document

chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/chat/<int:doc_id>", methods=["POST"])
def chat(doc_id):
    # body: { "question": "Why is date suspicious?" }
    payload = request.json or {}
    q = payload.get("question", "")
    if not q:
        return jsonify({"error": "question required"}), 400

    db = SessionLocal()
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        db.close()
        return jsonify({"error": "document not found"}), 404

    # load results
    import json, os
    try:
        with open(doc.raw_results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    except Exception:
        results = {}

    # pass to semantic/chat service
    reply = chat_with_document(question=q, document_results=results)
    db.close()
    return jsonify({"answer": reply}), 200
