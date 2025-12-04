from flask import Blueprint, jsonify
from db import SessionLocal
from models import Document

history_bp = Blueprint("history", __name__)

@history_bp.route("/history", methods=["GET"])
def history():
    db = SessionLocal()
    docs = db.query(Document).order_by(Document.upload_time.desc()).all()
    resp = []
    for d in docs:
        resp.append({
            "id": d.id,
            "filename": d.filename,
            "status": d.status,
            "authenticity_score": d.authenticity_score,
            "upload_time": d.upload_time.isoformat()
        })
    db.close()
    return jsonify(resp), 200
