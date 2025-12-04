import os
from flask import Blueprint, request, current_app, jsonify
from werkzeug.utils import secure_filename
from models import Document
from db import SessionLocal
from services.storage_service import save_file
import db as db_mod

upload_bp = Blueprint("upload", __name__)

def allowed_file(filename):
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in current_app.config["ALLOWED_EXTENSIONS"]

@upload_bp.route("/upload", methods=["POST"])
def upload():
    """
    Accepts multipart file. Saves to storage, creates DB record.
    """
    if "file" not in request.files:
        return jsonify({"error": "file part missing"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "no selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "file type not allowed"}), 400

    filename = secure_filename(file.filename)
    saved_path = save_file(file, filename, current_app.config["UPLOAD_FOLDER"])

    # create DB record
    db = SessionLocal()
    doc = Document(filename=filename, stored_path=saved_path, status="uploaded")
    db.add(doc)
    db.commit()
    db.refresh(doc)
    db.close()

    return jsonify({"document_id": doc.id, "filename": doc.filename}), 201
