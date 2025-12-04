from flask import Blueprint, jsonify, request, current_app, send_file
from db import SessionLocal
from models import Document
from services.ocr_service import run_ocr
from services.forensic_service import perform_forensics
from services.metadata_service import extract_metadata
from services.semantic_service import analyze_semantics
from services.scoring_service import compute_score
from services.report_service import generate_pdf_report
import os

analyze_bp = Blueprint("analyze", __name__)

@analyze_bp.route("/analyze/<int:doc_id>", methods=["POST"])
def analyze(doc_id):
    db = SessionLocal()
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        db.close()
        return jsonify({"error": "document not found"}), 404

    try:
        doc.status = "analyzing"
        db.commit()

        # 1. OCR
        ocr_output = run_ocr(doc.stored_path)

        # 2. Metadata
        metadata = extract_metadata(doc.stored_path)

        # 3. Image forensics
        image_findings = perform_forensics(doc.stored_path)

        # 4. Semantic checks (LLM or heuristic)
        semantic_findings = analyze_semantics(ocr_output, metadata)

        # 5. scoring
        score_payload = {
            "geometry": ocr_output.get("geometry_anomaly", 0.0),
            "image": image_findings.get("image_score", 0.0),
            "metadata": metadata.get("metadata_score", 0.0),
            "template": image_findings.get("template_score", 0.0),
            "semantic": semantic_findings.get("semantic_score", 0.0),
        }
        authenticity = compute_score(score_payload)

        # Save results to disk (json) and update DB
        import json, pathlib, time
        results = {
            "ocr": ocr_output,
            "metadata": metadata,
            "image_findings": image_findings,
            "semantic_findings": semantic_findings,
            "score_payload": score_payload,
            "authenticity_score": authenticity
        }
        out_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], "results")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"doc_{doc.id}_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Generate PDF report
        report_path = generate_pdf_report(doc.filename, results, out_path, current_app.config["UPLOAD_FOLDER"])

        doc.status = "done"
        doc.authenticity_score = authenticity
        doc.summary = semantic_findings.get("summary", "")
        doc.raw_results_path = out_path
        db.commit()
        db.refresh(doc)

        db.close()
        return jsonify({"document_id": doc.id, "authenticity_score": authenticity, "report": report_path}), 200

    except Exception as e:
        db.rollback()
        doc.status = "failed"
        db.commit()
        db.close()
        return jsonify({"error": str(e)}), 500

@analyze_bp.route("/report/<int:doc_id>", methods=["GET"])
def get_report(doc_id):
    db = SessionLocal()
    doc = db.query(Document).filter(Document.id == doc_id).first()
    db.close()
    if not doc or not doc.raw_results_path:
        return jsonify({"error": "report not available"}), 404

    # find report pdf in uploads/results or same folder
    report_pdf = os.path.join(current_app.config["UPLOAD_FOLDER"], "reports", f"report_{doc.id}.pdf")
    if os.path.exists(report_pdf):
        return send_file(report_pdf, as_attachment=True)
    return jsonify({"error": "report file missing"}), 404
