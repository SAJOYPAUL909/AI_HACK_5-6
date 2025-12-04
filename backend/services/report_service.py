from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os, json

def generate_pdf_report(filename, results, results_json_path, upload_folder):
    reports_dir = os.path.join(upload_folder, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    doc_id = os.path.splitext(filename)[0]
    # Try to bind document id by searching in results_json_path
    try:
        # results_json_path contains doc_{id}_results.json
        import re
        m = re.search(r"doc_(\d+)_results", os.path.basename(results_json_path))
        doc_num = int(m.group(1)) if m else doc_id
    except:
        doc_num = doc_id

    out_pdf = os.path.join(reports_dir, f"report_{doc_num}.pdf")
    c = canvas.Canvas(out_pdf, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, 750, f"Forensic Report — {filename}")
    c.setFont("Helvetica", 11)
    c.drawString(40, 730, f"Authenticity Score: {results.get('authenticity_score', 'N/A')}")
    # summary
    c.drawString(40, 700, "Summary:")
    text = c.beginText(40, 680)
    summary = results.get("semantic_findings", {}).get("summary", "No summary")
    text.textLines(str(summary))
    c.drawText(text)
    # image findings
    c.drawString(40, 600, "Image Findings:")
    text = c.beginText(40, 580)
    text.textLines(str(results.get("image_findings", {})))
    c.drawText(text)
    # metadata
    c.drawString(40, 500, "Metadata:")
    text = c.beginText(40, 480)
    text.textLines(str(results.get("metadata", {})))
    c.drawText(text)

    c.showPage()
    c.save()
    return out_pdf
