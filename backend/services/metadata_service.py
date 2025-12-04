from PyPDF2 import PdfReader
import os

def extract_metadata(file_path: str) -> dict:
    """
    Extracts metadata for PDF files. Returns a metadata dict and a metadata_score heuristic.
    """
    out = {"metadata": {}, "metadata_score": 0.0}
    try:
        if file_path.lower().endswith(".pdf"):
            reader = PdfReader(file_path)
            info = reader.metadata
            md = {}
            for k, v in info.items():
                md[k.replace("/", "")] = str(v)
            out["metadata"] = md
            # simple heuristics: if modifydate exists and differs far from create date -> suspicious
            create = md.get("CreationDate", "")
            mod = md.get("ModDate", "")
            if create and mod and create != mod:
                out["notes"] = ["CreateDate and ModifyDate mismatch."]
                out["metadata_score"] = 0.5
        else:
            out["metadata"] = {}
            out["metadata_score"] = 0.0
    except Exception as e:
        out["error"] = str(e)
    return out
