from PIL import Image, ImageChops, ImageEnhance
import os, math, json
from utils.image_utils import save_ela_image

def perform_forensics(file_path: str) -> dict:
    """
    Simple forensic heuristics:
    - ELA image to highlight recompression areas.
    - compute an image_score (0..1) where higher means more suspicious.
    - template_score is placeholder.
    """
    out = {"image_score": 0.0, "ela_path": None, "template_score": 0.0, "notes": []}
    try:
        # convert PDF first page if needed
        if file_path.lower().endswith(".pdf"):
            from pdf2image import convert_from_path
            pages = convert_from_path(file_path, dpi=150, first_page=1, last_page=1)
            img = pages[0].convert("RGB")
        else:
            img = Image.open(file_path).convert("RGB")

        # Save ELA image and produce a heuristic score
        ela_path, ela_score = save_ela_image(img, file_path)
        out["ela_path"] = ela_path
        # heuristics: ela_score normalized to 0..1
        out["image_score"] = min(1.0, ela_score / 50.0)
        # template score placeholder (0..1)
        out["template_score"] = 0.0
        if out["image_score"] > 0.4:
            out["notes"].append("ELA indicates potential recompression/editing artifacts.")
    except Exception as e:
        out["error"] = str(e)
    return out
