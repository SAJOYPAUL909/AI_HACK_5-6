from PIL import Image
import pytesseract
import os, json

def run_ocr(file_path: str) -> dict:
    """
    Run OCR (pytesseract) on supplied path.
    Returns text, word-level boxes and a naive geometry anomaly score.
    (In production replace with PaddleOCR and better field detection)
    """
    outputs = {"text": "", "words": [], "geometry_anomaly": 0.0}
    try:
        # if pdf -> convert first page to image using pdf2image
        if file_path.lower().endswith(".pdf"):
            from pdf2image import convert_from_path
            pages = convert_from_path(file_path, dpi=200, first_page=1, last_page=1)
            img = pages[0]
        else:
            img = Image.open(file_path).convert("RGB")
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        text = " ".join([w for w in data['text'] if w.strip()])
        outputs["text"] = text
        words = []
        for i, w in enumerate(data['text']):
            if w.strip():
                words.append({
                    "word": w,
                    "left": int(data['left'][i]),
                    "top": int(data['top'][i]),
                    "width": int(data['width'][i]),
                    "height": int(data['height'][i]),
                    "conf": int(data['conf'][i]) if data['conf'][i] else -1
                })
        outputs["words"] = words

        # naive geometry anomaly: low average confidence or many small bounding boxes with overlap
        if words:
            avg_conf = sum([w['conf'] for w in words if w['conf']>0]) / max(1, len(words))
            outputs["geometry_anomaly"] = max(0.0, (60 - avg_conf) / 60)  # 0..1
    except Exception as e:
        outputs["error"] = str(e)
    return outputs
