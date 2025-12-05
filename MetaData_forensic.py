# """
# FastAPI backend implementing the 4-layer tamper-detection pipeline (OCR + Region compare + ELA + Metadata + lightweight semantic checks).

# Single-file example: `fastapi_tamper_backend.py`

# Usage:
#   1. Install system deps: Tesseract OCR and poppler (for pdf2image)
#      - Ubuntu/Debian: sudo apt install tesseract-ocr poppler-utils
#      - Windows: install Tesseract from https://github.com/tesseract-ocr/tesseract and add to PATH
#      - Poppler for Windows: https://github.com/oschwartz10612/poppler-windows

#   2. Python packages (recommended in a venv):
#      pip install fastapi uvicorn python-multipart Pillow pytesseract pdf2image PyPDF2 exifread python-dateutil

#   3. Run:
#      uvicorn fastapi_tamper_backend:app --reload --port 8000

# Endpoints:
#   POST /analyze  - form-data with `gold` and `suspect` files (each PDF or image). Returns JSON insight plus ELA images as data-urls.

# Notes:
#   - This is a prototype meant to be production-hardened: security, rate-limiting, pagination (multi-page), large file handling and using PaddleOCR (or a hosted OCR) should be considered.
# """

# from pathlib import Path
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from typing import List, Dict, Any, Tuple
# from io import BytesIO
# from PIL import Image, ImageChops
# import pytesseract
# import tempfile
# import base64
# import math
# import os
# import json
# # from pdf2image import convert_from_bytes
# from pdf2image import convert_from_bytes, exceptions as pdf2image_exceptions

# from PyPDF2 import PdfReader
# import exifread
# from dateutil import parser as dateparser
# from datetime import datetime, time
# import re

# app = FastAPI(title="Tamper-Detect Backend (FastAPI)")

# # ------------------------------- Utilities -------------------------------

# def file_is_pdf(content_type: str, filename: str) -> bool:
#     return content_type == 'application/pdf' or filename.lower().endswith('.pdf')

# def read_first_page_image(file_bytes: bytes) -> Image.Image:
#     # quick magic check
#     head = file_bytes[:16]
#     if head.startswith(b'%PDF-'):
#         # looks like a PDF
#         try:
#             images = convert_from_bytes(file_bytes, first_page=1, last_page=1, dpi=200)
#             if images and len(images) > 0:
#                 return images[0].convert('RGB')
#             raise RuntimeError('pdf2image returned no pages')
#         except pdf2image_exceptions.PDFInfoNotInstalledError:
#             raise RuntimeError('Poppler (pdftoppm) missing — install poppler and ensure pdftoppm is on PATH')
#         except Exception as e:
#             # save debug file
#             tmp = Path('./debug_uploads')
#             tmp.mkdir(exist_ok=True)
#             p = tmp / f'bad_pdf_{int(time.time())}.pdf'
#             p.write_bytes(file_bytes)
#             raise RuntimeError(f'Failed to render PDF first page: {e}. Saved sample to {p}')
#     # check common image signatures (PNG, JPG, TIFF)
#     if head.startswith(b'\\x89PNG') or head[0:2] == b'\\xff\\xd8' or head.startswith(b'II') or head.startswith(b'MM'):
#         try:
#             img = Image.open(BytesIO(file_bytes)).convert('RGB')
#             return img
#         except Exception as e:
#             tmp = Path('./debug_uploads'); tmp.mkdir(exist_ok=True)
#             p = tmp / f'bad_image_{int(time.time())}.bin'
#             p.write_bytes(file_bytes)
#             raise RuntimeError(f'Image open failed: {e}. Saved sample to {p}')
#     # unknown type -> save and report
#     tmp = Path('./debug_uploads'); tmp.mkdir(exist_ok=True)
#     p = tmp / f'unknown_upload_{int(time.time())}.bin'
#     p.write_bytes(file_bytes)
#     raise RuntimeError(f'Unsupported or unknown file type (magic bytes: {head!r}). Saved sample to {p}')


# def extract_pdf_metadata(file_bytes: bytes) -> Dict[str, Any]:
#     try:
#         reader = PdfReader(BytesIO(file_bytes))
#         info = reader.metadata or {}
#         # Convert to dict with simple keys
#         meta = {}
#         for k, v in info.items():
#             # PyPDF2 returns keys like '/Producer'
#             kname = k[1:] if isinstance(k, str) and k.startswith('/') else k
#             try:
#                 meta[kname] = str(v)
#             except Exception:
#                 meta[kname] = v
#         # Add number of pages
#         meta['num_pages'] = len(reader.pages)
#         return {'type': 'pdf', 'raw': meta}
#     except Exception as e:
#         return {'type': 'pdf', 'error': str(e)}


# def extract_image_metadata(file_bytes: bytes) -> Dict[str, Any]:
#     try:
#         f = BytesIO(file_bytes)
#         tags = exifread.process_file(f, details=False)
#         meta = {k: str(v) for k, v in tags.items()}
#         return {'type': 'image', 'raw': meta}
#     except Exception as e:
#         return {'type': 'image', 'error': str(e)}


# def extract_metadata(upload: UploadFile, file_bytes: bytes) -> Dict[str, Any]:
#     if file_is_pdf(upload.content_type, upload.filename):
#         return extract_pdf_metadata(file_bytes)
#     else:
#         return extract_image_metadata(file_bytes)


# # ------------------------------- ELA -------------------------------

# def compute_ela(image: Image.Image, quality: int = 90, grid_size: Tuple[int, int] = (32, 32)) -> Dict[str, Any]:
#     """Compute Error Level Analysis heatmap and return a small grid of scores & dataURL of diff image."""
#     # Recompress image at lower quality
#     buffer = BytesIO()
#     image.save(buffer, format='JPEG', quality=quality)
#     buffer.seek(0)
#     recompressed = Image.open(buffer).convert('RGB')

#     # Resize recompressed to original size if mismatch
#     if recompressed.size != image.size:
#         recompressed = recompressed.resize(image.size)

#     diff = ImageChops.difference(image, recompressed)
#     # convert to grayscale for intensity
#     gray = diff.convert('L')

#     # coarse grid aggregation
#     gx, gy = grid_size
#     w, h = image.size
#     cell_w = max(1, w // gx)
#     cell_h = max(1, h // gy)
#     grid = []
#     for y in range(gy):
#         row = []
#         for x in range(gx):
#             sx = x * cell_w
#             sy = y * cell_h
#             box = (sx, sy, min(w, sx + cell_w), min(h, sy + cell_h))
#             region = gray.crop(box)
#             # compute mean brightness
#             stat = list(region.getdata())
#             avg = sum(stat) / len(stat) if len(stat) else 0
#             row.append(avg)
#         grid.append(row)

#     # data url for visual debugging (small thumbnail)
#     thumb = diff.copy()
#     max_thumb = 1200
#     if max(thumb.size) > max_thumb:
#         thumb.thumbnail((max_thumb, max_thumb))
#     outb = BytesIO()
#     thumb.save(outb, format='PNG')
#     data_url = 'data:image/png;base64,' + base64.b64encode(outb.getvalue()).decode('ascii')

#     # basic stats
#     flat = [v for row in grid for v in row]
#     mean = sum(flat) / len(flat) if len(flat) else 0
#     variance = sum((v - mean) ** 2 for v in flat) / len(flat) if len(flat) else 0
#     std = math.sqrt(variance)

#     return {
#         'grid': grid,
#         'mean': mean,
#         'std': std,
#         'preview': data_url
#     }


# # ------------------------------- OCR -------------------------------

# def ocr_with_pytesseract(image: Image.Image) -> Dict[str, Any]:
#     """Use pytesseract to extract words with bounding boxes and confidence.
#     Returns: { full_text, words: [{text, left, top, width, height, conf}] }
#     """
#     # pytesseract image_to_data
#     try:
#         data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
#     except Exception as e:
#         raise RuntimeError(f'pytesseract failed: {e}')
#     words = []
#     n = len(data.get('text', []))
#     for i in range(n):
#         text = data['text'][i]
#         if not text or text.strip() == '':
#             continue
#         words.append({
#             'text': text,
#             'left': int(data['left'][i]),
#             'top': int(data['top'][i]),
#             'width': int(data['width'][i]),
#             'height': int(data['height'][i]),
#             'conf': float(data['conf'][i]) if data['conf'][i] != '-1' else None
#         })
#     full = "\n".join([w['text'] for w in words])
#     return {'full_text': full, 'words': words}


# # ------------------------------- OCR Comparison Heuristics -------------------------------

# def levenshtein(a: str, b: str) -> int:
#     if a == b:
#         return 0
#     if len(a) == 0:
#         return len(b)
#     if len(b) == 0:
#         return len(a)
#     v0 = [i for i in range(len(b) + 1)]
#     v1 = [0] * (len(b) + 1)
#     for i in range(len(a)):
#         v1[0] = i + 1
#         for j in range(len(b)):
#             cost = 0 if a[i] == b[j] else 1
#             v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
#         v0, v1 = v1, v0
#     return v0[len(b)]


# def compare_ocr(gold: Dict[str, Any], suspect: Dict[str, Any]) -> List[Dict[str, Any]]:
#     findings = []
#     gold_words = gold.get('words', [])
#     suspect_words = suspect.get('words', [])

#     # Quick maps
#     suspect_text_map = {}
#     for i, s in enumerate(suspect_words):
#         suspect_text_map.setdefault(s['text'], []).append((i, s))

#     # Compare each gold token to best suspect candidate
#     for i, g in enumerate(gold_words):
#         best = {'idx': -1, 'dist': 9999, 's': None}
#         for j, s in enumerate(suspect_words):
#             d = levenshtein(g['text'], s['text'])
#             if d < best['dist']:
#                 best = {'idx': j, 'dist': d, 's': s}
#         if best['idx'] == -1:
#             continue
#         s = best['s']
#         if g['text'] != s['text']:
#             # numeric heuristics
#             if re.fullmatch(r'[\d\-\/\.:]{1,20}', g['text']) or re.fullmatch(r'[\d\-\/\.:]{1,20}', s['text']):
#                 gw = g.get('width')
#                 sw = s.get('width')
#                 width_change_pct = None
#                 if gw and sw:
#                     try:
#                         width_change_pct = ((sw - gw) / gw) * 100.0
#                     except Exception:
#                         width_change_pct = None
#                 overlap = None
#                 if gw and sw and 'left' in g and 'left' in s:
#                     gx0, gx1 = g['left'], g['left'] + g['width']
#                     sx0, sx1 = s['left'], s['left'] + s['width']
#                     overlap = max(0, min(gx1, sx1) - max(gx0, sx0))
#                 if width_change_pct is not None and width_change_pct > 15:
#                     findings.append({
#                         'type': 'inserted_digit',
#                         'reason': f'bbox width increased by {width_change_pct:.1f}%',
#                         'gold': g['text'], 'suspect': s['text'], 'gold_bbox': {k:g[k] for k in ('left','top','width','height')}, 'suspect_bbox': {k:s[k] for k in ('left','top','width','height')}
#                     })
#                 else:
#                     findings.append({ 'type': 'numeric_mismatch', 'gold': g['text'], 'suspect': s['text'], 'lev': best['dist'] })
#             else:
#                 findings.append({ 'type': 'token_changed', 'gold': g['text'], 'suspect': s['text'], 'lev': best['dist'] })
#         else:
#             # Same token; check bbox width anomaly
#             if 'width' in g and 'width' in s and g['width'] > 0:
#                 pct = ((s['width'] - g['width']) / g['width']) * 100.0
#                 if abs(pct) > 15:
#                     findings.append({ 'type': 'bbox_anomaly', 'token': g['text'], 'change_pct': pct, 'gold_bbox': {k:g[k] for k in ('left','top','width','height')}, 'suspect_bbox': {k:s[k] for k in ('left','top','width','height')} })

#     # tokens present only in suspect or only in gold
#     gold_texts = set([w['text'] for w in gold_words])
#     suspect_texts = set([w['text'] for w in suspect_words])
#     for tok in suspect_texts - gold_texts:
#         findings.append({ 'type': 'suspect_only', 'token': tok })
#     for tok in gold_texts - suspect_texts:
#         findings.append({ 'type': 'gold_only', 'token': tok })

#     return findings


# # ------------------------------- Lightweight Semantic Checks -------------------------------

# def extract_dates_from_text(text: str) -> List[datetime]:
#     # naive regex for dates in common formats; then parse with dateutil
#     date_patterns = [r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b', r'\b\d{4}[\-]\d{1,2}[\-]\d{1,2}\b']
#     dates = []
#     for pat in date_patterns:
#         for m in re.findall(pat, text):
#             try:
#                 d = dateparser.parse(m, dayfirst=True)
#                 if d:
#                     dates.append(d)
#             except Exception:
#                 pass
#     return dates


# def semantic_checks(gold_text: str, suspect_text: str) -> List[Dict[str, Any]]:
#     findings = []
#     # 1) Future date detection (compared to now)
#     now = datetime.utcnow()
#     s_dates = extract_dates_from_text(suspect_text)
#     for d in s_dates:
#         if d > now and (d - now).days > 1:
#             findings.append({ 'type': 'future_date', 'date': d.isoformat(), 'reason': 'suspect contains date in the future vs server now' })

#     # 2) Inconsistency: amounts mentioned vs other fields (naive numeric checks)
#     gold_nums = re.findall(r'\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\b', gold_text)
#     suspect_nums = re.findall(r'\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\b', suspect_text)
#     # if suspect has numbers not in gold, flag
#     for n in suspect_nums:
#         if n not in gold_nums:
#             findings.append({ 'type': 'numeric_inconsistency', 'number': n })

#     # 3) Prohibited words
#     prohibited = ['guarantee', 'assured return', 'assured-return', 'guaranteed return']
#     for p in prohibited:
#         if p in suspect_text.lower():
#             findings.append({ 'type': 'prohibited_word', 'word': p })

#     # 4) Simple contradiction detection: if suspect contains "signed on <date>" but date differs from date field
#     # (requires more structured field mapping; left as placeholder)

#     return findings


# # ------------------------------- Main Endpoint -------------------------------

# @app.post('/analyze')
# async def analyze(gold: UploadFile = File(...), suspect: UploadFile = File(...)):
#     # read bytes
#     gold_bytes = await gold.read()
#     suspect_bytes = await suspect.read()

#     # safety: size limits (e.g., 20MB)
#     max_size = 20 * 1024 * 1024
#     if len(gold_bytes) > max_size or len(suspect_bytes) > max_size:
#         raise HTTPException(status_code=413, detail='One of the files exceeds size limit (20MB)')

#     # 1) metadata
#     gold_meta = extract_metadata(gold, gold_bytes)
#     suspect_meta = extract_metadata(suspect, suspect_bytes)

#     # 2) render to images (first page)
#     try:
#         gold_img = read_first_page_image(gold_bytes)
#         suspect_img = read_first_page_image(suspect_bytes)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f'Failed to render files to images: {e}')

#     # 3) compute ELA
#     gold_ela = compute_ela(gold_img, quality=60)
#     suspect_ela = compute_ela(suspect_img, quality=60)

#     # 4) OCR
#     try:
#         gold_ocr = ocr_with_pytesseract(gold_img)
#         suspect_ocr = ocr_with_pytesseract(suspect_img)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f'OCR failed: {e}')

#     # 5) compare OCR
#     ocr_findings = compare_ocr(gold_ocr, suspect_ocr)

#     # 6) semantic checks
#     sem_findings = semantic_checks(gold_ocr.get('full_text', ''), suspect_ocr.get('full_text', ''))

#     # 7) metadata heuristics
#     meta_findings = []
#     try:
#         # check suspicious tools in suspect metadata
#         sm_raw = json.dumps(suspect_meta.get('raw', suspect_meta), default=str).lower()
#         suspicious_tools = ['adobe', 'photoshop', 'foxit', 'illustrator', 'pdfelement', 'pdf editor', 'pdf-xchange']
#         for t in suspicious_tools:
#             if t in sm_raw:
#                 meta_findings.append({ 'type': 'suspicious_tool_detected', 'tool': t })
#         # check modify vs create
#         gold_raw = gold_meta.get('raw', {})
#         suspect_raw = suspect_meta.get('raw', {})
#         def parse_possible_date(x):
#             for key in ['ModDate', 'ModifyDate', 'CreationDate', 'CreateDate', 'Producer']:
#                 if isinstance(x, dict) and key in x:
#                     try:
#                         return dateparser.parse(str(x[key]))
#                     except Exception:
#                         pass
#             return None
#         s_mod = parse_possible_date(suspect_raw)
#         g_create = parse_possible_date(gold_raw)
#         if s_mod and g_create and s_mod > g_create:
#             meta_findings.append({ 'type': 'modify_after_create', 'suspect_modify': s_mod.isoformat(), 'gold_create': g_create.isoformat() })
#     except Exception as e:
#         meta_findings.append({ 'type': 'meta_check_error', 'error': str(e) })

#     # 8) summarise flags
#     summary_flags = []
#     if any(f for f in meta_findings if f.get('type') == 'suspicious_tool_detected'):
#         summary_flags.append('suspicious_pdf_tool_detected')
#     if any(f for f in ocr_findings if f.get('type') in ('inserted_digit','suspect_only')):
#         summary_flags.append('possible_inserted_digit_or_token')
#     # ELA hotspot compare: count hotspots above mean+std
#     def ela_hotspot_count(ela):
#         flat = [v for row in ela['grid'] for v in row]
#         if not flat:
#             return 0
#         mean = sum(flat)/len(flat)
#         std = math.sqrt(sum((x-mean)**2 for x in flat)/len(flat))
#         thr = max(mean + 1.5*std, mean*1.8)
#         return sum(1 for x in flat if x > thr)
#     g_hot = ela_hotspot_count(gold_ela)
#     s_hot = ela_hotspot_count(suspect_ela)
#     if s_hot > g_hot + 2:
#         summary_flags.append('suspect_has_more_ela_hotspots')

#     # prepare JSON-safe response (trim heavy image arrays)
#     response = {
#         'generated_at': datetime.utcnow().isoformat() + 'Z',
#         'files': {
#             'gold': { 'filename': gold.filename, 'content_type': gold.content_type, 'size': len(gold_bytes) },
#             'suspect': { 'filename': suspect.filename, 'content_type': suspect.content_type, 'size': len(suspect_bytes) }
#         },
#         'metadata': { 'gold': gold_meta, 'suspect': suspect_meta },
#         'ocr': {
#             'gold': { 'full_text': gold_ocr.get('full_text'), 'words_count': len(gold_ocr.get('words', [])) },
#             'suspect': { 'full_text': suspect_ocr.get('full_text'), 'words_count': len(suspect_ocr.get('words', [])) },
#             'findings': ocr_findings
#         },
#         'ela': { 'gold': { 'mean': gold_ela['mean'], 'std': gold_ela['std'], 'preview': gold_ela['preview'] }, 'suspect': { 'mean': suspect_ela['mean'], 'std': suspect_ela['std'], 'preview': suspect_ela['preview'] } },
#         'semantic_findings': sem_findings,
#         'metadata_findings': meta_findings,
#         'summary_flags': summary_flags
#     }

#     return JSONResponse(content=response)


# # ------------------------------- Run note -------------------------------
# # Start with: uvicorn fastapi_tamper_backend:app --reload --port 8000
# # Test with curl:
# # curl -X POST -F "gold=@/path/to/original.pdf" -F "suspect=@/path/to/suspect.pdf" http://127.0.0.1:8000/analyze

"""
FastAPI backend implementing the 4-layer tamper-detection pipeline (OCR + Region compare + ELA + Metadata + lightweight semantic checks).

Single-file example: `fastapi_tamper_backend.py`

Usage:
  1. Install system deps: Tesseract OCR and poppler (for pdf2image)
     - Ubuntu/Debian: sudo apt install tesseract-ocr poppler-utils
     - Windows: install Tesseract from https://github.com/tesseract-ocr/tesseract and add to PATH
     - Poppler for Windows: https://github.com/oschwartz10612/poppler-windows

  2. Python packages (recommended in a venv):
     pip install fastapi uvicorn python-multipart Pillow pytesseract pdf2image PyPDF2 exifread python-dateutil

  3. Run:
     uvicorn fastapi_tamper_backend:app --reload --port 8000

Endpoints:
  POST /analyze  - form-data with `gold` and `suspect` files (each PDF or image). Returns JSON insight plus ELA images as data-urls.

Notes:
  - This is a prototype meant to be production-hardened: security, rate-limiting, pagination (multi-page), large file handling and using PaddleOCR (or a hosted OCR) should be considered.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple
from io import BytesIO
from PIL import Image, ImageChops
import pytesseract
import tempfile
import base64
import math
import os
import json
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader
import exifread
from dateutil import parser as dateparser
from datetime import datetime
import re

app = FastAPI(title="Tamper-Detect Backend (FastAPI)")

# ------------------------------- Utilities -------------------------------

def file_is_pdf(content_type: str, filename: str) -> bool:
    return content_type == 'application/pdf' or filename.lower().endswith('.pdf')


def read_first_page_image(file_bytes: bytes) -> Image.Image:
    """Render PDF first page (if PDF) to PIL Image, or load image bytes."""
    # Try PDF first
    try:
        images = convert_from_bytes(file_bytes, first_page=1, last_page=1, dpi=200)
        if images and len(images) > 0:
            return images[0].convert('RGB')
    except Exception:
        pass
    # Fallback: image
    try:
        img = Image.open(BytesIO(file_bytes)).convert('RGB')
        return img
    except Exception as e:
        raise ValueError(f'Unsupported file format or corrupt image/pdf: {e}')


def extract_pdf_metadata(file_bytes: bytes) -> Dict[str, Any]:
    try:
        reader = PdfReader(BytesIO(file_bytes))
        info = reader.metadata or {}
        # Convert to dict with simple keys
        meta = {}
        for k, v in info.items():
            # PyPDF2 returns keys like '/Producer'
            kname = k[1:] if isinstance(k, str) and k.startswith('/') else k
            try:
                meta[kname] = str(v)
            except Exception:
                meta[kname] = v
        # Add number of pages
        meta['num_pages'] = len(reader.pages)
        return {'type': 'pdf', 'raw': meta}
    except Exception as e:
        return {'type': 'pdf', 'error': str(e)}


def extract_image_metadata(file_bytes: bytes) -> Dict[str, Any]:
    try:
        f = BytesIO(file_bytes)
        tags = exifread.process_file(f, details=False)
        meta = {k: str(v) for k, v in tags.items()}
        return {'type': 'image', 'raw': meta}
    except Exception as e:
        return {'type': 'image', 'error': str(e)}


def extract_metadata(upload: UploadFile, file_bytes: bytes) -> Dict[str, Any]:
    if file_is_pdf(upload.content_type, upload.filename):
        return extract_pdf_metadata(file_bytes)
    else:
        return extract_image_metadata(file_bytes)


# ------------------------------- ELA -------------------------------

def compute_ela(image: Image.Image, quality: int = 90, grid_size: Tuple[int, int] = (32, 32)) -> Dict[str, Any]:
    """Compute Error Level Analysis heatmap and return a small grid of scores & dataURL of diff image."""
    # Recompress image at lower quality
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert('RGB')

    # Resize recompressed to original size if mismatch
    if recompressed.size != image.size:
        recompressed = recompressed.resize(image.size)

    diff = ImageChops.difference(image, recompressed)
    # convert to grayscale for intensity
    gray = diff.convert('L')

    # coarse grid aggregation
    gx, gy = grid_size
    w, h = image.size
    cell_w = max(1, w // gx)
    cell_h = max(1, h // gy)
    grid = []
    for y in range(gy):
        row = []
        for x in range(gx):
            sx = x * cell_w
            sy = y * cell_h
            box = (sx, sy, min(w, sx + cell_w), min(h, sy + cell_h))
            region = gray.crop(box)
            # compute mean brightness
            stat = list(region.getdata())
            avg = sum(stat) / len(stat) if len(stat) else 0
            row.append(avg)
        grid.append(row)

    # data url for visual debugging (small thumbnail)
    thumb = diff.copy()
    max_thumb = 1200
    if max(thumb.size) > max_thumb:
        thumb.thumbnail((max_thumb, max_thumb))
    outb = BytesIO()
    thumb.save(outb, format='PNG')
    data_url = 'data:image/png;base64,' + base64.b64encode(outb.getvalue()).decode('ascii')

    # basic stats
    flat = [v for row in grid for v in row]
    mean = sum(flat) / len(flat) if len(flat) else 0
    variance = sum((v - mean) ** 2 for v in flat) / len(flat) if len(flat) else 0
    std = math.sqrt(variance)

    return {
        'grid': grid,
        'mean': mean,
        'std': std,
        'preview': data_url
    }


# ------------------------------- OCR -------------------------------

def ocr_with_pytesseract(image: Image.Image) -> Dict[str, Any]:
    """Use pytesseract to extract words with bounding boxes and confidence.
    Returns: { full_text, words: [{text, left, top, width, height, conf}] }
    """
    # pytesseract image_to_data
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception as e:
        raise RuntimeError(f'pytesseract failed: {e}')
    words = []
    n = len(data.get('text', []))
    for i in range(n):
        text = data['text'][i]
        if not text or text.strip() == '':
            continue
        words.append({
            'text': text,
            'left': int(data['left'][i]),
            'top': int(data['top'][i]),
            'width': int(data['width'][i]),
            'height': int(data['height'][i]),
            'conf': float(data['conf'][i]) if data['conf'][i] != '-1' else None
        })
    full = "\n".join([w['text'] for w in words])
    return {'full_text': full, 'words': words}


# ------------------------------- OCR Comparison Heuristics -------------------------------

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    v0 = [i for i in range(len(b) + 1)]
    v1 = [0] * (len(b) + 1)
    for i in range(len(a)):
        v1[0] = i + 1
        for j in range(len(b)):
            cost = 0 if a[i] == b[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0, v1 = v1, v0
    return v0[len(b)]


def compare_ocr(gold: Dict[str, Any], suspect: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings = []
    gold_words = gold.get('words', [])
    suspect_words = suspect.get('words', [])

    # Quick maps
    suspect_text_map = {}
    for i, s in enumerate(suspect_words):
        suspect_text_map.setdefault(s['text'], []).append((i, s))

    # Compare each gold token to best suspect candidate
    for i, g in enumerate(gold_words):
        best = {'idx': -1, 'dist': 9999, 's': None}
        for j, s in enumerate(suspect_words):
            d = levenshtein(g['text'], s['text'])
            if d < best['dist']:
                best = {'idx': j, 'dist': d, 's': s}
        if best['idx'] == -1:
            continue
        s = best['s']
        if g['text'] != s['text']:
            # numeric heuristics
            if re.fullmatch(r'[\d\-\/\.:]{1,20}', g['text']) or re.fullmatch(r'[\d\-\/\.:]{1,20}', s['text']):
                gw = g.get('width')
                sw = s.get('width')
                width_change_pct = None
                if gw and sw:
                    try:
                        width_change_pct = ((sw - gw) / gw) * 100.0
                    except Exception:
                        width_change_pct = None
                overlap = None
                if gw and sw and 'left' in g and 'left' in s:
                    gx0, gx1 = g['left'], g['left'] + g['width']
                    sx0, sx1 = s['left'], s['left'] + s['width']
                    overlap = max(0, min(gx1, sx1) - max(gx0, sx0))
                if width_change_pct is not None and width_change_pct > 15:
                    findings.append({
                        'type': 'inserted_digit',
                        'reason': f'bbox width increased by {width_change_pct:.1f}%',
                        'gold': g['text'], 'suspect': s['text'], 'gold_bbox': {k:g[k] for k in ('left','top','width','height')}, 'suspect_bbox': {k:s[k] for k in ('left','top','width','height')}
                    })
                else:
                    findings.append({ 'type': 'numeric_mismatch', 'gold': g['text'], 'suspect': s['text'], 'lev': best['dist'] })
            else:
                findings.append({ 'type': 'token_changed', 'gold': g['text'], 'suspect': s['text'], 'lev': best['dist'] })
        else:
            # Same token; check bbox width anomaly
            if 'width' in g and 'width' in s and g['width'] > 0:
                pct = ((s['width'] - g['width']) / g['width']) * 100.0
                if abs(pct) > 15:
                    findings.append({ 'type': 'bbox_anomaly', 'token': g['text'], 'change_pct': pct, 'gold_bbox': {k:g[k] for k in ('left','top','width','height')}, 'suspect_bbox': {k:s[k] for k in ('left','top','width','height')} })

    # tokens present only in suspect or only in gold
    gold_texts = set([w['text'] for w in gold_words])
    suspect_texts = set([w['text'] for w in suspect_words])
    for tok in suspect_texts - gold_texts:
        findings.append({ 'type': 'suspect_only', 'token': tok })
    for tok in gold_texts - suspect_texts:
        findings.append({ 'type': 'gold_only', 'token': tok })

    return findings


# ------------------------------- Lightweight Semantic Checks -------------------------------

def extract_dates_from_text(text: str) -> List[datetime]:
    # naive regex for dates in common formats; then parse with dateutil
    date_patterns = [r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b', r'\b\d{4}[\-]\d{1,2}[\-]\d{1,2}\b']
    dates = []
    for pat in date_patterns:
        for m in re.findall(pat, text):
            try:
                d = dateparser.parse(m, dayfirst=True)
                if d:
                    dates.append(d)
            except Exception:
                pass
    return dates


def semantic_checks(gold_text: str, suspect_text: str) -> List[Dict[str, Any]]:
    findings = []
    # 1) Future date detection (compared to now)
    now = datetime.utcnow()
    s_dates = extract_dates_from_text(suspect_text)
    for d in s_dates:
        if d > now and (d - now).days > 1:
            findings.append({ 'type': 'future_date', 'date': d.isoformat(), 'reason': 'suspect contains date in the future vs server now' })

    # 2) Inconsistency: amounts mentioned vs other fields (naive numeric checks)
    gold_nums = re.findall(r'\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\b', gold_text)
    suspect_nums = re.findall(r'\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\b', suspect_text)
    # if suspect has numbers not in gold, flag
    for n in suspect_nums:
        if n not in gold_nums:
            findings.append({ 'type': 'numeric_inconsistency', 'number': n })

    # 3) Prohibited words
    prohibited = ['guarantee', 'assured return', 'assured-return', 'guaranteed return']
    for p in prohibited:
        if p in suspect_text.lower():
            findings.append({ 'type': 'prohibited_word', 'word': p })

    # 4) Simple contradiction detection: if suspect contains "signed on <date>" but date differs from date field
    # (requires more structured field mapping; left as placeholder)

    return findings


# ------------------------------- Main Endpoint -------------------------------

@app.post('/analyze')
async def analyze(gold: UploadFile = File(...), suspect: UploadFile = File(...)):
    # read bytes
    gold_bytes = await gold.read()
    suspect_bytes = await suspect.read()

    # safety: size limits (e.g., 20MB)
    max_size = 20 * 1024 * 1024
    if len(gold_bytes) > max_size or len(suspect_bytes) > max_size:
        raise HTTPException(status_code=413, detail='One of the files exceeds size limit (20MB)')

    # 1) metadata
    gold_meta = extract_metadata(gold, gold_bytes)
    suspect_meta = extract_metadata(suspect, suspect_bytes)

    # 2) render to images (first page)
    try:
        gold_img = read_first_page_image(gold_bytes)
        suspect_img = read_first_page_image(suspect_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Failed to render files to images: {e}')

    # 3) compute ELA
    gold_ela = compute_ela(gold_img, quality=60)
    suspect_ela = compute_ela(suspect_img, quality=60)

    # 4) OCR
    try:
        gold_ocr = ocr_with_pytesseract(gold_img)
        suspect_ocr = ocr_with_pytesseract(suspect_img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'OCR failed: {e}')

    # 5) compare OCR
    ocr_findings = compare_ocr(gold_ocr, suspect_ocr)

    # 6) semantic checks
    sem_findings = semantic_checks(gold_ocr.get('full_text', ''), suspect_ocr.get('full_text', ''))

    # 7) metadata heuristics
    meta_findings = []
    try:
        # check suspicious tools in suspect metadata
        sm_raw = json.dumps(suspect_meta.get('raw', suspect_meta), default=str).lower()
        suspicious_tools = ['adobe', 'photoshop', 'foxit', 'illustrator', 'pdfelement', 'pdf editor', 'pdf-xchange']
        for t in suspicious_tools:
            if t in sm_raw:
                meta_findings.append({ 'type': 'suspicious_tool_detected', 'tool': t })
        # check modify vs create
        gold_raw = gold_meta.get('raw', {})
        suspect_raw = suspect_meta.get('raw', {})
        def parse_possible_date(x):
            for key in ['ModDate', 'ModifyDate', 'CreationDate', 'CreateDate', 'Producer']:
                if isinstance(x, dict) and key in x:
                    try:
                        return dateparser.parse(str(x[key]))
                    except Exception:
                        pass
            return None
        s_mod = parse_possible_date(suspect_raw)
        g_create = parse_possible_date(gold_raw)
        if s_mod and g_create and s_mod > g_create:
            meta_findings.append({ 'type': 'modify_after_create', 'suspect_modify': s_mod.isoformat(), 'gold_create': g_create.isoformat() })
    except Exception as e:
        meta_findings.append({ 'type': 'meta_check_error', 'error': str(e) })

    # 8) summarise flags
    summary_flags = []
    if any(f for f in meta_findings if f.get('type') == 'suspicious_tool_detected'):
        summary_flags.append('suspicious_pdf_tool_detected')
    if any(f for f in ocr_findings if f.get('type') in ('inserted_digit','suspect_only')):
        summary_flags.append('possible_inserted_digit_or_token')
    # ELA hotspot compare: count hotspots above mean+std
    def ela_hotspot_count(ela):
        flat = [v for row in ela['grid'] for v in row]
        if not flat:
            return 0
        mean = sum(flat)/len(flat)
        std = math.sqrt(sum((x-mean)**2 for x in flat)/len(flat))
        thr = max(mean + 1.5*std, mean*1.8)
        return sum(1 for x in flat if x > thr)
    g_hot = ela_hotspot_count(gold_ela)
    s_hot = ela_hotspot_count(suspect_ela)
    if s_hot > g_hot + 2:
        summary_flags.append('suspect_has_more_ela_hotspots')

    # prepare JSON-safe response (trim heavy image arrays)
    response = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'files': {
            'gold': { 'filename': gold.filename, 'content_type': gold.content_type, 'size': len(gold_bytes) },
            'suspect': { 'filename': suspect.filename, 'content_type': suspect.content_type, 'size': len(suspect_bytes) }
        },
        'metadata': { 'gold': gold_meta, 'suspect': suspect_meta },
        'ocr': {
            'gold': { 'full_text': gold_ocr.get('full_text'), 'words_count': len(gold_ocr.get('words', [])) },
            'suspect': { 'full_text': suspect_ocr.get('full_text'), 'words_count': len(suspect_ocr.get('words', [])) },
            'findings': ocr_findings
        },
        'ela': { 'gold': { 'mean': gold_ela['mean'], 'std': gold_ela['std'], 'preview': gold_ela['preview'] }, 'suspect': { 'mean': suspect_ela['mean'], 'std': suspect_ela['std'], 'preview': suspect_ela['preview'] } },
        'semantic_findings': sem_findings,
        'metadata_findings': meta_findings,
        'summary_flags': summary_flags
    }

    return JSONResponse(content=response)


# ------------------------------- Run note -------------------------------
# Start with: uvicorn fastapi_tamper_backend:app --reload --port 8000
# Test with curl:
# curl -X POST -F "gold=@/path/to/original.pdf" -F "suspect=@/path/to/suspect.pdf" http://127.0.0.1:8000/analyze

