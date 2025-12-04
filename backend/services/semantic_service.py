def analyze_semantics(ocr_output, metadata):
    """
    Lightweight heuristic semantics analyzer.
    In prod replace with LLM + prompt engineering.
    Returns summary, semantic_score (0..1), and list of issues.
    """
    text = ocr_output.get("text", "")
    out = {"summary": "", "semantic_score": 0.0, "issues": []}
    # simple checks:
    if "guarantee" in text.lower() or "guaranteed" in text.lower():
        out["issues"].append("Prohibited word detected: guarantee/guaranteed")
        out["semantic_score"] += 0.3
    # detect future dates (YYYY)
    import re
    years = re.findall(r"\b(20[2-9][0-9])\b", text)
    for y in years:
        # if year greater than current year, flag
        from datetime import datetime
        if int(y) > datetime.now().year:
            out["issues"].append(f"Future year detected: {y}")
            out["semantic_score"] += 0.3

    if out["issues"]:
        out["summary"] = " / ".join(out["issues"])
    else:
        out["summary"] = "No obvious semantic issues detected."
    out["semantic_score"] = min(1.0, out["semantic_score"])
    return out

def chat_with_document(question: str, document_results: dict) -> str:
    """
    Very small heuristic "chat" for explanation. Replace with a real LLM call (OpenAI/GPT/Claude)
    using document_results as context.
    """
    q = question.lower()
    if "date" in q:
        found = document_results.get("semantic_findings", {}).get("issues", [])
        return f"Dates analysis: {document_results.get('metadata', {}).get('notes', 'No metadata issues found.')} \nSemantics: {found if found else 'No date-related issues found.'}"
    if "why" in q or "explain" in q:
        return f"Summary: {document_results.get('semantic_findings', {}).get('summary', 'No summary available')}\nImage notes: {document_results.get('image_findings', {}).get('notes', [])}"
    return "I don't have a detailed answer for that yet. Hook this to an LLM for richer responses."
