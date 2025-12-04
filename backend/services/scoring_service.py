def compute_score(payload: dict) -> float:
    """
    payload: { geometry, image, metadata, template, semantic } each 0..1 suspiciousness
    Uses weights:
      w1=0.30 (geometry), w2=0.20 (image), w3=0.20 (metadata), w4=0.20 (template), w5=0.10 (semantic)
    Returns 0..100 authenticity (higher=more authentic)
    """
    w1, w2, w3, w4, w5 = 0.30, 0.20, 0.20, 0.20, 0.10
    G = payload.get("geometry", 0.0)
    I = payload.get("image", 0.0)
    M = payload.get("metadata", 0.0)
    T = payload.get("template", 0.0)
    S = payload.get("semantic", 0.0)
    suspiciousness = (w1*G + w2*I + w3*M + w4*T + w5*S)
    score = max(0.0, min(100.0, 100.0 * (1.0 - suspiciousness)))
    return round(score, 2)
