"""
ml_stubs/tamper.py — Document Tampering / Forgery Detection
Implements Teammate 2's interface using the real ELA + Noise + Splicing + Font + Metadata pipeline.
Accepts a base64-encoded image string and returns the CommonResponse contract.
"""

import base64
import cv2
import numpy as np

# ── Real engine imports ─────────────────────────────────────────────────────
try:
    from src.preprocessing.preprocessor import DocumentPreprocessor
    from src.ocr_engine.ocr_extractor import OCRExtractor
    from src.tampering_engine.tampering_detector import TamperingDetector

    _preprocessor = DocumentPreprocessor()
    _ocr          = OCRExtractor()
    _detector     = TamperingDetector()
    _ENGINES_AVAILABLE = True
except Exception as _import_err:
    _ENGINES_AVAILABLE = False
    _import_err_msg = str(_import_err)


def _decode_b64_image(b64_string: str):
    """Returns (np.ndarray BGR image, raw bytes)."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img, img_bytes


def detect_tampering(document_image: str) -> dict:
    """
    Stage 04 — Detect.
    Runs 5-module forensic analysis:
      1. ELA  (JPEG compression error maps)
      2. Noise variance residuals
      3. Splicing / photo-region boundary analysis
      4. Font character consistency
      5. EXIF metadata forensics

    Returns CommonResponse:
        status  : "PASS" | "REVIEW" | "FLAG" | "REJECT" | "UNAVAILABLE"
        score   : float (0.0–1.0 tampering risk) | None
        evidence: list[str] human-readable findings
        reason  : str verdict summary
        data    : dict with detailed module scores, evidence regions, heatmap
    """
    if not _ENGINES_AVAILABLE:
        return {
            "status": "UNAVAILABLE",
            "score": None,
            "evidence": [],
            "reason": f"Tampering engine import failed: {_import_err_msg}",
            "data": None
        }

    try:
        img, raw_bytes = _decode_b64_image(document_image)
        if img is None:
            return {
                "status": "UNAVAILABLE",
                "score": None,
                "evidence": [],
                "reason": "Could not decode document image bytes.",
                "data": None
            }

        # Stage 01: Preprocess
        prep  = _preprocessor.process(img)
        clean = prep["processed_image"]

        # Stage 02: OCR tokens (for font forensics sub-module)
        ocr_res    = _ocr.process(clean)
        ocr_tokens = ocr_res.get("ocr_tokens", [])

        # Stage 04: Full tampering analysis
        result = _detector.detect(
            image=clean,
            image_bytes=raw_bytes,
            ocr_tokens=ocr_tokens,
            photo_bbox=None
        )

        # Map internal risk to CommonResponse status
        tampering_score = result["tampering_score"]   # 0–100
        risk_level      = result["risk_level"]        # CLEAN | LOW_RISK | HIGH_RISK | CRITICAL_RISK

        if risk_level == "CLEAN":
            status = "PASS"
        elif risk_level == "LOW_RISK":
            status = "REVIEW"
        elif risk_level == "HIGH_RISK":
            status = "FLAG"
        else:  # CRITICAL_RISK
            status = "REJECT"

        return {
            "status": status,
            "score":  round(tampering_score / 100.0, 4),   # normalise to 0–1
            "evidence": result["reasons"],
            "reason":   f"{result['verdict']} — Risk score {tampering_score:.1f}/100 ({risk_level})",
            "data": {
                "tampering_score":    tampering_score,
                "risk_level":         risk_level,
                "verdict":            result["verdict"],
                "tampering_detected": result["tampering_detected"],
                "module_scores":      result["module_scores"],
                "evidence_regions":   result["evidence_regions"],
                "heatmap_base64":     result["heatmap_base64"],
            }
        }

    except Exception as exc:
        return {
            "status": "UNAVAILABLE",
            "score": None,
            "evidence": [],
            "reason": f"Tampering detection exception: {exc}",
            "data": None
        }
