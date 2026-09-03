"""
ml_stubs/ocr.py — OCR / Text Extraction
Implements Teammate 1's interface using the real OCR & MRZ pipeline.
Accepts a base64-encoded image string and returns the CommonResponse contract.
"""

import base64
import re
import cv2
import numpy as np

# ── Real engine imports ─────────────────────────────────────────────────────
try:
    from src.preprocessing.preprocessor import DocumentPreprocessor
    from src.ocr_engine.ocr_extractor import OCRExtractor

    _preprocessor = DocumentPreprocessor()
    _ocr = OCRExtractor()
    _ENGINES_AVAILABLE = True
except Exception as _import_err:
    _ENGINES_AVAILABLE = False
    _import_err_msg = str(_import_err)


def _decode_b64_image(b64_string: str) -> np.ndarray:
    """Decodes a base64 data-URL or raw base64 string into an OpenCV BGR image."""
    if "," in b64_string:                   # data:image/jpeg;base64,<data>
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def perform_ocr(document_image: str) -> dict:
    """
    Stage 02 — Extract.
    Accepts base64-encoded document image, returns structured OCR fields.

    Returns CommonResponse:
        status  : "PASS" | "REVIEW" | "FLAG" | "REJECT" | "UNAVAILABLE"
        score   : float (0.0–1.0 OCR confidence) | None
        evidence: list[str] extracted field descriptions
        reason  : str summary
        data    : dict with extracted fields (name, docNo, dob, expiry, mrz, etc.)
    """
    if not _ENGINES_AVAILABLE:
        return {
            "status": "UNAVAILABLE",
            "score": None,
            "evidence": [],
            "reason": f"OCR engine import failed: {_import_err_msg}",
            "data": None
        }

    try:
        img = _decode_b64_image(document_image)
        if img is None:
            return {
                "status": "UNAVAILABLE",
                "score": None,
                "evidence": [],
                "reason": "Could not decode document image bytes.",
                "data": None
            }

        # Stage 01: Preprocess (deskew, CLAHE, denoise)
        prep = _preprocessor.process(img)
        clean = prep["processed_image"]

        # Stage 02: OCR + MRZ parse
        ocr_res = _ocr.process(clean)

        if not ocr_res.get("success"):
            return {
                "status": "REVIEW",
                "score": 0.0,
                "evidence": [],
                "reason": "No text detected — low-quality or blank image.",
                "data": None
            }

        fields = ocr_res.get("fields", {})
        conf   = ocr_res.get("overall_confidence", 0.0)
        mrz    = ocr_res.get("mrz")

        # Build evidence strings from populated fields
        evidence = [
            f"Field extracted: {k} = {v}"
            for k, v in fields.items()
            if v is not None
        ]

        # Determine status from confidence
        if conf >= 0.75:
            status = "PASS"
            reason = f"OCR successful. {len(evidence)} fields extracted. MRZ {'parsed' if mrz else 'not detected'}."
        elif conf >= 0.45:
            status = "REVIEW"
            reason = f"Low OCR confidence ({conf:.2f}). Manual verification recommended."
        else:
            status = "FLAG"
            reason = f"Very low OCR confidence ({conf:.2f}). Document may be degraded or non-standard."

        return {
            "status": status,
            "score": round(float(conf), 4),
            "evidence": evidence,
            "reason": reason,
            "data": {
                # Flat keys screening.html reads
                "name":   fields.get("full_name"),
                "docNo":  fields.get("document_number"),
                "dob":    fields.get("date_of_birth"),
                "expiry": fields.get("date_of_expiry"),
                "gender": fields.get("gender"),
                "nationality": fields.get("nationality"),
                "issuing_country": fields.get("issuing_country"),
                # Full structured output for backend pipeline
                "fields": fields,
                "field_confidences": ocr_res.get("field_confidences", {}),
                "mrz": mrz,
                "viz": ocr_res.get("viz"),
                "ocr_tokens": ocr_res.get("ocr_tokens", []),
            }
        }

    except Exception as exc:
        return {
            "status": "UNAVAILABLE",
            "score": None,
            "evidence": [],
            "reason": f"OCR pipeline exception: {exc}",
            "data": None
        }
