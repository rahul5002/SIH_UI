"""
ml_stubs/face.py — Face Verification / Biometric Matching
Implements Teammate 3's interface using the real MobileNetV3 embedding + cosine similarity engine.
Accepts base64-encoded images and returns the CommonResponse contract.
"""

import base64
import cv2
import numpy as np

# ── Real engine imports ─────────────────────────────────────────────────────
try:
    from src.face_engine.face_matcher import FaceVerificationEngine

    _face_engine = FaceVerificationEngine()
    _ENGINES_AVAILABLE = True
except Exception as _import_err:
    _ENGINES_AVAILABLE = False
    _import_err_msg = str(_import_err)


def _decode_b64_image(b64_string: str) -> np.ndarray:
    """Decodes a base64 data-URL or raw base64 string into an OpenCV BGR image."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def verify_face(live_face: str, document_face: str) -> dict:
    """
    Stage 05 — Verify (face component).
    Extracts faces from both images, computes L2-normalised 512-dim embeddings,
    calculates cosine similarity, and runs liveness anti-spoofing heuristics.

    Args:
        live_face     : base64 image of the live selfie / camera capture.
        document_face : base64 image of the full document page (face cropped internally).

    Returns CommonResponse:
        status  : "PASS" | "REVIEW" | "FLAG" | "REJECT" | "UNAVAILABLE"
        score   : float (0.0–1.0 biometric similarity) | None
        evidence: list[str] verification details
        reason  : str verdict summary
        data    : dict with similarity, liveness, face crops (base64), bboxes
    """
    if not _ENGINES_AVAILABLE:
        return {
            "status": "UNAVAILABLE",
            "score": None,
            "evidence": [],
            "reason": f"Face engine import failed: {_import_err_msg}",
            "data": None
        }

    try:
        doc_img  = _decode_b64_image(document_face)
        live_img = _decode_b64_image(live_face)

        if doc_img is None or live_img is None:
            return {
                "status": "UNAVAILABLE",
                "score": None,
                "evidence": [],
                "reason": "Failed to decode one or both face images.",
                "data": None
            }

        result = _face_engine.verify_faces(doc_img, live_img)

        if not result.get("success"):
            return {
                "status": "REVIEW",
                "score": 0.0,
                "evidence": [],
                "reason": result.get("message", "Face detection failed."),
                "data": result
            }

        sim   = result["similarity_score"]    # 0.0–1.0
        match = result["is_match"]
        live  = result["liveness"]
        verd  = result["verdict"]

        evidence = [
            f"Cosine similarity: {sim:.4f} (threshold: {result['threshold']})",
            f"Liveness score: {live['liveness_score']} — {'LIVE' if live['is_live'] else 'SPOOF SUSPECTED'}",
            f"Sharpness variance: {live['sharpness_var']}",
        ]

        if match:
            status = "PASS"
            reason = f"BIOMETRIC_MATCH — Similarity {sim:.4f} ≥ threshold. Liveness confirmed."
        elif not live["is_live"]:
            status = "FLAG"
            reason = f"LIVENESS_FAILED — Possible spoof attempt. Liveness score: {live['liveness_score']}"
        elif sim < result["threshold"]:
            status = "REJECT"
            reason = f"BIOMETRIC_MISMATCH — Similarity {sim:.4f} < threshold {result['threshold']}. Identity not verified."
        else:
            status = "REVIEW"
            reason = f"BORDERLINE — Similarity {sim:.4f}. Manual officer verification recommended."

        return {
            "status": status,
            "score":  sim,
            "evidence": evidence,
            "reason":   reason,
            "data": {
                "verdict":         verd,
                "is_match":        match,
                "similarity_score": sim,
                "threshold":       result["threshold"],
                "liveness":        live,
                "doc_face_bbox":   result.get("doc_face_bbox"),
                "live_face_bbox":  result.get("live_face_bbox"),
                "doc_face_base64": result.get("doc_face_base64"),
                "live_face_base64": result.get("live_face_base64"),
            }
        }

    except Exception as exc:
        return {
            "status": "UNAVAILABLE",
            "score": None,
            "evidence": [],
            "reason": f"Face verification exception: {exc}",
            "data": None
        }
