"""
ml_stubs/liveness.py — Anti-Spoofing / Liveness Detection
Implements liveness analysis as a standalone module using heuristics from the face engine.
Accepts a base64-encoded face/selfie image and returns the CommonResponse contract.
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


def check_liveness(face_image: str) -> dict:
    """
    Standalone liveness / anti-spoofing check.
    Uses multi-cue heuristics:
      1. Laplacian sharpness (blur detection — printed photo replay has blur halos)
      2. HSV colour saturation (screen-displayed photos are over-saturated or washed out)
      3. FFT frequency energy (printed & screen images show distinct spectral patterns)

    Args:
        face_image: base64-encoded selfie / face capture.

    Returns CommonResponse:
        status  : "PASS" | "REVIEW" | "FLAG" | "UNAVAILABLE"
        score   : float (0.0 = spoof, 1.0 = live) | None
        evidence: list[str] cue results
        reason  : str summary
        data    : dict with raw metrics
    """
    if not _ENGINES_AVAILABLE:
        return {
            "status": "UNAVAILABLE",
            "score": None,
            "evidence": [],
            "reason": f"Liveness engine import failed: {_import_err_msg}",
            "data": None
        }

    try:
        img = _decode_b64_image(face_image)
        if img is None:
            return {
                "status": "UNAVAILABLE",
                "score": None,
                "evidence": [],
                "reason": "Could not decode face image bytes.",
                "data": None
            }

        result = _face_engine.check_liveness_heuristics(img)

        liveness_score = result["liveness_score"]   # 0.0–1.0
        is_live        = result["is_live"]
        sharpness      = result["sharpness_var"]
        fft_energy     = result["fft_energy"]

        evidence = [
            f"Sharpness (Laplacian variance): {sharpness:.2f} — {'Sharp' if sharpness > 30 else 'Blurry/Flat'}",
            f"FFT spectral energy: {fft_energy:.2f}",
            f"Liveness score: {liveness_score:.2f}/1.00",
        ]

        if is_live:
            status = "PASS"
            reason = f"Liveness confirmed — score {liveness_score:.2f}. No spoofing artefacts detected."
        elif liveness_score >= 0.5:
            status = "REVIEW"
            reason = f"Borderline liveness — score {liveness_score:.2f}. Manual check recommended."
        else:
            status = "FLAG"
            reason = f"Possible spoof — liveness score {liveness_score:.2f}. Printed photo or screen replay suspected."

        return {
            "status":   status,
            "score":    liveness_score,
            "evidence": evidence,
            "reason":   reason,
            "data": {
                "liveness_score": liveness_score,
                "is_live":        is_live,
                "sharpness_var":  sharpness,
                "fft_energy":     fft_energy,
            }
        }

    except Exception as exc:
        return {
            "status": "UNAVAILABLE",
            "score": None,
            "evidence": [],
            "reason": f"Liveness check exception: {exc}",
            "data": None
        }
