"""
ml_stubs/dedup.py — Biometric Deduplication (1:N FAISS Search)
Implements the deduplication interface using an in-memory embedding store.
For the prototype, uses numpy-based cosine similarity against accumulated embeddings.
FAISS can be dropped in as a direct upgrade when available.

Accepts a base64-encoded face image and a case reference, returns CommonResponse.
"""

import base64
import os
import json
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

# ── In-memory / file-persisted embedding store ──────────────────────────────
# In production, replace with FAISS IndexFlatIP.
# For prototype: a simple list of {case_ref, embedding} dicts,
# persisted in a JSON sidecar next to audit.db.

_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "dedup_store.json")
_DEDUP_THRESHOLD = 0.92     # cosine similarity above which a duplicate is declared

_embedding_store: list = []   # [{case_ref: str, embedding: list[float]}, ...]

def _load_store():
    global _embedding_store
    if os.path.exists(_STORE_PATH):
        try:
            with open(_STORE_PATH, "r") as f:
                _embedding_store = json.load(f)
        except Exception:
            _embedding_store = []

def _save_store():
    try:
        with open(_STORE_PATH, "w") as f:
            json.dump(_embedding_store, f)
    except Exception:
        pass

_load_store()   # Load on module import


def _decode_b64_image(b64_string: str) -> np.ndarray:
    """Decodes base64 data-URL or raw string into OpenCV BGR image."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def check_deduplication(face_image: str, case_ref: str) -> dict:
    """
    1:N Biometric Deduplication Search.
    Computes a face embedding for the current subject and searches the
    accumulated embedding store for any prior match above threshold.

    On completion, adds the current embedding to the store (enrolment).

    Args:
        face_image: base64-encoded face / document image.
        case_ref  : unique case reference string (e.g. 'RAX-20260902-0312').

    Returns CommonResponse:
        status  : "PASS" | "FLAG" | "REVIEW" | "UNAVAILABLE"
        score   : float (0.0 = unique, 1.0 = confirmed duplicate) | None
        evidence: list[str]
        reason  : str
        data    : dict with top match info
    """
    if not _ENGINES_AVAILABLE:
        return {
            "status": "UNAVAILABLE",
            "score": None,
            "evidence": [],
            "reason": f"Dedup engine import failed: {_import_err_msg}",
            "data": None
        }

    try:
        img = _decode_b64_image(face_image)
        if img is None:
            return {
                "status": "UNAVAILABLE",
                "score": None,
                "evidence": [],
                "reason": "Could not decode face image for deduplication.",
                "data": None
            }

        # Extract face crop & embedding
        face_crop, _ = _face_engine.detect_and_crop_face(img, is_document=False)
        if face_crop is None or face_crop.size == 0:
            face_crop = img   # Fallback to full image

        embedding = _face_engine.compute_embedding(face_crop)

        best_sim  = 0.0
        best_ref  = None

        if _embedding_store:
            store_matrix = np.array([e["embedding"] for e in _embedding_store], dtype=np.float32)
            q = embedding.astype(np.float32)

            # Cosine similarities (embeddings are already L2-normalised)
            sims = store_matrix @ q / (
                np.linalg.norm(store_matrix, axis=1) * np.linalg.norm(q) + 1e-7
            )
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            best_ref = _embedding_store[best_idx]["case_ref"]

        # Normalise to 0–1
        norm_sim = round(max(0.0, min(1.0, (best_sim + 1.0) / 2.0)), 4)

        is_duplicate = norm_sim >= _DEDUP_THRESHOLD

        # Enrol current subject (even duplicates are enrolled for audit trail)
        _embedding_store.append({
            "case_ref":  case_ref,
            "embedding": embedding.tolist()
        })
        _save_store()

        if is_duplicate:
            status  = "FLAG"
            reason  = f"DUPLICATE_DETECTED — Biometric match with prior case '{best_ref}' (similarity {norm_sim:.4f})."
            score   = norm_sim
            evidence = [
                f"Prior case reference: {best_ref}",
                f"Cosine similarity: {norm_sim:.4f} ≥ threshold {_DEDUP_THRESHOLD}",
                "Subject may have crossed under different identity."
            ]
        elif norm_sim > 0.80:
            status  = "REVIEW"
            reason  = f"HIGH_SIMILARITY — Near-match with case '{best_ref}' ({norm_sim:.4f}). Manual review advised."
            score   = norm_sim
            evidence = [
                f"Best prior match: {best_ref}",
                f"Similarity: {norm_sim:.4f} (below duplicate threshold {_DEDUP_THRESHOLD})",
            ]
        else:
            status  = "PASS"
            reason  = f"UNIQUE — No biometric match in database ({len(_embedding_store) - 1} records searched)."
            score   = norm_sim if _embedding_store else 0.0
            evidence = [f"Database size: {len(_embedding_store) - 1} enrolled embeddings."]

        return {
            "status":   status,
            "score":    score,
            "evidence": evidence,
            "reason":   reason,
            "data": {
                "is_duplicate":      is_duplicate,
                "best_match_ref":    best_ref,
                "best_similarity":   norm_sim,
                "threshold":         _DEDUP_THRESHOLD,
                "db_size":           len(_embedding_store),
            }
        }

    except Exception as exc:
        return {
            "status": "UNAVAILABLE",
            "score": None,
            "evidence": [],
            "reason": f"Deduplication exception: {exc}",
            "data": None
        }
