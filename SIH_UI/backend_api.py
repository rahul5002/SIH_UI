"""
PRAMAAN AI — Unified Backend API
Serves the SIH_UI frontend contract AND the production /api/v1/* endpoints,
both backed by the real ML pipeline from src/.

SIH_UI frontend contract routes  (base64 JSON payloads):
  POST /api/auth/login
  POST /api/screening/extract
  POST /api/screening/validate
  POST /api/screening/detect
  POST /api/screening/verify
  POST /api/screening/log
  POST /api/screening/decision
  POST /api/screening/analyze
  GET  /api/audit
  GET  /api/health

Production multipart-upload routes  (src/api/app.py parity):
  GET  /api/v1/health
  POST /api/v1/screen-document
  POST /api/v1/ocr-extract
  POST /api/v1/detect-tampering
  POST /api/v1/verify-face
  POST /api/v1/generate-synthetic-data
  GET  /api/v1/run-benchmark
"""

import os
import sys
import time
import uuid
import base64
import sqlite3
import datetime
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# ── Path setup: allow src.* imports from project root ──────────────────────
_here = os.path.abspath(os.path.dirname(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

# ── Real ML Engine imports ──────────────────────────────────────────────────
from src.preprocessing.preprocessor import DocumentPreprocessor
from src.ocr_engine.ocr_extractor import OCRExtractor
from src.validation_engine.rules_validator import DocumentRulesValidator
from src.tampering_engine.tampering_detector import TamperingDetector
from src.face_engine.face_matcher import FaceVerificationEngine

# Optional: synthetic generator (may fail if Pillow font resources missing)
try:
    from src.synthetic_generator.generator import SyntheticDocumentGenerator
    from src.synthetic_generator.tampering_injector import TamperingInjector
    from src.synthetic_generator.benchmark import ScreeningBenchmark
    _SYNTH_AVAILABLE = True
except Exception:
    _SYNTH_AVAILABLE = False

# ── ML stub wrappers (maintain existing call-site compatibility) ────────────
from ml_stubs import ocr as _ocr_stub
from ml_stubs import tamper as _tamper_stub
from ml_stubs import face as _face_stub
from ml_stubs import liveness as _liveness_stub
from ml_stubs import dedup as _dedup_stub

# =============================================================================
# CONFIGURATION
# =============================================================================
AUTO_FLAG_RISK     = 45.0    # composite risk score (0–100) → auto-flag threshold
FACE_MATCH_MINIMUM = 0.85    # biometric similarity minimum for auto-clear

# =============================================================================
# ENGINE SINGLETONS  (eager init for fast first-request response)
# =============================================================================
print("[PRAMAAN] Initialising ML pipeline engines…")
_preprocessor    = DocumentPreprocessor()
_ocr_engine      = OCRExtractor()
_rules_validator = DocumentRulesValidator()
_tamper_detector = TamperingDetector()
_face_engine     = FaceVerificationEngine()
if _SYNTH_AVAILABLE:
    _synth_gen    = SyntheticDocumentGenerator()
    _tamper_inj   = TamperingInjector()
    _benchmark    = ScreeningBenchmark()
print("[PRAMAAN] All engines ready.")

# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(
    title="PRAMAAN AI — Fake Identity & Document Screening",
    description=(
        "Unified API serving the SIH_UI polished frontend and real ML pipeline. "
        "SIH 2026 — Problem Statement #26188."
    ),
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# DATABASE  (SQLite audit ledger)
# =============================================================================
_DB_PATH = os.path.join(_here, "audit.db")

def _init_db():
    conn = sqlite3.connect(_DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            case_ref  TEXT,
            status    TEXT,
            score     REAL,
            reason    TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

_init_db()

def _log_audit(case_ref: str, status: str, score: Optional[float], reason: str):
    conn = sqlite3.connect(_DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO audit_log (case_ref, status, score, reason) VALUES (?, ?, ?, ?)",
        (case_ref, status, score, reason)
    )
    conn.commit()
    conn.close()

# =============================================================================
# IMAGE HELPERS
# =============================================================================
def _decode_b64(b64_string: str):
    """Returns (np.ndarray BGR, raw_bytes) from a base64 data-URL or plain base64."""
    if not b64_string:
        return None, None
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    raw = base64.b64decode(b64_string)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img, raw

def _decode_bytes(image_bytes: bytes):
    """Decodes raw upload bytes into OpenCV BGR image."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode image.")
    return img

# =============================================================================
# PYDANTIC MODELS  (SIH_UI frontend contract)
# =============================================================================
class CommonResponse(BaseModel):
    status:   str
    score:    Optional[float] = None
    evidence: List[str] = []
    reason:   str
    data:     Optional[dict] = None

class LoginReq(BaseModel):
    badgeId:     str
    pin:         str
    checkpostId: str

class ExtractReq(BaseModel):
    documentImage: str          # base64 data-URL

class ValidateReq(BaseModel):
    mrzData: dict               # pre-parsed OCR data dict

class DetectReq(BaseModel):
    documentImage: str          # base64 data-URL

class VerifyReq(BaseModel):
    faceImage:           str    # base64 live selfie
    documentFaceRegion:  dict   # ignored; face cropped internally from doc

class LogReq(BaseModel):
    caseRef:        str
    extractionData: dict
    tamperRes:      dict
    verifyRes:      dict

class DecisionReq(BaseModel):
    caseRef:    str
    decision:   str
    finalScore: float

class AnalyzeReq(BaseModel):
    caseRef:       str
    documentImage: str          # base64
    faceImage:     str          # base64

# =============================================================================
# ── SIH_UI FRONTEND CONTRACT ROUTES ──────────────────────────────────────────
# =============================================================================

@app.get("/api/health")
def health_check():
    return {
        "status":    "ok",
        "timestamp": datetime.datetime.now().isoformat(),
        "engines": {
            "preprocessing":  "READY",
            "ocr_mrz":        "READY",
            "validation":     "READY",
            "tampering":      "READY (ELA+Noise+Splicing+Font+Metadata)",
            "face_liveness":  "READY (MobileNetV3)",
            "dedup":          "READY (numpy cosine / FAISS-upgradeable)",
            "synthetic":      "READY" if _SYNTH_AVAILABLE else "UNAVAILABLE",
        }
    }


@app.post("/api/auth/login")
def login(req: LoginReq):
    """Prototype mock authentication — do NOT use in production."""
    role = "officer"
    name = "Insp. R. Bhandari"
    bid  = req.badgeId.upper()

    if "ADMIN" in bid or req.pin == "999999":
        role, name = "admin",      "Cmdt. S. Sharma"
    elif "SUPER" in bid or req.pin == "888888":
        role, name = "supervisor", "Dy. Cmdt. M. Singh"
    elif len(req.pin) < 6 or len(req.badgeId) < 4:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "authenticated": True,
        "badgeId":       bid,
        "name":          name,
        "role":          role,
        "checkpostId":   req.checkpostId,
        "shift":         "Night",
        "sessionToken":  "mock-token-123"
    }


@app.post("/api/screening/extract")
def api_extract(req: ExtractReq):
    """
    Stage 02 — OCR Extraction.
    Decodes base64 document, runs preprocessing → OCR → MRZ parsing.
    """
    return _ocr_stub.perform_ocr(req.documentImage)


@app.post("/api/screening/validate")
def api_validate(req: ValidateReq):
    """
    Stage 03 — Document Rules Validation.
    Validates checksums, dates, VIZ/MRZ cross-check from prior OCR result.
    """
    # req.mrzData is the full OCR result dict (as passed by frontend)
    try:
        ocr_like = req.mrzData  # frontend passes extractionData which is the OCR result
        val_res  = _rules_validator.validate(ocr_like)

        verdict      = val_res["verdict"]
        val_score    = val_res["validation_score"]
        violations   = val_res.get("violations", [])
        is_expired   = val_res.get("is_expired", False)

        if verdict == "PASSED":
            status = "PASS"
        elif verdict == "EXPIRED":
            status = "REVIEW"
        elif verdict == "FAILED_INTEGRITY_CHECK":
            status = "FLAG"
        else:
            status = "REVIEW"

        return {
            "status":   status,
            "score":    round(val_score, 4),
            "evidence": violations,
            "reason":   f"Validation {verdict} — score {val_score:.2f}. {'Document EXPIRED.' if is_expired else ''}",
            "data":     val_res
        }
    except Exception as exc:
        return {
            "status":   "UNAVAILABLE",
            "score":    None,
            "evidence": [],
            "reason":   f"Validation exception: {exc}",
            "data":     None
        }


@app.post("/api/screening/detect")
def api_detect(req: DetectReq):
    """Stage 04 — Tampering & Forgery Detection (ELA + Noise + Splicing + Font + Metadata)."""
    return _tamper_stub.detect_tampering(req.documentImage)


@app.post("/api/screening/verify")
def api_verify(req: VerifyReq):
    """
    Stage 05 — Face Verification + Liveness.
    faceImage   = live selfie (base64)
    documentFaceRegion passed as context but face is re-cropped internally from document.
    For the SIH_UI contract, documentFaceRegion may contain the full doc image as 'docImage'.
    """
    doc_b64  = req.documentFaceRegion.get("docImage", req.faceImage)
    face_res = _face_stub.verify_face(req.faceImage, doc_b64)
    liv_res  = _liveness_stub.check_liveness(req.faceImage)

    # Merge liveness into face response data
    if face_res.get("data") and isinstance(face_res["data"], dict):
        face_res["data"]["liveness"] = liv_res.get("data") or {}
    else:
        face_res["data"] = {"liveness": liv_res.get("data") or {}}

    return face_res


@app.post("/api/screening/log")
def api_log(req: LogReq):
    """Stage 06 — Composite Risk Computation & Audit Ledger Write."""
    tamper_data = req.tamperRes.get("data") or {}
    verify_data = req.verifyRes.get("data") or {}

    # Tampering score: 0–1 (from stub) → scale to 0–100
    t_score_raw = req.tamperRes.get("score")
    # Face similarity: 0–1
    v_score_raw = req.verifyRes.get("score")

    if t_score_raw is None or v_score_raw is None:
        _log_audit(req.caseRef, "UNAVAILABLE", None, "Pipeline data incomplete — ML models still integrating")
        return {"success": False, "compositeScore": None, "suggestedDecision": "UNAVAILABLE"}

    # Composite risk: tampering drives primary risk (50%), face mismatch adds risk (50%)
    tamper_risk = float(t_score_raw) * 100.0
    face_risk   = (1.0 - float(v_score_raw)) * 100.0
    composite   = round(0.60 * tamper_risk + 0.40 * face_risk, 2)

    is_flagged = (
        composite > AUTO_FLAG_RISK
        or float(v_score_raw) < FACE_MATCH_MINIMUM
        or req.tamperRes.get("status") in ("FLAG", "REJECT")
        or req.verifyRes.get("status") in ("FLAG", "REJECT")
    )

    suggested = "flag" if is_flagged else "clear"
    _log_audit(req.caseRef, "LOGGED", composite, f"Suggested: {suggested}")

    return {
        "success":           True,
        "compositeScore":    composite,
        "suggestedDecision": suggested,
        "detail": {
            "tamper_risk": round(tamper_risk, 2),
            "face_risk":   round(face_risk, 2),
            "thresholds": {
                "auto_flag_risk":     AUTO_FLAG_RISK,
                "face_match_minimum": FACE_MATCH_MINIMUM,
            }
        }
    }


@app.post("/api/screening/decision")
def api_decision(req: DecisionReq):
    """Officer manual decision — anchors to audit ledger."""
    _log_audit(req.caseRef, req.decision.upper(), req.finalScore, "Officer manual decision applied")
    import hashlib
    block_hash = hashlib.sha256(f"{req.caseRef}{req.decision}{req.finalScore}".encode()).hexdigest()
    return {
        "success":   True,
        "blockNum":  49201 + (int(block_hash[:4], 16) % 1000),   # deterministic demo block
        "timestamp": datetime.datetime.now().isoformat()
    }


@app.post("/api/screening/analyze")
def api_analyze(req: AnalyzeReq):
    """
    Unified orchestrator — runs the entire 5-stage pipeline in one call.
    Used when the frontend wants a single-shot result (e.g., batch mode).
    """
    start = time.time()

    # Stage 01: decode images
    doc_img, doc_bytes = _decode_b64(req.documentImage)
    live_img, _        = _decode_b64(req.faceImage)

    if doc_img is None:
        raise HTTPException(status_code=400, detail="Could not decode documentImage.")

    # Stage 01: Preprocess
    prep      = _preprocessor.process(doc_img)
    clean_img = prep["processed_image"]

    # Stage 02: OCR
    ocr_res = _ocr_engine.process(clean_img)

    # Stage 03: Validation
    val_res = _rules_validator.validate(ocr_res)

    # Stage 04: Tampering
    tamper_res = _tamper_detector.detect(
        image=clean_img,
        image_bytes=doc_bytes,
        ocr_tokens=ocr_res.get("ocr_tokens", []),
        photo_bbox=None
    )

    # Stage 05: Face + liveness + dedup
    face_report  = None
    liv_report   = None
    dedup_report = None

    if live_img is not None:
        face_report  = _face_engine.verify_faces(clean_img, live_img)
        liv_report   = _face_engine.check_liveness_heuristics(live_img)
        dedup_report = _dedup_stub.check_deduplication(req.faceImage, req.caseRef)

    # Stage 06: Risk synthesis
    tamper_risk = tamper_res["tampering_score"]                              # 0–100
    val_risk    = (1.0 - val_res["validation_score"]) * 100.0
    ocr_risk    = (1.0 - ocr_res.get("overall_confidence", 0.8)) * 100.0
    overall_risk = round(0.50 * tamper_risk + 0.35 * val_risk + 0.15 * ocr_risk, 2)

    # Verdict
    reasons = []
    if val_res.get("is_expired"):
        reasons.append("Document is expired.")
    if tamper_res["tampering_detected"]:
        reasons.append(f"Tampering detected — score {tamper_risk:.1f}/100.")
    if face_report and not face_report.get("is_match"):
        reasons.append(f"Biometric mismatch — similarity {face_report.get('similarity_score', 0):.4f}.")
    if dedup_report and dedup_report.get("status") == "FLAG":
        reasons.append(dedup_report.get("reason", "Duplicate biometric."))

    if overall_risk < 30 and val_res["verdict"] == "PASSED" and not tamper_res["tampering_detected"]:
        overall_verdict = "PASSED_CLEAN"
        risk_level      = "LOW"
    elif val_res.get("is_expired"):
        overall_verdict = "REJECTED_EXPIRED_DOCUMENT"
        risk_level      = "MODERATE"
    elif tamper_res["tampering_detected"] or val_res["verdict"] == "FAILED_INTEGRITY_CHECK":
        overall_verdict = "FLAGGED_FORGERY_DETECTED"
        risk_level      = "CRITICAL"
    else:
        overall_verdict = "MANUAL_INSPECTION_REQUIRED"
        risk_level      = "HIGH"

    if not reasons:
        reasons.append("All pipeline checks passed.")

    _log_audit(req.caseRef, overall_verdict, overall_risk, " | ".join(reasons))

    return {
        "screening_id":     req.caseRef,
        "timestamp":        datetime.datetime.now().isoformat(),
        "overall_risk_score": overall_risk,
        "overall_verdict":  overall_verdict,
        "risk_level":       risk_level,
        "status":           "FLAG" if risk_level in ("CRITICAL", "HIGH") else
                            "REVIEW" if risk_level == "MODERATE" else "PASS",
        "ocr": {
            "status":   "PASS" if ocr_res.get("success") else "REVIEW",
            "score":    ocr_res.get("overall_confidence"),
            "evidence": [f"{k}: {v}" for k, v in (ocr_res.get("fields") or {}).items() if v],
            "reason":   "OCR complete." if ocr_res.get("success") else "OCR failed.",
            "data": {
                "name":   (ocr_res.get("fields") or {}).get("full_name"),
                "docNo":  (ocr_res.get("fields") or {}).get("document_number"),
                "dob":    (ocr_res.get("fields") or {}).get("date_of_birth"),
                "expiry": (ocr_res.get("fields") or {}).get("date_of_expiry"),
                "fields": ocr_res.get("fields"),
                "mrz":    ocr_res.get("mrz"),
            }
        },
        "validation": {
            "status":   "PASS" if val_res["verdict"] == "PASSED" else "FLAG",
            "score":    val_res["validation_score"],
            "evidence": val_res.get("violations", []),
            "reason":   val_res["verdict"],
            "data":     val_res
        },
        "tampering": {
            "status":   "PASS" if not tamper_res["tampering_detected"] else "FLAG",
            "score":    round(tamper_risk / 100.0, 4),
            "evidence": tamper_res["reasons"],
            "reason":   tamper_res["verdict"],
            "data":     {
                "tampering_score":    tamper_risk,
                "risk_level":         tamper_res["risk_level"],
                "verdict":            tamper_res["verdict"],
                "tampering_detected": tamper_res["tampering_detected"],
                "module_scores":      tamper_res["module_scores"],
                "evidence_regions":   tamper_res["evidence_regions"],
                "heatmap_base64":     tamper_res["heatmap_base64"],
            }
        },
        "face":       face_report,
        "liveness":   liv_report,
        "dedup":      dedup_report,
        "overall": {
            "riskScore": overall_risk,
            "flags":     reasons
        },
        "audit": {
            "caseRef":   req.caseRef,
            "timestamp": datetime.datetime.now().isoformat()
        },
        "processing_time_ms": round((time.time() - start) * 1000, 2)
    }


@app.get("/api/audit")
def api_audit():
    """Returns last 50 audit log entries for the Audit Log page."""
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT 50")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


# =============================================================================
# ── PRODUCTION /api/v1/* ROUTES  (multipart file uploads) ─────────────────
# =============================================================================

@app.get("/api/v1/health")
def v1_health():
    return {
        "status":    "ONLINE",
        "system":    "PRAMAAN AI Border Document Screening Platform",
        "version":   "2.0.0",
        "timestamp": datetime.datetime.now().isoformat(),
        "modules": {
            "stage_01_preprocessing":    "READY",
            "stage_02_ocr_mrz":          "READY",
            "stage_03_rules_validation": "READY",
            "stage_04_tampering":        "READY (ELA+Noise+Splicing+Font+Metadata)",
            "stage_05_face_liveness":    "READY (MobileNetV3)",
            "stage_06_dedup":            "READY (numpy cosine)",
        }
    }


@app.post("/api/v1/screen-document")
async def v1_screen(
    document:    UploadFile = File(...),
    live_selfie: Optional[UploadFile] = File(None)
):
    """Full pipeline via multipart file upload."""
    start    = time.time()
    case_id  = str(uuid.uuid4())[:8].upper()
    doc_bytes = await document.read()
    doc_img   = _decode_bytes(doc_bytes)

    prep      = _preprocessor.process(doc_img)
    clean     = prep["processed_image"]
    ocr_res   = _ocr_engine.process(clean)
    val_res   = _rules_validator.validate(ocr_res)
    tamper_res = _tamper_detector.detect(image=clean, image_bytes=doc_bytes, ocr_tokens=ocr_res.get("ocr_tokens", []))

    face_report = None
    if live_selfie:
        live_bytes = await live_selfie.read()
        live_img   = _decode_bytes(live_bytes)
        face_report = _face_engine.verify_faces(clean, live_img)

    tamper_risk = tamper_res["tampering_score"]
    val_risk    = (1.0 - val_res["validation_score"]) * 100.0
    ocr_risk    = (1.0 - ocr_res.get("overall_confidence", 0.8)) * 100.0
    overall     = round(0.50 * tamper_risk + 0.35 * val_risk + 0.15 * ocr_risk, 2)

    if overall < 30 and val_res["verdict"] == "PASSED" and not tamper_res["tampering_detected"]:
        verdict, rlevel = "PASSED_CLEAN",              "LOW"
    elif val_res.get("is_expired"):
        verdict, rlevel = "REJECTED_EXPIRED_DOCUMENT", "MODERATE"
    elif tamper_res["tampering_detected"] or val_res["verdict"] == "FAILED_INTEGRITY_CHECK":
        verdict, rlevel = "FLAGGED_FORGERY_DETECTED",  "CRITICAL"
    else:
        verdict, rlevel = "MANUAL_INSPECTION_REQUIRED","HIGH"

    return {
        "screening_id":       case_id,
        "timestamp":          datetime.datetime.now().isoformat(),
        "overall_risk_score": overall,
        "overall_verdict":    verdict,
        "risk_level":         rlevel,
        "extracted_fields":   ocr_res.get("fields", {}),
        "field_confidences":  ocr_res.get("field_confidences", {}),
        "ocr_confidence":     ocr_res.get("overall_confidence", 0.0),
        "mrz_details":        ocr_res.get("mrz"),
        "validation_report":  val_res,
        "tampering_report":   tamper_res,
        "face_report":        face_report,
        "processing_time_ms": round((time.time() - start) * 1000, 2)
    }


@app.post("/api/v1/ocr-extract")
async def v1_ocr(document: UploadFile = File(...)):
    doc_bytes = await document.read()
    doc_img   = _decode_bytes(doc_bytes)
    prep      = _preprocessor.process(doc_img)
    return _ocr_engine.process(prep["processed_image"])


@app.post("/api/v1/detect-tampering")
async def v1_tamper(document: UploadFile = File(...)):
    doc_bytes  = await document.read()
    doc_img    = _decode_bytes(doc_bytes)
    prep       = _preprocessor.process(doc_img)
    ocr_res    = _ocr_engine.process(prep["processed_image"])
    return _tamper_detector.detect(
        image=prep["processed_image"],
        image_bytes=doc_bytes,
        ocr_tokens=ocr_res.get("ocr_tokens", [])
    )


@app.post("/api/v1/verify-face")
async def v1_face(
    document:    UploadFile = File(...),
    live_selfie: UploadFile = File(...)
):
    doc_bytes  = await document.read()
    live_bytes = await live_selfie.read()
    doc_img    = _decode_bytes(doc_bytes)
    live_img   = _decode_bytes(live_bytes)
    return _face_engine.verify_faces(doc_img, live_img)


@app.post("/api/v1/generate-synthetic-data")
def v1_synth(
    document_type:      str = Form("PASSPORT"),
    tampering_scenario: str = Form("CLEAN"),
    country_code:       str = Form("IND")
):
    if not _SYNTH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Synthetic generator not available.")
    clean_doc, gt, avatar = _synth_gen.generate_passport(country_code=country_code)
    if tampering_scenario == "PHOTO_REPLACEMENT":
        doc, gt = _tamper_inj.inject_photo_replacement(clean_doc, gt)
    elif tampering_scenario == "TEXT_DATE":
        doc, gt = _tamper_inj.inject_date_alteration(clean_doc, gt)
    elif tampering_scenario == "FAKE_STAMP":
        doc, gt = _tamper_inj.inject_fake_stamp(clean_doc, gt)
    else:
        doc = clean_doc
    _, buf  = cv2.imencode(".png", doc)
    _, abuf = cv2.imencode(".png", avatar)
    return {
        "ground_truth":     gt,
        "document_base64":  f"data:image/png;base64,{base64.b64encode(buf).decode()}",
        "avatar_base64":    f"data:image/png;base64,{base64.b64encode(abuf).decode()}"
    }


@app.get("/api/v1/run-benchmark")
def v1_benchmark(num_samples: int = 8):
    if not _SYNTH_AVAILABLE:
        raise HTTPException(status_code=503, detail="Benchmark requires synthetic generator.")
    return _benchmark.run_benchmark(num_samples=num_samples)


# =============================================================================
# ── STATIC FRONTEND  (SIH_UI HTML/CSS/JS served from project root) ──────────
# =============================================================================
app.mount("/css",    StaticFiles(directory=os.path.join(_here, "css")),    name="css")
app.mount("/js",     StaticFiles(directory=os.path.join(_here, "js")),     name="js")
app.mount("/assets", StaticFiles(directory=os.path.join(_here, "assets")), name="assets")

# Named HTML pages
_pages = [
    ("",                "screening.html"),
    ("index.html",      "screening.html"),
    ("dashboard",       "screening.html"),
    ("dashboard.html",  "screening.html"),
    ("screening",       "screening.html"),
    ("screening.html",   "screening.html"),
    ("audit",           "audit.html"),
    ("audit.html",       "audit.html"),
    ("checkpoints",     "checkpoints.html"),
    ("checkpoints.html", "checkpoints.html"),
    ("alerts",          "alerts.html"),
    ("alerts.html",      "alerts.html"),
    ("settings",        "settings.html"),
    ("settings.html",    "settings.html"),
    ("login",           "login.html"),
    ("login.html",      "login.html"),
]



for _route, _file in _pages:
    _fp = os.path.join(_here, _file)
    if os.path.exists(_fp):
        # Use a closure to capture variables correctly
        def _make_route(fp):
            async def _handler():
                return FileResponse(fp)
            return _handler
        app.add_api_route(
            f"/{_route}" if _route else "/",
            _make_route(_fp),
            methods=["GET"],
            include_in_schema=False
        )
