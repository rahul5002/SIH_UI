"""
PRAMAAN AI FastAPI Web Application & Inference Service
Exposes REST endpoints for Document Screening, OCR Extraction, Tampering Forensics, and Biometric Verification.
"""

import time
import uuid
from datetime import datetime
from typing import Optional
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import os

from src.preprocessing.preprocessor import DocumentPreprocessor
from src.ocr_engine.ocr_extractor import OCRExtractor
from src.validation_engine.rules_validator import DocumentRulesValidator
from src.tampering_engine.tampering_detector import TamperingDetector
from src.face_engine.face_matcher import FaceVerificationEngine
from src.synthetic_generator.generator import SyntheticDocumentGenerator
from src.synthetic_generator.tampering_injector import TamperingInjector
from src.synthetic_generator.benchmark import ScreeningBenchmark
from src.api.schemas import FullScreeningResponse, OCRFields, ValidationReport, TamperingReport, FaceVerificationReport

app = FastAPI(
    title="PRAMAAN AI - Fake Identity & Document Screening System",
    description="AI-Powered Document Screening Platform for Border Checkpoints and Identity Verification (SIH 2026 - ID #26188)",
    version="1.0.0"
)

# Enable CORS for cross-origin web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI/ML Pipeline singletons
preprocessor = DocumentPreprocessor()
ocr_extractor = OCRExtractor()
rules_validator = DocumentRulesValidator()
tampering_detector = TamperingDetector()
face_engine = FaceVerificationEngine()
synthetic_generator = SyntheticDocumentGenerator()
tampering_injector = TamperingInjector()
benchmark_runner = ScreeningBenchmark()


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decodes raw byte array into OpenCV BGR numpy image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode image. Unsupported or corrupted file.")
    return img


@app.get("/api/v1/health")
def health_check():
    """System health and AI engine status."""
    return {
        "status": "ONLINE",
        "system": "PRAMAAN AI Border Document Screening Platform",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "modules": {
            "stage_01_preprocessing": "READY",
            "stage_02_ocr_mrz": "READY",
            "stage_03_rules_validation": "READY",
            "stage_04_tampering_forensics": "READY (ELA + Noise + Splicing + Font)",
            "stage_05_face_verification": "READY"
        }
    }


@app.post("/api/v1/screen-document")
async def screen_document(
    document: UploadFile = File(..., description="Document image file (Passport / Visa / ID)"),
    live_selfie: Optional[UploadFile] = File(None, description="Optional live selfie for biometric verification")
):
    """
    End-to-End Comprehensive Document Screening Pipeline.
    Executes Preprocessing -> OCR & MRZ -> Rules Validation -> Tampering Forensics -> Biometric Matching -> Unified Risk Scoring.
    """
    start_time = time.time()
    screening_id = str(uuid.uuid4())[:8].upper()

    doc_bytes = await document.read()
    doc_image = decode_image_bytes(doc_bytes)

    # 1. Preprocess
    prep_res = preprocessor.process(doc_image)
    clean_img = prep_res["processed_image"]

    # 2. OCR & MRZ Extraction
    ocr_res = ocr_extractor.process(clean_img)

    # 3. Document Rules & Checksum Validation
    val_res = rules_validator.validate(ocr_res)

    # 4. Tampering & Forgery Forensics
    tamper_res = tampering_detector.detect(
        image=clean_img,
        image_bytes=doc_bytes,
        ocr_tokens=ocr_res.get("ocr_tokens", []),
        photo_bbox=None
    )

    # 5. Face Verification (if live selfie provided)
    face_report = None
    if live_selfie is not None:
        live_bytes = await live_selfie.read()
        live_image = decode_image_bytes(live_bytes)
        face_res = face_engine.verify_faces(clean_img, live_image)
        face_report = FaceVerificationReport(
            success=face_res["success"],
            is_match=face_res.get("is_match", False),
            similarity_score=face_res.get("similarity_score", 0.0),
            verdict=face_res.get("verdict", "N/A"),
            threshold=face_res.get("threshold", 0.65),
            liveness_score=face_res.get("liveness", {}).get("liveness_score"),
            doc_face_base64=face_res.get("doc_face_base64"),
            live_face_base64=face_res.get("live_face_base64")
        )

    # 6. Global Risk Score Synthesis (0 - 100)
    # Weights: Tampering (50%), Rules Validation (35%), OCR Confidence (15%)
    tamper_risk = tamper_res["tampering_score"]
    val_risk = (1.0 - val_res["validation_score"]) * 100.0
    ocr_risk = (1.0 - ocr_res.get("overall_confidence", 0.8)) * 100.0

    overall_risk = round(0.50 * tamper_risk + 0.35 * val_risk + 0.15 * ocr_risk, 2)

    # Final Checkpoint Verdict
    if overall_risk < 30.0 and val_res["verdict"] == "PASSED" and not tamper_res["tampering_detected"]:
        overall_verdict = "PASSED_CLEAN"
        risk_level = "LOW"
    elif val_res["is_expired"]:
        overall_verdict = "REJECTED_EXPIRED_DOCUMENT"
        risk_level = "MODERATE"
    elif tamper_res["tampering_detected"] or val_res["verdict"] == "FAILED_INTEGRITY_CHECK":
        overall_verdict = "FLAGGED_FORGERY_DETECTED"
        risk_level = "CRITICAL"
    else:
        overall_verdict = "MANUAL_INSPECTION_REQUIRED"
        risk_level = "HIGH"

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "screening_id": screening_id,
        "timestamp": datetime.now().isoformat(),
        "overall_risk_score": overall_risk,
        "overall_verdict": overall_verdict,
        "risk_level": risk_level,
        "extracted_fields": ocr_res.get("fields", {}),
        "field_confidences": ocr_res.get("field_confidences", {}),
        "ocr_confidence": ocr_res.get("overall_confidence", 0.0),
        "mrz_details": ocr_res.get("mrz"),
        "validation_report": val_res,
        "tampering_report": tamper_res,
        "face_report": face_report.model_dump() if face_report else None,
        "processing_time_ms": elapsed_ms
    }


@app.post("/api/v1/ocr-extract")
async def extract_ocr(document: UploadFile = File(...)):
    """Stage 02: Extracts OCR text tokens, MRZ lines, and parsed fields."""
    doc_bytes = await document.read()
    doc_image = decode_image_bytes(doc_bytes)
    prep_res = preprocessor.process(doc_image)
    ocr_res = ocr_extractor.process(prep_res["processed_image"])
    return ocr_res


@app.post("/api/v1/detect-tampering")
async def detect_tampering(document: UploadFile = File(...)):
    """Stage 04: Executes ELA, Noise, Splicing, and Font Forensics, returning Heatmap and Evidence Boxes."""
    doc_bytes = await document.read()
    doc_image = decode_image_bytes(doc_bytes)
    prep_res = preprocessor.process(doc_image)
    ocr_res = ocr_extractor.process(prep_res["processed_image"])
    tamper_res = tampering_detector.detect(
        image=prep_res["processed_image"],
        image_bytes=doc_bytes,
        ocr_tokens=ocr_res.get("ocr_tokens", [])
    )
    return tamper_res


@app.post("/api/v1/verify-face")
async def verify_face(
    document: UploadFile = File(...),
    live_selfie: UploadFile = File(...)
):
    """Stage 05: Matches document photo with live camera selfie."""
    doc_bytes = await document.read()
    live_bytes = await live_selfie.read()
    doc_img = decode_image_bytes(doc_bytes)
    live_img = decode_image_bytes(live_bytes)
    return face_engine.verify_faces(doc_img, live_img)


@app.post("/api/v1/generate-synthetic-data")
def generate_synthetic_data(
    document_type: str = Form("PASSPORT"),
    tampering_scenario: str = Form("CLEAN"),
    country_code: str = Form("IND")
):
    """
    Generates synthetic sample documents with real ICAO MRZ and selectable tampering scenarios.
    Scenarios: 'CLEAN', 'PHOTO_REPLACEMENT', 'TEXT_DATE', 'FAKE_STAMP'.
    """
    clean_doc, gt, avatar = synthetic_generator.generate_passport(country_code=country_code)

    if tampering_scenario == "PHOTO_REPLACEMENT":
        test_doc, test_gt = tampering_injector.inject_photo_replacement(clean_doc, gt)
    elif tampering_scenario == "TEXT_DATE":
        test_doc, test_gt = tampering_injector.inject_date_alteration(clean_doc, gt)
    elif tampering_scenario == "FAKE_STAMP":
        test_doc, test_gt = tampering_injector.inject_fake_stamp(clean_doc, gt)
    else:
        test_doc, test_gt = clean_doc, gt

    # Encode to Base64
    _, buf = cv2.imencode('.png', test_doc)
    import base64
    doc_b64 = f"data:image/png;base64,{base64.b64encode(buf).decode('utf-8')}"

    _, av_buf = cv2.imencode('.png', avatar)
    avatar_b64 = f"data:image/png;base64,{base64.b64encode(av_buf).decode('utf-8')}"

    return {
        "ground_truth": test_gt,
        "document_base64": doc_b64,
        "avatar_base64": avatar_b64
    }


@app.get("/api/v1/run-benchmark")
def run_benchmark(num_samples: int = 8):
    """Executes automated multi-sample benchmark across synthetic dataset."""
    return benchmark_runner.run_benchmark(num_samples=num_samples)


# Static Files & Dashboard Mount
web_dir = os.path.join(os.path.dirname(__file__), "..", "web")
if os.path.exists(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    @app.get("/")
    def serve_dashboard():
        return FileResponse(os.path.join(web_dir, "index.html"))
