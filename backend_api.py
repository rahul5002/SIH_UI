import sqlite3
import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ml_stubs import ocr, tamper, face, liveness, dedup

# ==========================================
# CONFIGURATION
# ==========================================
AUTO_FLAG_RISK = 45.0
FACE_MATCH_MINIMUM = 0.85

app = FastAPI(title="PRAMAAN AI Backend", description="API for SIH 2026 Prototype")

# ==========================================
# DATABASE SETUP (Prototype Audit Log)
# ==========================================
def init_db():
    conn = sqlite3.connect("audit.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_ref TEXT,
            status TEXT,
            score REAL,
            reason TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def log_audit(case_ref: str, status: str, score: Optional[float], reason: str):
    conn = sqlite3.connect("audit.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO audit_log (case_ref, status, score, reason) VALUES (?, ?, ?, ?)",
        (case_ref, status, score, reason)
    )
    conn.commit()
    conn.close()

# ==========================================
# PYDANTIC MODELS
# ==========================================
class CommonResponse(BaseModel):
    status: str  # "PASS|REVIEW|FLAG|REJECT|UNAVAILABLE"
    score: Optional[float]
    evidence: List[str]
    reason: str
    data: Optional[dict] = None

class LoginReq(BaseModel):
    badgeId: str
    pin: str
    checkpostId: str

class ExtractReq(BaseModel):
    documentImage: str

class ValidateReq(BaseModel):
    mrzData: dict

class DetectReq(BaseModel):
    documentImage: str

class VerifyReq(BaseModel):
    faceImage: str
    documentFaceRegion: dict

class LogReq(BaseModel):
    caseRef: str
    extractionData: dict
    tamperRes: dict
    verifyRes: dict

class DecisionReq(BaseModel):
    caseRef: str
    decision: str
    finalScore: float

class AnalyzeReq(BaseModel):
    caseRef: str
    documentImage: str
    faceImage: str

# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/api/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.datetime.now().isoformat()}

@app.post("/api/auth/login")
def login(req: LoginReq):
    """
    PROTOTYPE ONLY - MOCK AUTHENTICATION
    This is not production authentication.
    """
    role = "officer"
    name = "Insp. R. Bhandari"
    
    bid = req.badgeId.upper()
    if "ADMIN" in bid or req.pin == "999999":
        role = "admin"
        name = "Cmdt. S. Sharma"
    elif "SUPER" in bid or req.pin == "888888":
        role = "supervisor"
        name = "Dy. Cmdt. M. Singh"
    elif len(req.pin) < 6 or len(req.badgeId) < 4:
        raise HTTPException(status_code=401, detail="Invalid credentials")
        
    return {
        "authenticated": True,
        "badgeId": bid,
        "name": name,
        "role": role,
        "checkpostId": req.checkpostId,
        "shift": "Night",
        "sessionToken": "mock-token-123"
    }

@app.post("/api/screening/extract", response_model=CommonResponse)
def extract_document(req: ExtractReq):
    return ocr.perform_ocr(req.documentImage)

@app.post("/api/screening/validate", response_model=CommonResponse)
def validate_mrz(req: ValidateReq):
    return {
        "status": "UNAVAILABLE",
        "score": None,
        "evidence": [],
        "reason": "Real MRZ validation module not integrated yet.",
        "data": None
    }

@app.post("/api/screening/detect", response_model=CommonResponse)
def detect_tampering(req: DetectReq):
    return tamper.detect_tampering(req.documentImage)

@app.post("/api/screening/verify", response_model=CommonResponse)
def verify_person(req: VerifyReq):
    return face.verify_face(req.faceImage, "dummy_doc_face")

@app.post("/api/screening/log")
def log_stage(req: LogReq):
    # Retrieve prototype mock scores gracefully
    t_score = req.tamperRes.get("score")
    v_score = req.verifyRes.get("score")
    
    if t_score is None or v_score is None:
        log_audit(req.caseRef, "UNAVAILABLE", None, "Pipeline data incomplete")
        return {"success": False, "compositeScore": None, "suggestedDecision": "UNAVAILABLE"}

    # Do not blindly subtract face from tampering. Let's make an explicit independent evaluation.
    compositeScore = t_score * 100.0
    suggestedDecision = "flag" if (compositeScore > AUTO_FLAG_RISK or v_score < FACE_MATCH_MINIMUM) else "clear"
    
    log_audit(req.caseRef, "LOGGED", compositeScore, f"Suggested: {suggestedDecision}")
    
    return {
        "success": True,
        "compositeScore": compositeScore,
        "suggestedDecision": suggestedDecision
    }

@app.post("/api/screening/decision")
def record_decision(req: DecisionReq):
    log_audit(req.caseRef, req.decision.upper(), req.finalScore, "Officer manual decision applied")
    return {
        "success": True,
        "blockNum": 49201,
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.post("/api/screening/analyze")
def analyze_full_pipeline(req: AnalyzeReq):
    """
    Orchestrates the entire screening pipeline for a single request.
    Standardized response structure.
    """
    ext_res = extract_document(ExtractReq(documentImage=req.documentImage))
    val_res = validate_mrz(ValidateReq(mrzData={"mock": "data"}))
    det_res = detect_tampering(DetectReq(documentImage=req.documentImage))
    ver_res = verify_person(VerifyReq(faceImage=req.faceImage, documentFaceRegion={"mock":"region"}))
    
    liv_res = liveness.check_liveness(req.faceImage)
    ded_res = dedup.check_deduplication(req.faceImage, req.caseRef)
    
    # Calculate Explicit Overall Risk Decision
    t_score = det_res["score"]
    v_score = ver_res["score"]
    
    overall_status = "UNAVAILABLE"
    composite_risk = None
    reasons = []

    if t_score is None or v_score is None:
        reasons.append("Waiting on real ML model integration.")
    else:
        # Score semantics:
        # Tampering: 0.0 (no risk) -> 1.0 (high risk)
        # Face: 0.0 (no match) -> 1.0 (strong match)
        composite_risk = float(t_score * 100.0)
        flagged = False
        
        if v_score < FACE_MATCH_MINIMUM:
            flagged = True
            reasons.append(f"Face match ({v_score}) below minimum threshold ({FACE_MATCH_MINIMUM}).")
        
        if composite_risk > AUTO_FLAG_RISK:
            flagged = True
            reasons.append(f"Tampering risk ({composite_risk}) exceeds auto-flag threshold ({AUTO_FLAG_RISK}).")
            
        overall_status = "FLAG" if flagged else "PASS"
        if not flagged:
            reasons.append("Pipeline checks passed.")

    # Log the audit trail
    log_audit(req.caseRef, overall_status, composite_risk, " | ".join(reasons))
    
    return {
        "status": overall_status,
        "ocr": ext_res,
        "validation": val_res,
        "tampering": det_res,
        "face": ver_res,
        "liveness": liv_res,
        "dedup": ded_res,
        "overall": {
            "riskScore": composite_risk,
            "flags": reasons
        },
        "audit": {
            "caseRef": req.caseRef,
            "timestamp": datetime.datetime.now().isoformat()
        }
    }

@app.get("/api/audit")
def get_audit():
    conn = sqlite3.connect("audit.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT 50")
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows

# ==========================================
# STATIC FILES (Frontend)
# ==========================================
# Mount the root directory to serve frontend HTML/JS/CSS.
app.mount("/", StaticFiles(directory=".", html=True), name="static")
