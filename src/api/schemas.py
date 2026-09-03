"""
Pydantic Schemas for PRAMAAN AI REST API
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class EvidenceRegion(BaseModel):
    bbox: BoundingBox
    confidence: float
    reason: str


class OCRFields(BaseModel):
    document_type: Optional[str] = None
    document_code: Optional[str] = None
    issuing_country: Optional[str] = None
    full_name: Optional[str] = None
    surname: Optional[str] = None
    given_names: Optional[str] = None
    document_number: Optional[str] = None
    nationality: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    date_of_expiry: Optional[str] = None


class ChecksumResult(BaseModel):
    document_number_valid: bool = False
    date_of_birth_valid: bool = False
    date_of_expiry_valid: bool = False
    composite_valid: bool = False
    all_valid: bool = False


class ValidationReport(BaseModel):
    validation_score: float
    verdict: str
    risk_level: str
    is_expired: bool
    calculated_age: Optional[int] = None
    violations: List[str] = []


class TamperingReport(BaseModel):
    tampering_score: float
    risk_level: str
    verdict: str
    tampering_detected: bool
    module_scores: Dict[str, float]
    evidence_regions: List[Dict[str, Any]]
    reasons: List[str]
    heatmap_base64: Optional[str] = None


class FaceVerificationReport(BaseModel):
    success: bool
    is_match: bool
    similarity_score: float
    verdict: str
    threshold: float
    liveness_score: Optional[float] = None
    doc_face_base64: Optional[str] = None
    live_face_base64: Optional[str] = None


class FullScreeningResponse(BaseModel):
    screening_id: str
    timestamp: str
    overall_risk_score: float
    overall_verdict: str
    risk_level: str
    extracted_fields: OCRFields
    field_confidences: Dict[str, float]
    ocr_confidence: float
    validation_report: ValidationReport
    tampering_report: TamperingReport
    face_report: Optional[FaceVerificationReport] = None
    processing_time_ms: float
