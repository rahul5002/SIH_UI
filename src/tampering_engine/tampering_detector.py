"""
Master Tampering & Forgery Detection Engine (Stage 04 - Detect ⭐)
Fuses ELA, Noise Variance, Splicing Boundary Analysis, Font Forensics, and Metadata into unified risk assessment.
"""

import base64
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

from src.tampering_engine.ela_analyzer import ELAAnalyzer
from src.tampering_engine.noise_analyzer import NoiseAnalyzer
from src.tampering_engine.splicing_detector import SplicingDetector
from src.tampering_engine.font_forensics import FontForensicsAnalyzer
from src.tampering_engine.metadata_analyzer import MetadataAnalyzer
from src.config import (
    TAMPER_WEIGHT_ELA, TAMPER_WEIGHT_NOISE, TAMPER_WEIGHT_SPLICING,
    TAMPER_WEIGHT_FONT, TAMPER_WEIGHT_METADATA,
    TAMPER_THRESHOLD_CLEAN, TAMPER_THRESHOLD_LOW_RISK, TAMPER_THRESHOLD_HIGH_RISK,
    ELA_MEAN_ERROR_DIVISOR, ELA_ANOMALY_WEIGHT
)


class TamperingDetector:
    """
    State-of-the-art multi-modal document forgery & digital tampering detection suite.
    """

    def __init__(self):
        self.ela_analyzer = ELAAnalyzer()
        self.noise_analyzer = NoiseAnalyzer()
        self.splicing_detector = SplicingDetector()
        self.font_analyzer = FontForensicsAnalyzer()
        self.metadata_analyzer = MetadataAnalyzer()

    def generate_composite_heatmap(
        self, 
        original_image: np.ndarray, 
        ela_gray: np.ndarray, 
        noise_map: np.ndarray
    ) -> Tuple[np.ndarray, str]:
        """
        Creates a color-mapped visual heatmap blending ELA compression differences and noise variance.
        Returns:
            - composite_bgr: 3-channel visual overlay image with hot colors (red/yellow) on suspicious areas.
            - base64_png: Base64 data URI string for direct API and frontend rendering.
        """
        h, w = original_image.shape[:2]
        
        # Ensure identical sizes
        ela_resized = cv2.resize(ela_gray, (w, h))
        noise_resized = cv2.resize(noise_map, (w, h))

        # Normalize and fuse
        fused = cv2.addWeighted(ela_resized, 0.60, noise_resized, 0.40, 0)
        
        # Apply Jet / Turbo colormap for high-contrast thermal visualization
        colormap = cv2.applyColorMap(fused, cv2.COLORMAP_JET)

        # Blend with original image at 40% transparency
        overlay = cv2.addWeighted(original_image, 0.60, colormap, 0.40, 0)

        # Encode to PNG Base64
        _, buffer = cv2.imencode('.png', overlay)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        data_uri = f"data:image/png;base64,{b64_str}"

        return overlay, data_uri

    def detect(
        self, 
        image: np.ndarray, 
        image_bytes: Optional[bytes] = None,
        ocr_tokens: Optional[List[Dict[str, Any]]] = None,
        photo_bbox: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Runs comprehensive multi-layer forensic analysis.
        """
        h, w = image.shape[:2]

        # 1. Error Level Analysis (ELA)
        ela_bgr, ela_gray, mean_ela_error = self.ela_analyzer.compute_ela(image)
        ela_anomalies = self.ela_analyzer.detect_anomalous_regions(ela_gray)
        ela_score = min(1.0, (mean_ela_error / ELA_MEAN_ERROR_DIVISOR) + (len(ela_anomalies) * ELA_ANOMALY_WEIGHT))

        # 2. Noise Variance Residuals
        noise_map, noise_score, noise_anomalies = self.noise_analyzer.compute_noise_map(image)

        # 3. Photo Replacement & Splicing Edges
        splicing_res = self.splicing_detector.analyze_photo_boundary(image, photo_bbox)
        splicing_score = splicing_res["splicing_score"]

        # 4. Font & Character Consistency
        font_res = self.font_analyzer.analyze_text_tokens(ocr_tokens or [])
        font_score = font_res["anomaly_score"]

        # 5. Metadata Forensics
        meta_res = self.metadata_analyzer.analyze_bytes(image_bytes) if image_bytes else {"is_suspicious": False, "risk_score": 0.0, "findings": []}
        meta_score = meta_res["risk_score"]

        # Weighted Ensemble Risk Score (0.0 to 1.0 -> 0 to 100 scale)
        weights = {
            "ela": TAMPER_WEIGHT_ELA,
            "noise": TAMPER_WEIGHT_NOISE,
            "splicing": TAMPER_WEIGHT_SPLICING,
            "font": TAMPER_WEIGHT_FONT,
            "metadata": TAMPER_WEIGHT_METADATA
        }

        combined_score = (
            ela_score * weights["ela"] +
            noise_score * weights["noise"] +
            splicing_score * weights["splicing"] +
            font_score * weights["font"] +
            meta_score * weights["metadata"]
        )

        tampering_percentage = round(float(combined_score * 100), 2)

        # Classify Risk Level
        if tampering_percentage < TAMPER_THRESHOLD_CLEAN:
            risk_level = "CLEAN"
            verdict = "GENUINE_DOCUMENT"
        elif tampering_percentage < TAMPER_THRESHOLD_LOW_RISK:
            risk_level = "LOW_RISK"
            verdict = "MINOR_IRREGULARITIES"
        elif tampering_percentage < TAMPER_THRESHOLD_HIGH_RISK:
            risk_level = "HIGH_RISK"
            verdict = "SUSPECTED_FORGERY"
        else:
            risk_level = "CRITICAL_RISK"
            verdict = "CONFIRMED_TAMPERED"

        # Aggregate Evidence Bounding Boxes
        evidence_regions = []
        evidence_regions.extend(ela_anomalies)
        evidence_regions.extend(splicing_res.get("evidence", []))
        for tok in font_res.get("flagged_tokens", []):
            evidence_regions.append({
                "bbox": tok["bbox"],
                "confidence": 0.80,
                "reason": tok["reason"]
            })

        # Generate Visual Heatmap Overlay
        heatmap_overlay, heatmap_b64 = self.generate_composite_heatmap(image, ela_gray, noise_map)

        # Generate Human-Readable Explainability Reasons
        reasons = []
        if splicing_res["splicing_detected"]:
            reasons.append("Spliced photo replacement detected with edge gradient discontinuity")
        if len(ela_anomalies) > 0:
            reasons.append(f"{len(ela_anomalies)} localized digital compression/editing anomalies detected via ELA")
        if noise_score > 0.4:
            reasons.append("Irregular sensor noise distribution across document zones")
        if font_res["font_anomaly_detected"]:
            reasons.append(f"{len(font_res['flagged_tokens'])} altered characters/numbers detected with baseline jitter")
        if meta_res["is_suspicious"]:
            reasons.extend(meta_res["findings"])
        if not reasons:
            reasons.append("Document displays uniform sensor noise, standard font alignment, and intact compression profile.")

        return {
            "tampering_score": tampering_percentage,
            "risk_level": risk_level,
            "verdict": verdict,
            "tampering_detected": tampering_percentage >= TAMPER_THRESHOLD_LOW_RISK,
            "module_scores": {
                "ela_score": round(ela_score * 100, 2),
                "noise_inconsistency": round(noise_score * 100, 2),
                "splicing_boundary_score": round(splicing_score * 100, 2),
                "font_anomaly_score": round(font_score * 100, 2),
                "metadata_risk_score": round(meta_score * 100, 2)
            },
            "evidence_regions": evidence_regions,
            "reasons": reasons,
            "heatmap_base64": heatmap_b64
        }
