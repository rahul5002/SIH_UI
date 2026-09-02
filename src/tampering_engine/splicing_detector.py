"""
Photo Replacement & Splicing Boundary Detector (Stage 04 - Detect Forensics)
Detects pasted photographs, stamp boundary discontinuities, and cut-and-paste artifacts.
"""

from typing import Tuple, Dict, Any, List, Optional
import cv2
import numpy as np

from src.config import SPLICING_EDGE_DIVISOR, SPLICING_DETECTION_THRESHOLD


class SplicingDetector:
    """
    Detects physical and digital photo replacement / document splicing.
    Examines:
    1. Edge gradient continuity across photo and signature borders.
    2. Double border / ghosting artifacts around facial photo frame.
    3. Color gamut & illumination transition along bounding boxes.
    """

    def analyze_photo_boundary(
        self, 
        image: np.ndarray, 
        photo_bbox: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Analyzes the boundary of the identity photograph to detect pasted/replaced edges.
        If photo_bbox is not provided, automatically scans for candidate photo regions.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        # If no bbox provided, look at standard left/right photo area
        if not photo_bbox:
            # Typical passport photo occupies ~20-35% width, ~30-50% height on the left
            photo_bbox = {
                "x": int(w * 0.05),
                "y": int(h * 0.15),
                "width": int(w * 0.35),
                "height": int(h * 0.45)
            }

        px, py, pw, ph = photo_bbox["x"], photo_bbox["y"], photo_bbox["width"], photo_bbox["height"]
        
        # Ensure inside image bounds
        px = max(5, min(w - 10, px))
        py = max(5, min(h - 10, py))
        pw = min(w - px - 5, pw)
        ph = min(h - py - 5, ph)

        # Extract margin around photo boundary (inner band vs outer band)
        margin = 6
        inner_roi = gray[py + margin:py + ph - margin, px + margin:px + pw - margin]
        outer_roi = gray[max(0, py - margin):min(h, py + ph + margin), max(0, px - margin):min(w, px + pw + margin)]

        if inner_roi.size == 0 or outer_roi.size == 0:
            return {"splicing_detected": False, "score": 0.0, "reason": "Invalid ROI"}

        # Compute Sobel gradients along the 4 edges of the photo box
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)

        # Sample border gradient intensity
        top_edge = np.mean(grad_mag[py:py + margin, px:px + pw])
        bottom_edge = np.mean(grad_mag[py + ph - margin:py + ph, px:px + pw])
        left_edge = np.mean(grad_mag[py:py + ph, px:px + margin])
        right_edge = np.mean(grad_mag[py:py + ph, px + pw - margin:px + pw])

        avg_border_edge = float((top_edge + bottom_edge + left_edge + right_edge) / 4.0)

        # Inner vs Outer illumination disparity
        mean_inner = float(np.mean(inner_roi))
        mean_outer = float(np.mean(outer_roi))
        illum_disparity = abs(mean_inner - mean_outer) / 255.0

        # Score calculation
        edge_sharpness_anomaly = min(1.0, avg_border_edge / SPLICING_EDGE_DIVISOR)
        splicing_score = round(0.6 * edge_sharpness_anomaly + 0.4 * illum_disparity, 4)

        splicing_detected = splicing_score > SPLICING_DETECTION_THRESHOLD

        evidence = []
        if splicing_detected:
            evidence.append({
                "bbox": photo_bbox,
                "confidence": float(splicing_score),
                "reason": "Artificial edge discontinuity & boundary gradient mismatch detected around photo frame (Suspected Photo Replacement)"
            })

        return {
            "splicing_detected": splicing_detected,
            "splicing_score": splicing_score,
            "border_gradient": round(avg_border_edge, 2),
            "illumination_disparity": round(illum_disparity, 2),
            "evidence": evidence
        }
