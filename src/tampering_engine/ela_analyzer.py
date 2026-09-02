"""
Error Level Analysis (ELA) Engine (Stage 04 - Detect Forensics)
Identifies digital manipulation by analyzing JPEG compression error differentials.
"""

import io
from typing import Tuple, Dict, Any, List
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

from src.config import (
    ELA_QUALITY, ELA_MULTIPLIER, ELA_MEAN_ERROR_DIVISOR,
    ELA_ANOMALY_WEIGHT, ELA_GRID_SIZE, ELA_THRESHOLD_STD, ELA_MIN_INTENSITY
)


class ELAAnalyzer:
    """
    Performs Error Level Analysis (ELA).
    When an image is edited, the newly inserted or modified sections have different
    compression error levels compared to the original unmodified sections.
    """

    def __init__(self, quality: int = ELA_QUALITY, multiplier: float = ELA_MULTIPLIER):
        self.quality = quality
        self.multiplier = multiplier

    def compute_ela(self, image_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Computes the Error Level Analysis image and variance metrics.
        Returns:
            - ela_image: Enhanced 3-channel color difference map.
            - ela_gray: Grayscale error intensity.
            - mean_error: Average error magnitude across the document.
        """
        # Convert BGR (OpenCV) to RGB (PIL)
        if len(image_np.shape) == 3:
            pil_img = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.fromarray(image_np)

        # Save to memory buffer at fixed JPEG quality
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=self.quality)
        buffer.seek(0)
        recompressed_img = Image.open(buffer)

        # Calculate absolute difference
        diff = ImageChops.difference(pil_img, recompressed_img)

        # Scale difference to enhance visibility of compression artifacts
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema]) if isinstance(extrema[0], tuple) else extrema[1]
        scale = self.multiplier if max_diff == 0 else min(self.multiplier, 255.0 / (max_diff + 1e-5))

        enhancer = ImageEnhance.Brightness(diff)
        ela_enhanced = enhancer.enhance(scale)

        ela_np = np.array(ela_enhanced)
        if len(ela_np.shape) == 3:
            ela_bgr = cv2.cvtColor(ela_np, cv2.COLOR_RGB2BGR)
            ela_gray = cv2.cvtColor(ela_bgr, cv2.COLOR_BGR2GRAY)
        else:
            ela_bgr = cv2.cvtColor(ela_np, cv2.COLOR_GRAY2BGR)
            ela_gray = ela_np

        mean_error = float(np.mean(ela_gray))
        return ela_bgr, ela_gray, mean_error

    def detect_anomalous_regions(
        self, 
        ela_gray: np.ndarray, 
        grid_size: int = ELA_GRID_SIZE, 
        threshold_std: float = ELA_THRESHOLD_STD
    ) -> List[Dict[str, Any]]:
        """
        Scans localized grid cells to detect patches with statistically significant ELA deviation.
        """
        h, w = ela_gray.shape[:2]
        global_mean = np.mean(ela_gray)
        global_std = np.std(ela_gray)

        threshold = global_mean + (threshold_std * global_std)
        anomalies = []

        for y in range(0, h - grid_size, grid_size // 2):
            for x in range(0, w - grid_size, grid_size // 2):
                cell = ela_gray[y:y + grid_size, x:x + grid_size]
                cell_mean = np.mean(cell)

                if cell_mean > threshold and cell_mean > ELA_MIN_INTENSITY:
                    deviation = (cell_mean - global_mean) / (global_std + 1e-5)
                    anomalies.append({
                        "bbox": {"x": x, "y": y, "width": grid_size, "height": grid_size},
                        "intensity": float(cell_mean),
                        "deviation_sigma": round(float(deviation), 2),
                        "type": "ELA_COMPRESSION_ANOMALY"
                    })

        # Merge overlapping bounding boxes
        return self._merge_overlapping_boxes(anomalies)

    def _merge_overlapping_boxes(self, anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merges nearby grid anomalies into cohesive evidence regions."""
        if not anomalies:
            return []

        boxes = [[a["bbox"]["x"], a["bbox"]["y"], a["bbox"]["x"] + a["bbox"]["width"], a["bbox"]["y"] + a["bbox"]["height"]] for a in anomalies]
        merged = []

        while boxes:
            cur = boxes.pop(0)
            merged_box = list(cur)
            i = 0
            while i < len(boxes):
                b = boxes[i]
                # Check proximity / overlap
                if (cur[0] - 20 <= b[2] and cur[2] + 20 >= b[0] and
                    cur[1] - 20 <= b[3] and cur[3] + 20 >= b[1]):
                    merged_box[0] = min(merged_box[0], b[0])
                    merged_box[1] = min(merged_box[1], b[1])
                    merged_box[2] = max(merged_box[2], b[2])
                    merged_box[3] = max(merged_box[3], b[3])
                    boxes.pop(i)
                else:
                    i += 1
            merged.append(merged_box)

        evidence_regions = []
        for b in merged:
            evidence_regions.append({
                "bbox": {
                    "x": int(b[0]), 
                    "y": int(b[1]), 
                    "width": int(b[2] - b[0]), 
                    "height": int(b[3] - b[1])
                },
                "confidence": 0.85,
                "reason": "High compression error variance indicating modified/inserted digital element (ELA Anomaly)"
            })
        return evidence_regions
