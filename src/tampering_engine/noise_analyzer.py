"""
Noise Variance & Wavelet Residual Forensics (Stage 04 - Detect Forensics)
Detects splicing, cloning, and photo swaps by measuring sensor noise distribution inconsistencies.
"""

from typing import Tuple, Dict, Any, List
import cv2
import numpy as np


class NoiseAnalyzer:
    """
    Analyzes high-frequency sensor noise across image patches.
    Authentic document scans share uniform sensor noise characteristics. Spliced patches (e.g., swapped photos or cloned stamps)
    display stark noise variance discrepancies.
    """

    def extract_noise_residuals(self, image: np.ndarray) -> np.ndarray:
        """
        Extracts high-frequency noise map by subtracting median filtered image from original.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        # High-pass residual via median filter subtraction
        denoised = cv2.medianBlur(gray, 3)
        residual = cv2.absdiff(gray, denoised)
        return residual

    def compute_noise_map(self, image: np.ndarray, block_size: int = 24) -> Tuple[np.ndarray, float, List[Dict[str, Any]]]:
        """
        Calculates local noise variance across overlapping blocks.
        Returns:
            - noise_heatmap: 2D array of localized noise variance.
            - noise_inconsistency_score: Global metric (0.0 to 1.0) indicating variance dispersion.
            - anomalies: List of localized high/low noise anomaly boxes.
        """
        residual = self.extract_noise_residuals(image)
        h, w = residual.shape[:2]

        noise_map = np.zeros((h // block_size, w // block_size), dtype=np.float32)

        for by, y in enumerate(range(0, h - block_size, block_size)):
            for bx, x in enumerate(range(0, w - block_size, block_size)):
                patch = residual[y:y + block_size, x:x + block_size]
                noise_map[by, bx] = float(np.var(patch))

        global_mean = float(np.mean(noise_map))
        global_std = float(np.std(noise_map)) + 1e-5

        # Score based on coefficient of variation (std / mean)
        cv = global_std / (global_mean + 1e-5)
        inconsistency_score = min(1.0, cv / 1.5)

        # Detect outlier blocks
        anomalies = []
        for by in range(noise_map.shape[0]):
            for bx in range(noise_map.shape[1]):
                val = noise_map[by, bx]
                z_score = abs(val - global_mean) / global_std
                if z_score > 3.0:
                    px = bx * block_size
                    py = by * block_size
                    anomalies.append({
                        "bbox": {"x": px, "y": py, "width": block_size, "height": block_size},
                        "z_score": round(float(z_score), 2),
                        "reason": "Sensor noise variance mismatch (Potential Photo Swap or Spliced Stamp)"
                    })

        # Resize noise map to original image resolution for visual heatmap blending
        resized_noise_map = cv2.resize(noise_map, (w, h), interpolation=cv2.INTER_CUBIC)
        # Normalize to 0-255
        norm_noise_map = cv2.normalize(resized_noise_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return norm_noise_map, round(inconsistency_score, 4), anomalies
