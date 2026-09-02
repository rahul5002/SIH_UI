"""
Font & Character Forensics Analyzer (Stage 04 - Detect Forensics)
Detects digital text manipulation: altered dates of birth, modified expiry years, pasted numbers.
"""

from typing import Dict, Any, List
import cv2
import numpy as np


class FontForensicsAnalyzer:
    """
    Analyzes font characteristics across text lines in identity documents.
    Authentic official documents maintain strict typesetting, baseline alignment, uniform stroke widths, and consistent font kerning.
    Modified/overwritten text exhibits baseline jitter, anomalous character height, and stroke thickness jumps.
    """

    def analyze_text_tokens(self, ocr_tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes geometric consistency of OCR bounding boxes.
        Groups text into horizontal lines and checks for height and baseline anomalies.
        """
        if not ocr_tokens or len(ocr_tokens) < 3:
            return {
                "font_anomaly_detected": False,
                "anomaly_score": 0.0,
                "flagged_tokens": []
            }

        flagged = []
        
        # Sort tokens by Y coordinate to cluster into lines
        sorted_tokens = sorted(ocr_tokens, key=lambda t: t["bbox"]["y"])
        
        # Group into lines (within 15px vertical threshold)
        lines = []
        current_line = [sorted_tokens[0]]

        for tok in sorted_tokens[1:]:
            prev_y = current_line[-1]["bbox"]["y"]
            if abs(tok["bbox"]["y"] - prev_y) <= 15:
                current_line.append(tok)
            else:
                lines.append(current_line)
                current_line = [tok]
        if current_line:
            lines.append(current_line)

        # Inspect each line with >= 2 tokens
        for line in lines:
            if len(line) < 2:
                continue

            heights = [t["bbox"]["height"] for t in line]
            baselines = [t["bbox"]["y"] + t["bbox"]["height"] for t in line]

            mean_h = np.mean(heights)
            std_h = np.std(heights)

            mean_base = np.mean(baselines)
            std_base = np.std(baselines)

            # Look for rogue tokens in line
            for t in line:
                h_diff = abs(t["bbox"]["height"] - mean_h)
                base_diff = abs((t["bbox"]["y"] + t["bbox"]["height"]) - mean_base)

                # Check if this token deviates significantly
                if (h_diff > 12 and h_diff > 0.4 * mean_h) or (base_diff > 10):
                    flagged.append({
                        "text": t["text"],
                        "bbox": t["bbox"],
                        "height_deviation": round(float(h_diff), 2),
                        "baseline_jitter": round(float(base_diff), 2),
                        "reason": f"Font baseline/height anomaly on token '{t['text']}' (Suspected Altered Text or Pasted Digit)"
                    })

        anomaly_score = min(1.0, len(flagged) * 0.25)

        return {
            "font_anomaly_detected": len(flagged) > 0,
            "anomaly_score": round(anomaly_score, 4),
            "flagged_tokens": flagged
        }
