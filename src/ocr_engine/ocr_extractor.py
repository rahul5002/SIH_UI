"""
OCR & Information Extraction Engine (Stage 02 - Extract)
Extracts structured identity fields from Passports, Visas, and National IDs with confidence scoring.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

from src.ocr_engine.mrz_parser import MRZParser


class OCRExtractor:
    """
    Intelligent document OCR engine combining EasyOCR, deep text segment analysis,
    and ICAO Doc 9303 MRZ parsing.
    """

    def __init__(self, languages: List[str] = ['en'], use_gpu: bool = False):
        self.languages = languages
        self.use_gpu = use_gpu
        self._reader = None

    @property
    def reader(self):
        if self._reader is None and EASYOCR_AVAILABLE:
            self._reader = easyocr.Reader(self.languages, gpu=self.use_gpu, verbose=False)
        return self._reader

    def extract_raw_ocr(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs OCR model on image and extracts text bounding boxes, detected string, and confidence.
        """
        if not EASYOCR_AVAILABLE or self.reader is None:
            return []

        # Convert to RGB if BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image

        results = self.reader.readtext(rgb)
        parsed_results = []
        for bbox, text, conf in results:
            # bbox is list of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            pts = np.array(bbox, dtype=np.int32)
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)

            parsed_results.append({
                "text": text.strip(),
                "confidence": round(float(conf), 4),
                "bbox": {
                    "x": int(x_min),
                    "y": int(y_min),
                    "width": int(x_max - x_min),
                    "height": int(y_max - y_min),
                    "polygon": pts.tolist()
                }
            })
        return parsed_results

    def separate_mrz_and_viz(self, ocr_items: List[Dict[str, Any]], img_height: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Separates OCR elements into Visual Inspection Zone (VIZ) and Machine Readable Zone (MRZ)
        based on vertical position (bottom 35% typically holds MRZ) and '<' character count.
        """
        mrz_items = []
        viz_items = []

        mrz_threshold_y = img_height * 0.60

        for item in ocr_items:
            text = item["text"]
            y_pos = item["bbox"]["y"]

            # Strong indicator of MRZ: contains '<' or located in bottom region with typical MRZ format
            if ('<' in text or y_pos >= mrz_threshold_y) and len(re.sub(r'[^A-Z0-9<]', '', text.upper())) >= 15:
                mrz_items.append(item)
            else:
                viz_items.append(item)

        # Sort MRZ top to bottom
        mrz_items = sorted(mrz_items, key=lambda it: it["bbox"]["y"])
        return viz_items, mrz_items

    def extract_viz_fields(self, viz_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extracts key visual fields (Name, Document Number, DOB, Expiry, Nationality, Gender)
        from the Visual Inspection Zone using regex and positional heuristics.
        """
        viz_text = " \n ".join([it["text"] for it in viz_items])
        confidences = [it["confidence"] for it in viz_items]
        avg_viz_conf = float(np.mean(confidences)) if confidences else 0.0

        # Regex patterns for common fields
        doc_num_pattern = re.compile(r'\b([A-Z][0-9]{7,8}|[A-Z0-9]{8,10})\b')
        date_pattern = re.compile(r'\b(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4}|\d{4}[\/\-\.]\d{2}[\/\-\.]\d{2})\b')
        gender_pattern = re.compile(r'\b(SEX|GENDER)[\s:\/\-]*([MFX])\b', re.IGNORECASE)

        # Search patterns
        doc_matches = doc_num_pattern.findall(viz_text)
        date_matches = date_pattern.findall(viz_text)
        gender_matches = gender_pattern.findall(viz_text)

        extracted_doc_num = doc_matches[0] if doc_matches else None
        gender = gender_matches[0][1].upper() if gender_matches else None

        # Positional field parsing
        name_candidate = None
        for i, item in enumerate(viz_items):
            txt = item["text"].upper()
            if "NAME" in txt or "SURNAME" in txt or "GIVEN" in txt:
                # Look at next item or right side
                if i + 1 < len(viz_items):
                    name_candidate = viz_items[i + 1]["text"]
                    break

        return {
            "viz_document_number": extracted_doc_num,
            "viz_name": name_candidate,
            "viz_gender": gender,
            "viz_dates_detected": date_matches,
            "viz_raw_text": viz_text,
            "viz_confidence": round(avg_viz_conf, 4)
        }

    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Full OCR pipeline:
        1. Extract all text tokens with bboxes and confidences.
        2. Detect and parse MRZ zone.
        3. Extract Visual Inspection Zone (VIZ) fields.
        4. Fuse fields and compute comprehensive confidence scores.
        """
        h, w = image.shape[:2]
        raw_ocr = self.extract_raw_ocr(image)

        if not raw_ocr:
            return {
                "success": False,
                "message": "No text detected by OCR engine.",
                "fields": {},
                "mrz": None,
                "confidence_scores": {"overall_score": 0.0},
                "ocr_tokens": []
            }

        viz_items, mrz_items = self.separate_mrz_and_viz(raw_ocr, h)
        
        # Parse MRZ
        mrz_candidate_lines = [item["text"] for item in mrz_items]
        mrz_data = MRZParser.parse_auto(mrz_candidate_lines)

        # Parse VIZ
        viz_data = self.extract_viz_fields(viz_items)

        # Merge extracted fields (MRZ is authoritative for MRTD, supplemented by VIZ)
        final_fields = {}
        field_confidences = {}

        if mrz_data:
            final_fields["document_type"] = mrz_data["document_type"]
            final_fields["document_code"] = mrz_data["document_code"]
            final_fields["issuing_country"] = mrz_data["issuing_country"]
            final_fields["full_name"] = mrz_data["full_name"]
            final_fields["surname"] = mrz_data["surname"]
            final_fields["given_names"] = mrz_data["given_names"]
            final_fields["document_number"] = mrz_data["document_number"]
            final_fields["nationality"] = mrz_data["nationality"]
            final_fields["date_of_birth"] = mrz_data["date_of_birth"]
            final_fields["gender"] = mrz_data["gender"]
            final_fields["date_of_expiry"] = mrz_data["date_of_expiry"]
            
            mrz_conf = np.mean([it["confidence"] for it in mrz_items]) if mrz_items else 0.85
            for k in final_fields:
                field_confidences[k] = round(float(mrz_conf), 4)
        else:
            final_fields["document_type"] = "UNKNOWN_OR_NON_MRZ"
            final_fields["document_number"] = viz_data.get("viz_document_number")
            final_fields["full_name"] = viz_data.get("viz_name")
            final_fields["gender"] = viz_data.get("viz_gender")
            final_fields["date_of_birth"] = viz_data.get("viz_dates_detected", [None])[0] if viz_data.get("viz_dates_detected") else None
            final_fields["date_of_expiry"] = viz_data.get("viz_dates_detected", [None, None])[1] if len(viz_data.get("viz_dates_detected", [])) > 1 else None
            
            for k in final_fields:
                field_confidences[k] = viz_data.get("viz_confidence", 0.70)

        # Overall OCR confidence
        all_confs = [it["confidence"] for it in raw_ocr]
        overall_conf = float(np.mean(all_confs)) if all_confs else 0.0

        return {
            "success": True,
            "fields": final_fields,
            "field_confidences": field_confidences,
            "overall_confidence": round(overall_conf, 4),
            "mrz": mrz_data,
            "viz": viz_data,
            "ocr_tokens": raw_ocr
        }
