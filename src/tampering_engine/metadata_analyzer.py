"""
Metadata & Digital Container Forensics (Stage 04 - Detect Forensics)
Scans EXIF tags and file headers for digital editing software footprints (Photoshop, GIMP, Canva, ExifTool).
"""

import io
from typing import Dict, Any, List
from PIL import Image, ExifTags


class MetadataAnalyzer:
    """
    Forensic analysis of image metadata and digital encoding parameters.
    """

    SUSPICIOUS_SOFTWARE_KEYWORDS = [
        "photoshop", "gimp", "canva", "adobe", "paint.net", "pixlr", 
        "lightroom", "picsart", "coreldraw", "snapseed", "affinity"
    ]

    def analyze_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extracts EXIF metadata, software tags, and detects editing traces from raw image bytes.
        """
        findings = []
        software_detected = None
        has_exif = False
        parsed_exif = {}

        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            exif_raw = pil_img.getexif()

            if exif_raw:
                has_exif = True
                for tag_id, val in exif_raw.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    parsed_exif[tag_name] = str(val)

                    if tag_name.lower() in ["software", "processingsoftware", "imagedescription"]:
                        val_str = str(val).lower()
                        for kw in self.SUSPICIOUS_SOFTWARE_KEYWORDS:
                            if kw in val_str:
                                software_detected = str(val)
                                findings.append(f"EDITING_SOFTWARE_DETECTED: Created/Modified with '{val}'")
                                break

            # Raw byte string scan for software metadata markers
            raw_str = image_bytes[:4096].lower()
            for kw in self.SUSPICIOUS_SOFTWARE_KEYWORDS:
                if kw.encode('utf-8') in raw_str:
                    if not software_detected:
                        software_detected = kw.capitalize()
                        findings.append(f"SOFTWARE_SIGNATURE_FOUND: Signature '{kw}' found in file header")

        except Exception as e:
            findings.append(f"METADATA_PARSE_ERROR: {str(e)}")

        is_suspicious = len(findings) > 0
        risk_score = 0.85 if software_detected else 0.0

        return {
            "has_exif": has_exif,
            "software_detected": software_detected,
            "is_suspicious": is_suspicious,
            "risk_score": risk_score,
            "findings": findings,
            "parsed_exif": parsed_exif
        }
