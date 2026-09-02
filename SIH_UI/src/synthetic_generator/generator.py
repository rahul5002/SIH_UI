"""
Synthetic Document Generator (Passports TD3, Visas TD2, National IDs TD1)
Generates high-fidelity identity documents with Guilloche patterns, ICAO Doc 9303 MRZ, and security stamps.
"""

from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
from datetime import datetime, timedelta

from src.ocr_engine.mrz_parser import MRZParser


class SyntheticDocumentGenerator:
    """
    Generates realistic synthetic travel and identity documents with verifiable ICAO Doc 9303 MRZ.
    """

    SAMPLE_NAMES = [
        ("SHARMA", "RAHUL"),
        ("VERMA", "ANANYA"),
        ("PATEL", "ARAVIND"),
        ("SINGH", "PRIYA"),
        ("GUPTA", "VIKRAM"),
        ("KHAN", "SAMEER"),
        ("MENON", "DEEPA"),
        ("DESHMUKH", "ROHIT")
    ]

    SAMPLE_COUNTRIES = [
        ("IND", "REPUBLIC OF INDIA"),
        ("USA", "UNITED STATES OF AMERICA"),
        ("GBR", "UNITED KINGDOM"),
        ("SGP", "REPUBLIC OF SINGAPORE"),
        ("DEU", "FEDERAL REPUBLIC OF GERMANY")
    ]

    def _draw_guilloche_pattern(self, draw: ImageDraw.Draw, width: int, height: int, color=(210, 225, 240)):
        """Draws complex mathematical curves simulating passport security background patterns."""
        cx, cy = width // 2, height // 2
        for r in range(50, max(width, height), 25):
            points = []
            for angle in range(0, 360, 5):
                rad = np.radians(angle)
                wave = np.sin(rad * 8) * 8
                x = cx + (r + wave) * np.cos(rad)
                y = cy + (r + wave) * np.sin(rad)
                points.append((x, y))
            if len(points) > 2:
                draw.line(points + [points[0]], fill=color, width=1)

    def _create_avatar_photo(self, width: int = 240, height: int = 300, gender: str = "M") -> np.ndarray:
        """Draws realistic placeholder portrait photograph for document holder."""
        avatar = np.ones((height, width, 3), dtype=np.uint8) * 235  # Off-white studio background
        
        # Skin tone
        skin_color = (165, 195, 225) if gender == "M" else (175, 205, 235)  # BGR
        
        # Draw torso / shirt
        torso_color = (90, 60, 40) if gender == "M" else (70, 90, 150)
        cv2.ellipse(avatar, (width // 2, height + 40), (100, 110), 0, 0, 360, torso_color, -1)
        
        # Draw Neck
        cv2.rectangle(avatar, (width // 2 - 25, height // 2 + 10), (width // 2 + 25, height // 2 + 65), skin_color, -1)
        
        # Draw Head / Face Oval
        head_center = (width // 2, height // 2 - 20)
        cv2.ellipse(avatar, head_center, (60, 75), 0, 0, 360, skin_color, -1)
        
        # Hair
        hair_color = (30, 25, 20)
        cv2.ellipse(avatar, (width // 2, height // 2 - 45), (65, 45), 0, 180, 360, hair_color, -1)
        if gender == "F":
            # Longer hair sides
            cv2.ellipse(avatar, (width // 2 - 55, height // 2), (20, 55), 0, 0, 360, hair_color, -1)
            cv2.ellipse(avatar, (width // 2 + 55, height // 2), (20, 55), 0, 0, 360, hair_color, -1)

        # Eyes & Eyebrows
        cv2.circle(avatar, (width // 2 - 22, height // 2 - 25), 5, (40, 30, 20), -1)
        cv2.circle(avatar, (width // 2 + 22, height // 2 - 25), 5, (40, 30, 20), -1)
        cv2.line(avatar, (width // 2 - 32, height // 2 - 35), (width // 2 - 12, height // 2 - 35), hair_color, 2)
        cv2.line(avatar, (width // 2 + 12, height // 2 - 35), (width // 2 + 32, height // 2 - 35), hair_color, 2)

        # Nose & Mouth
        cv2.line(avatar, (width // 2, height // 2 - 20), (width // 2 - 3, height // 2), (130, 160, 190), 2)
        cv2.line(avatar, (width // 2 - 3, height // 2), (width // 2 + 5, height // 2), (130, 160, 190), 2)
        cv2.ellipse(avatar, (width // 2, height // 2 + 20), (16, 5), 0, 0, 180, (120, 130, 190), 2)

        # Add subtle natural camera blur
        avatar = cv2.GaussianBlur(avatar, (3, 3), 0.5)
        return avatar

    def generate_passport(
        self,
        surname: Optional[str] = None,
        given_names: Optional[str] = None,
        doc_number: Optional[str] = None,
        country_code: str = "IND",
        country_name: str = "REPUBLIC OF INDIA",
        gender: str = "M",
        dob: Optional[str] = None,
        expiry: Optional[str] = None,
        width: int = 1000,
        height: int = 700
    ) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
        """
        Generates a pristine, genuine TD3 Passport document with mathematically valid ICAO 9303 MRZ.
        Returns:
            - doc_image: BGR numpy image of passport page.
            - ground_truth: Dictionary of ground truth fields and checksums.
            - avatar_photo: Standalone photo used for biometric matching.
        """
        # Random defaults if not provided
        if not surname or not given_names:
            pair = random.choice(self.SAMPLE_NAMES)
            surname, given_names = pair[0], pair[1]

        if not doc_number:
            doc_number = f"Z{random.randint(1000000, 9999999)}"

        if not dob:
            dob_dt = datetime.now() - timedelta(days=random.randint(7500, 18000))
            dob = dob_dt.strftime("%Y-%m-%d")
            dob_mrz = dob_dt.strftime("%y%m%d")
        else:
            dob_dt = datetime.strptime(dob, "%Y-%m-%d")
            dob_mrz = dob_dt.strftime("%y%m%d")

        if not expiry:
            exp_dt = datetime.now() + timedelta(days=random.randint(365, 3650))
            expiry = exp_dt.strftime("%Y-%m-%d")
            exp_mrz = exp_dt.strftime("%y%m%d")
        else:
            exp_dt = datetime.strptime(expiry, "%Y-%m-%d")
            exp_mrz = exp_dt.strftime("%y%m%d")

        # Compute ICAO 9303 Check Digits
        doc_clean = doc_number.replace('<', '')
        doc_cd = str(MRZParser.compute_check_digit(doc_clean))
        dob_cd = str(MRZParser.compute_check_digit(dob_mrz))
        exp_cd = str(MRZParser.compute_check_digit(exp_mrz))

        # Composite check digit
        comp_str = (doc_clean + '<' * (9 - len(doc_clean))) + doc_cd + dob_mrz + dob_cd + exp_mrz + exp_cd + ('<' * 14) + '<'
        comp_cd = str(MRZParser.compute_check_digit(comp_str[:43]))

        # Assemble MRZ lines (TD3: 2 lines of 44 chars)
        # Line 1: P<INDSHARMA<<RAHUL<<<<<<<<<<<<<<<<<<<<<<<<<<
        name_field = f"{surname}<<{given_names.replace(' ', '<')}"
        mrz_line1 = f"P<{country_code}{name_field}"
        mrz_line1 = (mrz_line1 + '<' * 44)[:44]

        # Line 2: Z1234567<8IND9205141M2911204<<<<<<<<<<<<<<0
        mrz_line2 = f"{doc_clean}{'<' * (9 - len(doc_clean))}{doc_cd}{country_code}{dob_mrz}{dob_cd}{gender}{exp_mrz}{exp_cd}{'<' * 14}{'<'}0"
        # compute actual composite check
        comp_data = mrz_line2[0:10] + mrz_line2[13:20] + mrz_line2[21:43]
        actual_comp_cd = str(MRZParser.compute_check_digit(comp_data))
        mrz_line2 = mrz_line2[:43] + actual_comp_cd

        # Create Canvas (Pillow)
        pil_img = Image.new("RGB", (width, height), (245, 248, 252))
        draw = ImageDraw.Draw(pil_img)

        # Draw Security Background & Borders
        self._draw_guilloche_pattern(draw, width, height, color=(220, 232, 245))
        draw.rectangle([(20, 20), (width - 20, height - 20)], outline=(180, 195, 215), width=3)
        draw.rectangle([(25, 25), (width - 25, height - 25)], outline=(200, 215, 230), width=1)

        # Draw Header
        draw.text((360, 45), country_name, fill=(20, 40, 80))
        draw.text((430, 75), "PASSPORT / PASSEPORT", fill=(80, 90, 110))

        # Draw Avatar Photo
        avatar = self._create_avatar_photo(width=220, height=280, gender=gender)
        avatar_pil = Image.fromarray(cv2.cvtColor(avatar, cv2.COLOR_BGR2RGB))
        photo_box = (50, 120)
        pil_img.paste(avatar_pil, photo_box)
        draw.rectangle([(48, 118), (48 + 224, 118 + 284)], outline=(160, 175, 195), width=2)

        # Draw Visual Inspection Zone (VIZ) Text
        viz_x = 320
        fields_y = [
            ("Type / Type", "P", 130),
            ("Country Code / Code pays", country_code, 130 + 150),
            ("Passport No. / No. du passeport", doc_number, 130 + 350),
            ("Surname / Nom", surname, 190),
            ("Given Names / Prénoms", given_names, 245),
            ("Nationality / Nationalité", country_name.split()[-1], 300),
            ("Date of Birth / Date de naissance", dob, 355),
            ("Sex / Sexe", gender, 355 + 220),
            ("Date of Expiry / Date d'expiration", expiry, 410)
        ]

        # Draw labels & values
        for item in fields_y:
            lbl, val, x_pos = item[0], item[1], item[2] if len(item) == 4 else viz_x
            y_pos = item[2] if len(item) == 3 else item[3]
            draw.text((x_pos, y_pos), lbl, fill=(110, 125, 145))
            draw.text((x_pos, y_pos + 18), str(val), fill=(10, 20, 40))

        # Draw Security Stamp (Circular Immigration Crest)
        stamp_center = (width - 150, 240)
        draw.ellipse([(stamp_center[0] - 65, stamp_center[1] - 65), (stamp_center[0] + 65, stamp_center[1] + 65)], outline=(190, 70, 70), width=2)
        draw.ellipse([(stamp_center[0] - 55, stamp_center[1] - 55), (stamp_center[0] + 55, stamp_center[1] + 55)], outline=(190, 70, 70), width=1)
        draw.text((stamp_center[0] - 40, stamp_center[1] - 15), "IMMIGRATION", fill=(190, 70, 70))
        draw.text((stamp_center[0] - 25, stamp_center[1] + 5), country_code, fill=(190, 70, 70))

        # Draw Machine Readable Zone (MRZ) at Bottom
        draw.line([(30, height - 170), (width - 30, height - 170)], fill=(180, 195, 215), width=2)
        
        # Monospace MRZ characters
        mrz_y1 = height - 140
        mrz_y2 = height - 85
        char_spacing = (width - 100) / 44.0

        for i, ch in enumerate(mrz_line1):
            draw.text((50 + i * char_spacing, mrz_y1), ch, fill=(15, 15, 25))

        for i, ch in enumerate(mrz_line2):
            draw.text((50 + i * char_spacing, mrz_y2), ch, fill=(15, 15, 25))

        # Convert back to OpenCV BGR
        doc_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Ground truth bundle
        ground_truth = {
            "document_type": "PASSPORT",
            "format": "TD3",
            "is_tampered": False,
            "tampering_type": "NONE",
            "country_code": country_code,
            "country_name": country_name,
            "surname": surname,
            "given_names": given_names,
            "full_name": f"{given_names} {surname}",
            "document_number": doc_number,
            "date_of_birth": dob,
            "gender": gender,
            "date_of_expiry": expiry,
            "photo_bbox": {"x": photo_box[0], "y": photo_box[1], "width": 220, "height": 280},
            "mrz_line1": mrz_line1,
            "mrz_line2": mrz_line2,
            "checksums_valid": True
        }

        return doc_bgr, ground_truth, avatar
