"""
Tampering & Forgery Injector
Creates realistic synthetic forgery samples: Photo Replacement, Altered Dates, Fake Stamps, and Checksum Mismatches.
"""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw
import random
import copy

from src.synthetic_generator.generator import SyntheticDocumentGenerator


class TamperingInjector:
    """
    Injects physical and digital tampering artifacts into synthetic documents
    to benchmark and evaluate the AI screening pipeline.
    """

    def __init__(self):
        self.generator = SyntheticDocumentGenerator()

    def inject_photo_replacement(
        self, 
        clean_doc: np.ndarray, 
        ground_truth: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Scenario 1: Spliced Photo Replacement.
        Replaces the genuine face with an impostor avatar with sharp splicing seams and compression mismatch.
        """
        tampered_doc = clean_doc.copy()
        gt = copy.deepcopy(ground_truth)

        # Create different impostor avatar (opposite gender or different hair/skin)
        impostor_gender = "F" if gt.get("gender") == "M" else "M"
        impostor_avatar = self.generator._create_avatar_photo(width=220, height=280, gender=impostor_gender)

        # Add heavy compression artifacts to impostor photo to simulate spliced image from different source
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 35]
        _, encimg = cv2.imencode('.jpg', impostor_avatar, encode_param)
        spliced_photo = cv2.imdecode(encimg, 1)

        p_box = gt["photo_bbox"]
        px, py, pw, ph = p_box["x"], p_box["y"], p_box["width"], p_box["height"]

        # Paste spliced photo onto document
        tampered_doc[py:py + ph, px:px + pw] = spliced_photo

        # Draw subtle artificial border edge discontinuity
        cv2.rectangle(tampered_doc, (px - 1, py - 1), (px + pw + 1, py + ph + 1), (120, 110, 100), 2)

        gt["is_tampered"] = True
        gt["tampering_type"] = "PHOTO_REPLACEMENT"
        gt["tampering_target"] = "PHOTO"
        gt["tampering_bbox"] = p_box
        gt["expected_verdict"] = "FORGERY_DETECTED"

        return tampered_doc, gt

    def inject_date_alteration(
        self, 
        clean_doc: np.ndarray, 
        ground_truth: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Scenario 2: Visual Expiry Date Alteration.
        Overwrites the printed visual expiry date with a patched year (e.g. extending expiry by 5 years)
        while the MRZ still reflects the original expired year (causing both font anomaly and cross-zone mismatch).
        """
        tampered_doc = clean_doc.copy()
        gt = copy.deepcopy(ground_truth)

        # Patch region over Expiry Date in VIZ
        patch_x, patch_y, patch_w, patch_h = 320, 425, 200, 30
        
        # Draw background patch with slightly different shade (tampering artifact)
        cv2.rectangle(tampered_doc, (patch_x, patch_y), (patch_x + patch_w, patch_y + patch_h), (240, 243, 248), -1)

        # Altered future date
        fake_expiry = "2032-12-31"
        pil_img = Image.fromarray(cv2.cvtColor(tampered_doc, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        # Draw altered date with slight baseline offset and different font color
        draw.text((patch_x + 5, patch_y + 4), fake_expiry, fill=(5, 5, 25))

        tampered_doc = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        gt["is_tampered"] = True
        gt["tampering_type"] = "TEXT_MANIPULATION_DATE"
        gt["tampering_target"] = "DATE_OF_EXPIRY"
        gt["original_expiry"] = gt["date_of_expiry"]
        gt["altered_expiry"] = fake_expiry
        gt["tampering_bbox"] = {"x": patch_x, "y": patch_y, "width": patch_w, "height": patch_h}
        gt["expected_verdict"] = "FORGERY_DETECTED"

        return tampered_doc, gt

    def inject_fake_stamp(
        self, 
        clean_doc: np.ndarray, 
        ground_truth: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Scenario 3: Fake / Cloned Visa Stamp.
        Superimposes a forged secondary immigration stamp with unnatural noise & edge transparency.
        """
        tampered_doc = clean_doc.copy()
        gt = copy.deepcopy(ground_truth)

        stamp_x, stamp_y = 650, 380
        stamp_radius = 55

        # Draw forged stamp
        cv2.circle(tampered_doc, (stamp_x, stamp_y), stamp_radius, (30, 30, 210), 3)
        cv2.circle(tampered_doc, (stamp_x, stamp_y), stamp_radius - 8, (30, 30, 210), 1)
        cv2.putText(tampered_doc, "ENTRY GRANTED", (stamp_x - 45, stamp_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 30, 210), 1)
        cv2.putText(tampered_doc, "BOMBAY AIRPORT", (stamp_x - 48, stamp_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 30, 210), 1)

        gt["is_tampered"] = True
        gt["tampering_type"] = "FAKE_STAMP_FORGERY"
        gt["tampering_target"] = "VISA_STAMP"
        gt["tampering_bbox"] = {
            "x": stamp_x - stamp_radius, 
            "y": stamp_y - stamp_radius, 
            "width": stamp_radius * 2, 
            "height": stamp_radius * 2
        }
        gt["expected_verdict"] = "FORGERY_DETECTED"

        return tampered_doc, gt

    def generate_tampered_sample(self, scenario: str = "PHOTO_REPLACEMENT") -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
        """
        Generates a paired sample (tampered doc, ground truth, original face).
        """
        clean_doc, gt, original_face = self.generator.generate_passport()

        if scenario == "PHOTO_REPLACEMENT":
            tampered_doc, tampered_gt = self.inject_photo_replacement(clean_doc, gt)
        elif scenario == "TEXT_DATE":
            tampered_doc, tampered_gt = self.inject_date_alteration(clean_doc, gt)
        elif scenario == "FAKE_STAMP":
            tampered_doc, tampered_gt = self.inject_fake_stamp(clean_doc, gt)
        else:
            tampered_doc, tampered_gt = clean_doc, gt

        return tampered_doc, tampered_gt, original_face
