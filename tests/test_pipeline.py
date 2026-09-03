"""
Unit and Integration Test Suite for PRAMAAN AI
Verifies all 5 stages: Preprocessing, OCR & MRZ, Rules Validation, Tampering Forensics, Face Verification.
"""

import sys
import os
import unittest
import numpy as np
import cv2

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.preprocessor import DocumentPreprocessor
from src.ocr_engine.mrz_parser import MRZParser
from src.validation_engine.checksum_validator import ChecksumValidator
from src.validation_engine.rules_validator import DocumentRulesValidator
from src.tampering_engine.tampering_detector import TamperingDetector
from src.face_engine.face_matcher import FaceVerificationEngine
from src.synthetic_generator.generator import SyntheticDocumentGenerator
from src.synthetic_generator.tampering_injector import TamperingInjector


class TestPramaanAI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.preprocessor = DocumentPreprocessor()
        cls.generator = SyntheticDocumentGenerator()
        cls.injector = TamperingInjector()
        cls.rules_validator = DocumentRulesValidator()
        cls.tampering_detector = TamperingDetector()
        cls.face_engine = FaceVerificationEngine()

    def test_01_mrz_parser_and_checksums(self):
        """Tests ICAO Doc 9303 TD3 standard parsing and 7-3-1 check digit validation."""
        line1 = "P<INDSHARMA<<RAHUL<<<<<<<<<<<<<<<<<<<<<<<<<<"
        line2 = "Z1234567<8IND9205141M2911204<<<<<<<<<<<<<<0"
        
        parsed = MRZParser.parse_td3_passport(line1, line2)
        self.assertEqual(parsed["document_type"], "PASSPORT")
        self.assertEqual(parsed["issuing_country"], "IND")
        self.assertEqual(parsed["surname"], "SHARMA")
        self.assertEqual(parsed["given_names"], "RAHUL")
        self.assertEqual(parsed["document_number"], "Z1234567")
        self.assertEqual(parsed["gender"], "M")
        
        # Verify 7-3-1 check digits
        self.assertEqual(MRZParser.compute_check_digit("Z1234567"), 1)
        self.assertEqual(MRZParser.compute_check_digit("920514"), 1)
        self.assertEqual(MRZParser.compute_check_digit("291120"), 5)

    def test_02_synthetic_generator(self):
        """Tests pristine genuine document generation with valid MRZ."""
        doc_img, gt, avatar = self.generator.generate_passport(
            surname="SINGH",
            given_names="PRIYA",
            doc_number="A9876543",
            country_code="IND",
            gender="F",
            dob="1995-08-15",
            expiry="2030-08-14"
        )
        self.assertIsNotNone(doc_img)
        self.assertEqual(doc_img.shape[2], 3)
        self.assertEqual(gt["document_number"], "A9876543")
        self.assertTrue(gt["checksums_valid"])

    def test_03_tampering_injection_and_detection(self):
        """Tests photo replacement tampering injection and AI forensic detection."""
        clean_doc, gt, avatar = self.generator.generate_passport()
        tampered_doc, tampered_gt = self.injector.inject_photo_replacement(clean_doc, gt)

        # Run ELA and Tampering Detection
        tamper_res = self.tampering_detector.detect(
            image=tampered_doc,
            photo_bbox=tampered_gt["photo_bbox"]
        )

        self.assertIn("tampering_score", tamper_res)
        self.assertIn("heatmap_base64", tamper_res)
        self.assertTrue(len(tamper_res["evidence_regions"]) > 0)
        self.assertGreaterEqual(tamper_res["tampering_score"], 35.0)

    def test_04_face_verification(self):
        """Tests facial detection, feature embedding, and cosine similarity."""
        avatar1 = self.generator._create_avatar_photo(gender="M")
        avatar2 = avatar1.copy()

        # Compare identical avatar
        res_same = self.face_engine.verify_faces(avatar1, avatar2)
        self.assertTrue(res_same["success"])
        self.assertGreater(res_same["similarity_score"], 0.85)

        # Compare different avatar (opposite gender)
        avatar_diff = self.generator._create_avatar_photo(gender="F")
        res_diff = self.face_engine.verify_faces(avatar1, avatar_diff)
        self.assertTrue(res_diff["success"])
        self.assertLess(res_diff["similarity_score"], res_same["similarity_score"])
        self.assertFalse(res_diff["is_match"])


    def test_05_rules_validation_expired(self):
        """Tests date logic validator on expired documents."""
        ocr_fake_result = {
            "fields": {
                "date_of_birth": "1990-01-01",
                "date_of_expiry": "2020-01-01",  # Past date
                "issuing_country": "IND",
                "nationality": "IND"
            },
            "mrz": None,
            "viz": None
        }
        res = self.rules_validator.validate(ocr_fake_result)
        self.assertTrue(res["is_expired"])


if __name__ == "__main__":
    unittest.main()
