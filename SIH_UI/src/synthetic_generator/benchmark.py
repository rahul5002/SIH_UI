"""
Synthetic Dataset Benchmark & Performance Evaluator
Runs multi-sample evaluation measuring OCR accuracy, Rule verification, Tampering Precision/Recall, and Face matching.
"""

from typing import Dict, Any, List
import numpy as np

from src.synthetic_generator.generator import SyntheticDocumentGenerator
from src.synthetic_generator.tampering_injector import TamperingInjector
from src.preprocessing.preprocessor import DocumentPreprocessor
from src.ocr_engine.ocr_extractor import OCRExtractor
from src.validation_engine.rules_validator import DocumentRulesValidator
from src.tampering_engine.tampering_detector import TamperingDetector
from src.face_engine.face_matcher import FaceVerificationEngine


class ScreeningBenchmark:
    """
    Automated evaluation framework for the PRAMAAN AI pipeline.
    """

    def __init__(self):
        self.generator = SyntheticDocumentGenerator()
        self.injector = TamperingInjector()
        self.preprocessor = DocumentPreprocessor()
        self.ocr_extractor = OCRExtractor()
        self.rules_validator = DocumentRulesValidator()
        self.tampering_detector = TamperingDetector()
        self.face_engine = FaceVerificationEngine()

    def run_benchmark(self, num_samples: int = 10) -> Dict[str, Any]:
        """
        Generates genuine and tampered test document batches and evaluates accuracy metrics.
        """
        results = []
        
        scenarios = ["CLEAN", "PHOTO_REPLACEMENT", "TEXT_DATE", "FAKE_STAMP"]
        
        tp = 0  # Tampered & Detected
        fp = 0  # Clean but flagged as Tampered
        tn = 0  # Clean & passed
        fn = 0  # Tampered but missed

        ocr_scores = []
        validation_scores = []

        for i in range(num_samples):
            scenario = scenarios[i % len(scenarios)]
            clean_doc, gt, original_face = self.generator.generate_passport()

            if scenario == "CLEAN":
                test_doc, test_gt = clean_doc, gt
            elif scenario == "PHOTO_REPLACEMENT":
                test_doc, test_gt = self.injector.inject_photo_replacement(clean_doc, gt)
            elif scenario == "TEXT_DATE":
                test_doc, test_gt = self.injector.inject_date_alteration(clean_doc, gt)
            else:
                test_doc, test_gt = self.injector.inject_fake_stamp(clean_doc, gt)

            # Run Screening Pipeline
            preprocessed = self.preprocessor.process(test_doc)
            ocr_res = self.ocr_extractor.process(preprocessed["processed_image"])
            val_res = self.rules_validator.validate(ocr_res)
            tamper_res = self.tampering_detector.detect(
                image=preprocessed["processed_image"],
                ocr_tokens=ocr_res.get("ocr_tokens", []),
                photo_bbox=test_gt.get("photo_bbox")
            )

            is_ground_truth_tampered = test_gt["is_tampered"]
            is_predicted_tampered = tamper_res["tampering_detected"] or (val_res["verdict"] == "FAILED_INTEGRITY_CHECK")

            if is_ground_truth_tampered and is_predicted_tampered:
                tp += 1
            elif not is_ground_truth_tampered and is_predicted_tampered:
                fp += 1
            elif not is_ground_truth_tampered and not is_predicted_tampered:
                tn += 1
            else:
                fn += 1

            ocr_scores.append(ocr_res.get("overall_confidence", 0.0))
            validation_scores.append(val_res.get("validation_score", 0.0))

            results.append({
                "sample_id": i + 1,
                "scenario": scenario,
                "ground_truth_tampered": is_ground_truth_tampered,
                "predicted_tampered": is_predicted_tampered,
                "tampering_score": tamper_res["tampering_score"],
                "validation_verdict": val_res["verdict"],
                "ocr_confidence": ocr_res.get("overall_confidence", 0.0)
            })

        # Calculate metrics
        total = len(results)
        accuracy = (tp + tn) / float(total) if total > 0 else 0.0
        precision = tp / float(tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / float(tp + fn) if (tp + fn) > 0 else 1.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "total_samples": total,
            "metrics": {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1_score, 4),
                "mean_ocr_confidence": round(float(np.mean(ocr_scores)), 4) if ocr_scores else 0.0,
                "mean_validation_score": round(float(np.mean(validation_scores)), 4) if validation_scores else 0.0
            },
            "confusion_matrix": {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn
            },
            "detailed_samples": results
        }
