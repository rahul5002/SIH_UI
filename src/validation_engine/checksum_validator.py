"""
ICAO Doc 9303 Check Digit & Checksum Validator
"""

from typing import Dict, Any, List
from src.ocr_engine.mrz_parser import MRZParser


class ChecksumValidator:
    """
    Validates machine-readable identity document checksums according to ICAO Doc 9303 standards.
    """

    @staticmethod
    def validate_mrz_checksums(mrz_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates individual and composite check digits for a parsed MRZ dictionary.
        Returns detailed status of each check digit with computed vs extracted values.
        """
        if not mrz_data or "checksums" not in mrz_data:
            return {
                "valid": False,
                "score": 0.0,
                "details": {"error": "No MRZ checksum data available"},
                "failed_checks": ["MRZ_NOT_FOUND"]
            }

        checks = mrz_data["checksums"]
        failed = []

        if not checks.get("document_number_valid", False):
            failed.append("DOCUMENT_NUMBER_CHECKSUM_MISMATCH")
        if not checks.get("date_of_birth_valid", False):
            failed.append("DATE_OF_BIRTH_CHECKSUM_MISMATCH")
        if not checks.get("date_of_expiry_valid", False):
            failed.append("EXPIRY_DATE_CHECKSUM_MISMATCH")
        if not checks.get("composite_valid", False):
            failed.append("COMPOSITE_CHECKSUM_MISMATCH")

        passed_count = 4 - len(failed)
        checksum_score = passed_count / 4.0

        return {
            "valid": len(failed) == 0,
            "score": round(checksum_score, 2),
            "passed_checks": passed_count,
            "total_checks": 4,
            "failed_checks": failed,
            "details": checks
        }
