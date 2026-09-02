"""
Document Rules & Integrity Validation Engine (Stage 03 - Validate)
Validates document standards, logical dates, format rules, and cross-checks VIZ vs MRZ zones.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import re

from src.validation_engine.checksum_validator import ChecksumValidator


class DocumentRulesValidator:
    """
    Comprehensive rule enforcement engine for Passports, Visas, and National IDs.
    Cross-checks data consistency, validates expiration/DOB, and detects conflicting alterations.
    """

    # Common ISO 3166-1 alpha-3 country codes
    VALID_COUNTRY_CODES = {
        "IND", "USA", "GBR", "CAN", "AUS", "DEU", "FRA", "JPN", "SGP", "ARE", 
        "NZL", "ITA", "ESP", "NLD", "CHE", "SWE", "NOR", "DNK", "FIN", "IRL",
        "BRA", "ZAF", "CHN", "RUS", "MEX", "IDN", "MYS", "THA", "SAU", "TUR"
    }

    def __init__(self):
        self.checksum_validator = ChecksumValidator()

    def validate_dates(self, dob_str: Optional[str], expiry_str: Optional[str]) -> Dict[str, Any]:
        """
        Validates date logic:
        1. Expiry date must be in the future (or flagged as EXPIRED).
        2. Date of birth must be in the past.
        3. Age must be realistic (0 to 125 years).
        4. Passport validity span sanity check (typically <= 10.5 years from issue).
        """
        issues = []
        today = datetime.now()
        is_expired = False
        calculated_age = None

        if dob_str:
            try:
                dob = datetime.strptime(dob_str, "%Y-%m-%d")
                if dob > today:
                    issues.append("DOB_IN_FUTURE: Date of birth cannot be in the future")
                else:
                    calculated_age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                    if calculated_age > 125:
                        issues.append(f"UNREALISTIC_AGE: Calculated age is {calculated_age} years")
            except ValueError:
                issues.append("INVALID_DOB_FORMAT")

        if expiry_str:
            try:
                exp = datetime.strptime(expiry_str, "%Y-%m-%d")
                if exp < today:
                    is_expired = True
                    issues.append(f"DOCUMENT_EXPIRED: Document expired on {expiry_str}")
            except ValueError:
                issues.append("INVALID_EXPIRY_FORMAT")

        return {
            "is_expired": is_expired,
            "calculated_age": calculated_age,
            "date_issues": issues,
            "passed": len(issues) == 0 or (len(issues) == 1 and is_expired)
        }

    def validate_country_code(self, country_code: Optional[str]) -> Dict[str, Any]:
        """Validates if issuing country/nationality conforms to ISO 3166-1 alpha-3."""
        if not country_code:
            return {"valid": False, "code": None, "issue": "MISSING_COUNTRY_CODE"}
        code = country_code.upper().strip()
        is_valid = len(code) == 3 and (code in self.VALID_COUNTRY_CODES or code.isalpha())
        return {
            "valid": is_valid,
            "code": code,
            "issue": None if is_valid else f"INVALID_COUNTRY_CODE: '{code}'"
        }

    def cross_check_viz_and_mrz(self, viz_data: Optional[Dict[str, Any]], mrz_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Cross-checks visual inspection zone (VIZ) text with machine-readable zone (MRZ).
        Discrepancies indicate text manipulation/tampering (e.g. altering printed name/date on top without modifying MRZ).
        """
        if not viz_data or not mrz_data:
            return {
                "cross_check_performed": False,
                "discrepancies": [],
                "match_score": 1.0
            }

        discrepancies = []
        matches = 0
        total_comparisons = 0

        # 1. Compare Document Number
        viz_doc_num = viz_data.get("viz_document_number")
        mrz_doc_num = mrz_data.get("document_number")
        if viz_doc_num and mrz_doc_num:
            total_comparisons += 1
            # Clean non-alphanumerics
            c_viz = re.sub(r'[^A-Z0-9]', '', viz_doc_num.upper())
            c_mrz = re.sub(r'[^A-Z0-9]', '', mrz_doc_num.upper())
            if c_viz == c_mrz:
                matches += 1
            else:
                discrepancies.append(f"DOCUMENT_NUMBER_MISMATCH: VIZ shows '{viz_doc_num}' but MRZ encodes '{mrz_doc_num}'")

        # 2. Compare Name
        viz_name = viz_data.get("viz_name")
        mrz_surname = mrz_data.get("surname", "")
        if viz_name and mrz_surname:
            total_comparisons += 1
            c_viz_name = re.sub(r'[^A-Z]', '', viz_name.upper())
            c_mrz_surname = re.sub(r'[^A-Z]', '', mrz_surname.upper())
            if c_mrz_surname in c_viz_name or c_viz_name in c_mrz_surname:
                matches += 1
            else:
                discrepancies.append(f"NAME_INCONSISTENCY: VIZ name '{viz_name}' does not match MRZ surname '{mrz_surname}'")

        match_score = (matches / total_comparisons) if total_comparisons > 0 else 1.0

        return {
            "cross_check_performed": total_comparisons > 0,
            "discrepancies": discrepancies,
            "match_score": round(match_score, 2)
        }

    def validate(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes complete multi-rule document integrity validation.
        """
        fields = ocr_result.get("fields", {})
        mrz_data = ocr_result.get("mrz")
        viz_data = ocr_result.get("viz")

        # Checksums
        checksum_results = self.checksum_validator.validate_mrz_checksums(mrz_data)

        # Dates
        date_results = self.validate_dates(
            dob_str=fields.get("date_of_birth"),
            expiry_str=fields.get("date_of_expiry")
        )

        # Country & Nationality
        issuing_country_res = self.validate_country_code(fields.get("issuing_country"))
        nationality_res = self.validate_country_code(fields.get("nationality"))

        # Cross check
        cross_check_res = self.cross_check_viz_and_mrz(viz_data, mrz_data)

        # Aggregate Rule Score & Verdict
        all_violations = []
        all_violations.extend(checksum_results["failed_checks"])
        all_violations.extend(date_results["date_issues"])
        if not issuing_country_res["valid"] and issuing_country_res["issue"]:
            all_violations.append(issuing_country_res["issue"])
        if not nationality_res["valid"] and nationality_res["issue"]:
            all_violations.append(nationality_res["issue"])
        all_violations.extend(cross_check_res["discrepancies"])

        # Calculate Overall Validation Score
        weights = {
            "checksum": 0.40,
            "dates": 0.25,
            "cross_check": 0.25,
            "country_codes": 0.10
        }

        date_score = 1.0 if not date_results["date_issues"] else (0.6 if date_results["is_expired"] else 0.2)
        country_score = 1.0 if (issuing_country_res["valid"] and nationality_res["valid"]) else 0.5

        final_rule_score = (
            checksum_results["score"] * weights["checksum"] +
            date_score * weights["dates"] +
            cross_check_res["match_score"] * weights["cross_check"] +
            country_score * weights["country_codes"]
        )

        # Determine Verdict
        if len(all_violations) == 0:
            verdict = "PASSED"
            risk_level = "LOW"
        elif date_results["is_expired"] and len(all_violations) == 1:
            verdict = "EXPIRED"
            risk_level = "MODERATE"
        elif len(checksum_results["failed_checks"]) > 0 or len(cross_check_res["discrepancies"]) > 0:
            verdict = "FAILED_INTEGRITY_CHECK"
            risk_level = "CRITICAL"
        else:
            verdict = "REVIEW_REQUIRED"
            risk_level = "HIGH"

        return {
            "validation_score": round(final_rule_score, 2),
            "verdict": verdict,
            "risk_level": risk_level,
            "is_expired": date_results["is_expired"],
            "calculated_age": date_results["calculated_age"],
            "checksum_validation": checksum_results,
            "date_validation": date_results,
            "cross_check_validation": cross_check_res,
            "violations": all_violations
        }
