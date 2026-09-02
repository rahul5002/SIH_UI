"""
ICAO Doc 9303 MRZ (Machine Readable Zone) Parser
Supports TD1 (3x30 - ID Cards), TD2 (2x36 - Visas), and TD3 (2x44 - Passports).
"""

import re
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime


class MRZParser:
    """
    Parser for ICAO Doc 9303 Machine Readable Travel Documents (MRTD).
    Decodes TD1, TD2, and TD3 specifications and computes 7-3-1 check digits.
    """

    WEIGHTS = [7, 3, 1]

    @staticmethod
    def char_value(c: str) -> int:
        """Converts MRZ character to numeric value according to ICAO 9303."""
        if c == '<':
            return 0
        if c.isdigit():
            return int(c)
        if c.isalpha():
            return ord(c.upper()) - ord('A') + 10
        return 0

    @classmethod
    def compute_check_digit(cls, data: str) -> int:
        """Calculates ICAO 7-3-1 weighted modulo-10 check digit."""
        total = 0
        for i, char in enumerate(data):
            w = cls.WEIGHTS[i % 3]
            total += cls.char_value(char) * w
        return total % 10

    @classmethod
    def verify_check_digit(cls, data: str, expected_digit: str) -> bool:
        """Verifies if computed check digit matches the expected check digit in MRZ."""
        if not expected_digit.isdigit():
            return False
        return cls.compute_check_digit(data) == int(expected_digit)

    @staticmethod
    def parse_mrz_date(date_str: str, is_expiry: bool = False) -> Tuple[Optional[str], Optional[datetime]]:
        """
        Parses YYMMDD date format into YYYY-MM-DD.
        Infers century based on whether it is DOB or Expiry.
        """
        if len(date_str) != 6 or not date_str.isdigit():
            return None, None
        
        yy = int(date_str[:2])
        mm = int(date_str[2:4])
        dd = int(date_str[4:6])

        if not (1 <= mm <= 12 and 1 <= dd <= 31):
            return None, None

        current_year_last2 = datetime.now().year % 100
        if is_expiry:
            # Expiry dates in 2000s unless year is way before
            century = 2000 if yy <= (current_year_last2 + 40) else 1900
        else:
            # DOB: if yy > current year -> 1900s, else 2000s
            century = 2000 if yy <= current_year_last2 else 1900

        full_year = century + yy
        try:
            dt = datetime(full_year, mm, dd)
            return dt.strftime("%Y-%m-%d"), dt
        except ValueError:
            return None, None

    @classmethod
    def clean_mrz_lines(cls, lines: List[str]) -> List[str]:
        """Filters, cleans, and standardizes OCR MRZ candidate lines."""
        cleaned = []
        for line in lines:
            # Remove non-MRZ characters except uppercase letters, digits, and filler '<'
            sanitized = re.sub(r'[^A-Z0-9<]', '', line.upper().replace(' ', ''))
            if len(sanitized) >= 28:
                cleaned.append(sanitized)
        return cleaned

    @classmethod
    def parse_td3_passport(cls, line1: str, line2: str) -> Dict[str, Any]:
        """
        Parses TD3 format (Passport standard: 2 lines of 44 characters).
        Line 1: P<ISSNAME<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        Line 2: DOCNUM<CD<NATDOB<CD<SEXP<CD<OPTIONAL<<<<<<COMPCD
        """
        line1 = (line1 + '<' * 44)[:44]
        line2 = (line2 + '<' * 44)[:44]

        doc_code = line1[0:2].replace('<', '')
        issuing_country = line1[2:5].replace('<', '')
        
        # Name parsing: SURNAME<<GIVEN_NAMES
        name_field = line1[5:44]
        name_parts = name_field.split('<<')
        surname = name_parts[0].replace('<', ' ').strip()
        given_names = " ".join([p.replace('<', ' ').strip() for p in name_parts[1:] if p]).strip()
        full_name = f"{given_names} {surname}".strip() if given_names else surname

        doc_number_raw = line2[0:9]
        doc_number = doc_number_raw.replace('<', '').strip()
        doc_num_check = line2[9]
        doc_num_valid = cls.verify_check_digit(doc_number_raw, doc_num_check)

        nationality = line2[10:13].replace('<', '')
        
        dob_raw = line2[13:19]
        dob_check = line2[19]
        dob_formatted, dob_dt = cls.parse_mrz_date(dob_raw, is_expiry=False)
        dob_valid = cls.verify_check_digit(dob_raw, dob_check)

        gender_char = line2[20]
        gender = "M" if gender_char == "M" else ("F" if gender_char == "F" else "X")

        expiry_raw = line2[21:27]
        expiry_check = line2[27]
        expiry_formatted, exp_dt = cls.parse_mrz_date(expiry_raw, is_expiry=True)
        expiry_valid = cls.verify_check_digit(expiry_raw, expiry_check)

        optional_data = line2[28:42].replace('<', '').strip()
        composite_data = line2[0:10] + line2[13:20] + line2[21:43]
        composite_check = line2[43]
        composite_valid = cls.verify_check_digit(composite_data, composite_check)

        all_checks_passed = all([doc_num_valid, dob_valid, expiry_valid, composite_valid])

        return {
            "document_type": "PASSPORT",
            "format": "TD3",
            "document_code": doc_code,
            "issuing_country": issuing_country,
            "surname": surname,
            "given_names": given_names,
            "full_name": full_name,
            "document_number": doc_number,
            "nationality": nationality,
            "date_of_birth": dob_formatted,
            "date_of_birth_raw": dob_raw,
            "gender": gender,
            "date_of_expiry": expiry_formatted,
            "date_of_expiry_raw": expiry_raw,
            "optional_data": optional_data,
            "checksums": {
                "document_number_valid": doc_num_valid,
                "date_of_birth_valid": dob_valid,
                "date_of_expiry_valid": expiry_valid,
                "composite_valid": composite_valid,
                "all_valid": all_checks_passed
            },
            "raw_mrz": f"{line1}\n{line2}"
        }

    @classmethod
    def parse_td2_visa(cls, line1: str, line2: str) -> Dict[str, Any]:
        """
        Parses TD2 format (Visa standard: 2 lines of 36 characters).
        """
        line1 = (line1 + '<' * 36)[:36]
        line2 = (line2 + '<' * 36)[:36]

        doc_code = line1[0:2].replace('<', '')
        issuing_country = line1[2:5].replace('<', '')
        
        name_field = line1[5:36]
        name_parts = name_field.split('<<')
        surname = name_parts[0].replace('<', ' ').strip()
        given_names = " ".join([p.replace('<', ' ').strip() for p in name_parts[1:] if p]).strip()
        full_name = f"{given_names} {surname}".strip() if given_names else surname

        doc_number_raw = line2[0:9]
        doc_number = doc_number_raw.replace('<', '').strip()
        doc_num_check = line2[9]
        doc_num_valid = cls.verify_check_digit(doc_number_raw, doc_num_check)

        nationality = line2[10:13].replace('<', '')
        
        dob_raw = line2[13:19]
        dob_check = line2[19]
        dob_formatted, _ = cls.parse_mrz_date(dob_raw, is_expiry=False)
        dob_valid = cls.verify_check_digit(dob_raw, dob_check)

        gender_char = line2[20]
        gender = "M" if gender_char == "M" else ("F" if gender_char == "F" else "X")

        expiry_raw = line2[21:27]
        expiry_check = line2[27]
        expiry_formatted, _ = cls.parse_mrz_date(expiry_raw, is_expiry=True)
        expiry_valid = cls.verify_check_digit(expiry_raw, expiry_check)

        optional_data = line2[28:35].replace('<', '').strip()
        composite_data = line2[0:10] + line2[13:20] + line2[21:35]
        composite_check = line2[35]
        composite_valid = cls.verify_check_digit(composite_data, composite_check)

        return {
            "document_type": "VISA",
            "format": "TD2",
            "document_code": doc_code,
            "issuing_country": issuing_country,
            "surname": surname,
            "given_names": given_names,
            "full_name": full_name,
            "document_number": doc_number,
            "nationality": nationality,
            "date_of_birth": dob_formatted,
            "date_of_birth_raw": dob_raw,
            "gender": gender,
            "date_of_expiry": expiry_formatted,
            "date_of_expiry_raw": expiry_raw,
            "optional_data": optional_data,
            "checksums": {
                "document_number_valid": doc_num_valid,
                "date_of_birth_valid": dob_valid,
                "date_of_expiry_valid": expiry_valid,
                "composite_valid": composite_valid,
                "all_valid": all([doc_num_valid, dob_valid, expiry_valid, composite_valid])
            },
            "raw_mrz": f"{line1}\n{line2}"
        }

    @classmethod
    def parse_td1_id_card(cls, line1: str, line2: str, line3: str) -> Dict[str, Any]:
        """
        Parses TD1 format (National ID card standard: 3 lines of 30 characters).
        """
        line1 = (line1 + '<' * 30)[:30]
        line2 = (line2 + '<' * 30)[:30]
        line3 = (line3 + '<' * 30)[:30]

        doc_code = line1[0:2].replace('<', '')
        issuing_country = line1[2:5].replace('<', '')
        doc_number_raw = line1[5:14]
        doc_num_check = line1[14]
        doc_number = doc_number_raw.replace('<', '').strip()
        doc_num_valid = cls.verify_check_digit(doc_number_raw, doc_num_check)

        dob_raw = line2[0:6]
        dob_check = line2[6]
        dob_formatted, _ = cls.parse_mrz_date(dob_raw, is_expiry=False)
        dob_valid = cls.verify_check_digit(dob_raw, dob_check)

        gender_char = line2[7]
        gender = "M" if gender_char == "M" else ("F" if gender_char == "F" else "X")

        expiry_raw = line2[8:14]
        expiry_check = line2[14]
        expiry_formatted, _ = cls.parse_mrz_date(expiry_raw, is_expiry=True)
        expiry_valid = cls.verify_check_digit(expiry_raw, expiry_check)

        nationality = line2[15:18].replace('<', '')
        
        name_parts = line3.split('<<')
        surname = name_parts[0].replace('<', ' ').strip()
        given_names = " ".join([p.replace('<', ' ').strip() for p in name_parts[1:] if p]).strip()
        full_name = f"{given_names} {surname}".strip() if given_names else surname

        composite_data = line1[5:30] + line2[0:7] + line2[8:15] + line2[18:29]
        composite_check = line2[29]
        composite_valid = cls.verify_check_digit(composite_data, composite_check)

        return {
            "document_type": "NATIONAL_ID",
            "format": "TD1",
            "document_code": doc_code,
            "issuing_country": issuing_country,
            "surname": surname,
            "given_names": given_names,
            "full_name": full_name,
            "document_number": doc_number,
            "nationality": nationality,
            "date_of_birth": dob_formatted,
            "date_of_birth_raw": dob_raw,
            "gender": gender,
            "date_of_expiry": expiry_formatted,
            "date_of_expiry_raw": expiry_raw,
            "optional_data": "",
            "checksums": {
                "document_number_valid": doc_num_valid,
                "date_of_birth_valid": dob_valid,
                "date_of_expiry_valid": expiry_valid,
                "composite_valid": composite_valid,
                "all_valid": all([doc_num_valid, dob_valid, expiry_valid, composite_valid])
            },
            "raw_mrz": f"{line1}\n{line2}\n{line3}"
        }

    @classmethod
    def parse_auto(cls, candidate_lines: List[str]) -> Optional[Dict[str, Any]]:
        """
        Auto-detects MRZ format (TD3, TD2, TD1) from a list of OCR candidate lines.
        """
        lines = cls.clean_mrz_lines(candidate_lines)
        if not lines:
            return None

        # Check for 2 lines of ~44 chars (TD3 Passport)
        td3_candidates = [l for l in lines if len(l) >= 40]
        if len(td3_candidates) >= 2:
            return cls.parse_td3_passport(td3_candidates[-2], td3_candidates[-1])

        # Check for 3 lines of ~30 chars (TD1 ID)
        td1_candidates = [l for l in lines if 28 <= len(l) <= 34]
        if len(td1_candidates) >= 3:
            return cls.parse_td1_id_card(td1_candidates[-3], td1_candidates[-2], td1_candidates[-1])

        # Check for 2 lines of ~36 chars (TD2 Visa)
        td2_candidates = [l for l in lines if 34 <= len(l) <= 38]
        if len(td2_candidates) >= 2:
            return cls.parse_td2_visa(td2_candidates[-2], td2_candidates[-1])

        # Fallback: if at least 2 lines exist
        if len(lines) >= 2:
            if len(lines[-1]) >= 40 or len(lines[-2]) >= 40:
                return cls.parse_td3_passport(lines[-2], lines[-1])
            else:
                return cls.parse_td2_visa(lines[-2], lines[-1])

        return None
