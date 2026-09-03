# PRAMAAN AI — SIH 2026 Prototype
**Problem Statement 26188: AI-Based Fake Identity & Document Screening System**

This repository contains the complete frontend shell and backend integration layer for the PRAMAAN AI prototype.

## Architecture
- **Frontend**: Vanilla HTML/CSS/JS (no React/Vite/build tools).
- **Backend**: FastAPI (serves both API endpoints and static frontend files).
- **ML Integration**: Local Python stubs ready for drop-in model implementations.
- **Storage**: SQLite local database for the prototype audit log.

## Folder Structure
```text
/
├── index.html, login.html, etc.  # Frontend Pages
├── css/, js/, assets/            # Frontend Assets (100% local, no CDNs)
├── ml_stubs/                     # ML Team integration files
├── backend_api.py                # FastAPI Application & API Routes
├── requirements.txt              # Python Dependencies
├── audit.db                      # Local SQLite Audit Ledger (Generated)
└── README.md                     # Project Documentation
```

## Installation & Run Instructions

### 1. Setup Environment
Ensure you have Python 3.8+ installed.
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Run the Application
Start the FastAPI server. It serves both the frontend and the API from the same port.
```bash
python -m uvicorn backend_api:app --host 127.0.0.1 --port 8080
```

### 3. Access the Application
- **Frontend App:** http://localhost:8080
- **Swagger API Docs:** http://localhost:8080/docs

---

## AI Teammate Integration Guide

The ML pipelines are currently mocked in the `ml_stubs/` directory. Each AI teammate should modify their respective file to integrate their actual model logic without needing to rewrite any frontend or backend API code.

- **OCR / Text Extraction:** Modify `ml_stubs/ocr.py`
- **Tamper Detection (CNN/ELA):** Modify `ml_stubs/tamper.py`
- **Face Verification (ArcFace):** Modify `ml_stubs/face.py`
- **Liveness Detection:** Modify `ml_stubs/liveness.py`
- **Biometric Deduplication (FAISS):** Modify `ml_stubs/dedup.py`

### Score Semantics
When integrating models, adhere to the following scale logic to ensure the backend orchestrator correctly calculates the composite risk:
- **Tampering Score:** `0.0` = No tampering risk ... `1.0` = High risk / confirmed forged.
- **Face Score:** `0.0` = No match ... `1.0` = Strong match.
- **Liveness Score:** `0.0` = Spoof detected ... `1.0` = Confirmed live.
- **Dedup Score:** `0.0` = Unique ... `1.0` = Confirmed duplicate profile in database.

Current global thresholds:
- `AUTO_FLAG_RISK = 45.0`
- `FACE_MATCH_MINIMUM = 0.85`

### API Response Contract
All individual screening stages must return a common JSON format to ensure frontend stability:
```json
{
  "status": "PASS|REVIEW|FLAG|REJECT|UNAVAILABLE",
  "score": 0.95,
  "evidence": ["Detail 1", "Detail 2"],
  "reason": "String explaining the outcome",
  "data": { "optional": "fields like OCR text" }
}
```

The orchestrator (`/api/screening/analyze`) structures this uniformly using the keys: `ocr`, `validation`, `tampering`, `face`, `liveness`, `dedup`, `overall`, and `audit`.

### Endpoints Table
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | System health check |
| POST | `/api/auth/login` | Officer authentication |
| POST | `/api/screening/extract` | Document OCR extraction |
| POST | `/api/screening/validate` | MRZ and expiration validation |
| POST | `/api/screening/detect` | Visual tampering detection |
| POST | `/api/screening/verify` | Face match, liveness, deduplication |
| POST | `/api/screening/log` | Anchor event to ledger |
| POST | `/api/screening/decision` | Manual officer decision |
| POST | `/api/screening/analyze` | Unified orchestrator (all stages) |
| GET | `/api/audit` | Fetch audit logs |

---

## Important Notes

- **Prototype Authentication Note:** Authentication is mocked for the prototype demonstration. Do not use real passwords or secrets. Role assignment (`officer`, `supervisor`, `admin`) is handled entirely by the backend response based on the mock credentials used.
- **Offline/Local Deployment Note:** The frontend is 100% self-contained. All fonts (Inter, JetBrains Mono) and libraries (Chart.js) are bundled in the `assets/` directory. No CDNs or external APIs are used. This application is capable of running fully offline on an isolated border outpost network.
