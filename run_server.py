"""
PRAMAAN AI — Server Startup Script
Launches the FastAPI AI inference engine and Web Dashboard.
"""

import os
import sys

# Configure UTF-8 encoding for Windows terminal stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import uvicorn

if __name__ == "__main__":
    print("==================================================================")
    print("  [+] PRAMAAN AI: Fake Identity & Document Screening System")
    print("  SIH 2026 Problem Statement ID: 26188")
    print("==================================================================")
    print("  [*] Starting AI Server & Web Dashboard on http://127.0.0.1:8000")
    print("  [*] Interactive Swagger API Docs at http://127.0.0.1:8000/docs")
    print("==================================================================")
    
    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=False)
