"""
Startup Script for PRAMAAN AI Server (Unified)
SIH 2026

Runs the FastAPI application defined in backend_api.py on port 8080.
"""

import uvicorn
import os
import sys

if __name__ == "__main__":
    # Ensure the SIH_UI folder is in the Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 60)
    print("Starting PRAMAAN AI Unified Server (Frontend + ML API)")
    print("=" * 60)
    print("Listening on: http://127.0.0.1:8080")
    print("Press Ctrl+C to stop.")
    
    uvicorn.run(
        "backend_api:app",
        host="0.0.0.0",
        port=8080,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
