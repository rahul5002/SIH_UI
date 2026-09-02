/* =========================================================================
   PRAMAAN AI — Configuration
   Single source of truth for tuning parameters and thresholds.
   Backend must keep these aligned with API logic.
   ========================================================================= */

const CONFIG = {
  THRESHOLDS: {
    // Composite risk score (0-100) at or above which a case is auto-flagged for review
    AUTO_FLAG_RISK: 45,
    
    // Minimum ArcFace cosine similarity score (0.0 - 1.0) required to automatically verify a live face
    FACE_MATCH_MIN: 0.85
  }
};

window.CONFIG = CONFIG;
