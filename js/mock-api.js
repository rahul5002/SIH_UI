/* =========================================================================
   PRAMAAN AI — API Adapter Layer  (js/mock-api.js)
   -------------------------------------------------------------------------
   All public function signatures are preserved so screening.html, etc.
   need no call-site changes. All URLs are relative so this works on any
   machine where FastAPI serves both frontend and API from the same port.
   ========================================================================= */

const API = (() => {

  // ── Internal helper ────────────────────────────────────────────────────
  async function callApi(endpoint, payload) {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || ('API error ' + res.status));
    }
    return res.json();
  }

  // ── Screening pipeline ─────────────────────────────────────────────────

  async function stageExtract(payload) {
    return callApi('/api/screening/extract', {
      documentImage: payload.documentImage || 'mock_base64_doc'
    });
  }

  async function stageValidate(payload) {
    return callApi('/api/screening/validate', {
      mrzData: payload.mrzData || {}
    });
  }

  async function stageDetect(payload) {
    return callApi('/api/screening/detect', {
      documentImage: payload.documentImage || 'mock_base64_doc'
    });
  }

  async function stageVerify(payload) {
    return callApi('/api/screening/verify', {
      faceImage: payload.faceImage || 'mock_base64_face',
      documentFaceRegion: payload.documentFaceRegion || {}
    });
  }

  async function stageLog(payload) {
    return callApi('/api/screening/log', {
      caseRef:        payload.caseRef        || 'UNKNOWN',
      extractionData: payload.extractionData || {},
      tamperRes:      payload.tamperRes      || {},
      verifyRes:      payload.verifyRes      || {}
    });
  }

  async function anchorDecision(payload) {
    return callApi('/api/screening/decision', {
      caseRef:    payload.caseRef   || 'UNKNOWN',
      decision:   payload.decision  || 'PASS',
      finalScore: payload.finalScore != null ? payload.finalScore : 0.0
    });
  }

  // ── Authentication (Prototype only — mocked in backend) ────────────────
  async function authLogin(badgeId, pin) {
    const res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ badgeId, pin, checkpostId: 'Raxaul ICP' })
    });
    if (!res.ok) throw new Error('Invalid credentials');
    return res.json();
  }

  // ── Dashboard / supporting data (static until backend routes exist) ─────
  async function fetchDashboardStats(checkpostId) {
    return { screened: 0, flagged: 0, rejected: 0, failedOcr: 0, _mock: true };
  }
  async function fetchActivity(checkpostId) { return []; }
  async function fetchCheckpoints(region)   { return []; }
  async function fetchAlerts(checkpostId)   { return []; }

  async function fetchAuditLog(page = 1, risk = 'all', cp = 'all') {
    const res = await fetch('/api/audit');
    if (!res.ok) return [];
    const rows = await res.json();
    return rows
      .filter(r => risk === 'all' || r.status === risk.toUpperCase())
      .map(r => ({
        t:    r.timestamp,
        ref:  r.case_ref,
        doc:  r.reason || '\u2014',
        risk: r.score,
        dec:  (r.status || '').toLowerCase(),
        cp:   'Raxaul ICP'
      }));
  }

  // ── Public surface ─────────────────────────────────────────────────────
  return {
    fetchDashboardStats, fetchActivity, fetchCheckpoints, fetchAlerts, fetchAuditLog,
    stageExtract, stageValidate, stageDetect, stageVerify, stageLog, anchorDecision,
    authLogin
  };
})();

window.API = API;
