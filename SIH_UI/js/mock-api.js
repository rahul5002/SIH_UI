/* =========================================================================
   PRAMAAN AI — API Adapter Layer  (js/mock-api.js)
   -------------------------------------------------------------------------
   Maps the SIH_UI frontend call-sites to the real backend API.
   All public function signatures are preserved — screening.html, audit.html,
   etc. need ZERO changes.  Response shapes from the real ML backend are
   normalised here so every consumer gets what it expects.
   ========================================================================= */

const API = (() => {

  // ── Internal helper — JSON body request ────────────────────────────────
  async function callApi(endpoint, payload) {
    const res = await fetch(endpoint, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || ('API error ' + res.status));
    }
    return res.json();
  }

  // ── Stage 02 — Extract (OCR + MRZ) ─────────────────────────────────────
  async function stageExtract(payload) {
    const raw = await callApi('/api/screening/extract', {
      documentImage: payload.documentImage || ''
    });

    // Normalise: real backend returns {status, score, evidence, reason, data:{name,docNo,dob,expiry,...}}
    // screening.html reads extractRes?.data?.name etc. — already correct.
    // Also expose the full OCR result so stageValidate can use it.
    return raw;
  }

  // ── Stage 03 — Validate (rules, checksums, dates, VIZ/MRZ cross-check) ─
  async function stageValidate(payload) {
    // Pass the full OCR result as mrzData so the backend can run full validation
    const mrzData = payload.mrzData
      || (payload.extractionData && payload.extractionData.data)
      || {};

    return callApi('/api/screening/validate', { mrzData });
  }

  // ── Stage 04 — Detect (ELA + Noise + Splicing + Font + Metadata) ────────
  async function stageDetect(payload) {
    return callApi('/api/screening/detect', {
      documentImage: payload.documentImage || ''
    });
  }

  // ── Stage 05 — Verify (Face match + Liveness) ───────────────────────────
  async function stageVerify(payload) {
    return callApi('/api/screening/verify', {
      faceImage: payload.faceImage || '',
      // Pass document image inside documentFaceRegion.docImage so backend
      // can crop the doc face internally from the full page.
      documentFaceRegion: {
        docImage: payload.documentImage || '',
        ...(payload.documentFaceRegion || {})
      }
    });
  }

  // ── Stage 06 — Log / Risk Fusion ────────────────────────────────────────
  async function stageLog(payload) {
    return callApi('/api/screening/log', {
      caseRef:        payload.caseRef        || 'UNKNOWN',
      extractionData: payload.extractionData || {},
      tamperRes:      payload.tamperRes      || {},
      verifyRes:      payload.verifyRes      || {}
    });
  }

  // ── Officer Decision (anchors to audit ledger) ──────────────────────────
  async function anchorDecision(payload) {
    return callApi('/api/screening/decision', {
      caseRef:    payload.caseRef   || 'UNKNOWN',
      decision:   payload.decision  || 'PASS',
      finalScore: payload.finalScore != null ? payload.finalScore : 0.0
    });
  }

  // ── Authentication ──────────────────────────────────────────────────────
  async function authLogin(badgeId, pin) {
    const res = await fetch('/api/auth/login', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ badgeId, pin, checkpostId: 'Raxaul ICP' })
    });
    if (!res.ok) throw new Error('Invalid credentials');
    return res.json();
  }

  // ── Dashboard stats — pull from audit log ──────────────────────────────
  async function fetchDashboardStats(checkpostId) {
    try {
      const rows = await fetch('/api/audit').then(r => r.ok ? r.json() : []);
      const screened = rows.length;
      const flagged  = rows.filter(r => ['FLAG','FLAGGED_FORGERY_DETECTED','FLAGGED','LOGGED'].includes((r.status||'').toUpperCase())).length;
      const rejected = rows.filter(r => ['REJECT','REJECTED_EXPIRED_DOCUMENT'].includes((r.status||'').toUpperCase())).length;
      const failedOcr = rows.filter(r => (r.status||'').toUpperCase() === 'UNAVAILABLE').length;
      return { screened, flagged, rejected, failedOcr };
    } catch (_) {
      return { screened: 0, flagged: 0, rejected: 0, failedOcr: 0 };
    }
  }

  // ── Recent activity feed — from audit log ──────────────────────────────
  async function fetchActivity(checkpostId) {
    try {
      const rows = await fetch('/api/audit').then(r => r.ok ? r.json() : []);
      return rows.slice(0, 12).map(r => ({
        name:  r.case_ref || '—',
        doc:   r.reason   || '—',
        score: r.score != null ? r.score.toFixed(1) : '—',
        risk:  _mapStatusToRisk(r.status),
        time:  r.timestamp ? new Date(r.timestamp).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' }) : '—'
      }));
    } catch (_) {
      return [];
    }
  }

  function _mapStatusToRisk(status) {
    const s = (status || '').toUpperCase();
    if (['PASSED_CLEAN','PASS','CLEAR'].includes(s))             return 'clear';
    if (['REJECTED_EXPIRED_DOCUMENT','REJECT'].includes(s))      return 'reject';
    return 'flag';
  }

  async function fetchCheckpoints() { return []; }
  async function fetchAlerts()      { return []; }

  // ── Audit log page ──────────────────────────────────────────────────────
  async function fetchAuditLog(page = 1, risk = 'all', cp = 'all') {
    try {
      const res  = await fetch('/api/audit');
      if (!res.ok) return [];
      const rows = await res.json();
      return rows
        .filter(r => risk === 'all' || _mapStatusToRisk(r.status) === risk)
        .map(r => ({
          t:   r.timestamp,
          ref: r.case_ref,
          doc: r.reason || '—',
          risk: r.score != null ? r.score : null,
          dec: _mapStatusToRisk(r.status),
          cp:  'Raxaul ICP'
        }));
    } catch (_) {
      return [];
    }
  }

  // ── Synthetic Data Generation (1-Click AI Test Scenarios) ───────────────
  async function generateSyntheticData(scenario = 'CLEAN') {
    const formData = new FormData();
    formData.append('document_type', 'PASSPORT');
    formData.append('tampering_scenario', scenario);
    formData.append('country_code', 'IND');

    const res = await fetch('/api/v1/generate-synthetic-data', {
      method: 'POST',
      body: formData
    });
    if (!res.ok) throw new Error('Failed to generate synthetic document');
    return res.json();
  }

  // ── Public API surface ─────────────────────────────────────────────────
  return {
    fetchDashboardStats, fetchActivity, fetchCheckpoints, fetchAlerts, fetchAuditLog,
    stageExtract, stageValidate, stageDetect, stageVerify, stageLog, anchorDecision,
    authLogin, generateSyntheticData
  };
})();

window.API = API;

