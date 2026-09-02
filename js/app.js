/* =========================================================================
   PRAMAAN AI — App Shell & UI Helpers
   Frontend logic only. No hardcoded domain data here (see mock-api.js).
   ========================================================================= */

const PRAMAAN = (() => {

  const STAGES = [
    { code:"01", key:"capture", label:"Capture" },
    { code:"02", key:"extract", label:"Extract" },
    { code:"03", key:"validate", label:"Validate" },
    { code:"04", key:"detect", label:"Detect" },
    { code:"05", key:"verify", label:"Verify" },
    { code:"06", key:"log", label:"Log/Anchor" },
  ];

  const OFFICER = { name:"Insp. R. Bhandari", badge:"SSB-BH-22417", checkpost:"Raxaul ICP", sector:"Sector 4" };

  const ICONS = {
    dashboard: `<svg viewBox="0 0 24 24" fill="none" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3.5" y="3.5" width="7.2" height="9" rx="1"/><rect x="13.3" y="3.5" width="7.2" height="5.5" rx="1"/><rect x="13.3" y="11.5" width="7.2" height="9" rx="1"/><rect x="3.5" y="15" width="7.2" height="5.5" rx="1"/></svg>`,
    screening: `<svg viewBox="0 0 24 24" fill="none" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="3" width="16" height="18" rx="1.4"/><path d="M8 8h8M8 12h8M8 16h5"/></svg>`,
    audit: `<svg viewBox="0 0 24 24" fill="none" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6h16M4 12h16M4 18h10"/><circle cx="19.5" cy="18" r="1.6"/></svg>`,
    checkpoints: `<svg viewBox="0 0 24 24" fill="none" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 21s7-6.2 7-11.5A7 7 0 0 0 5 9.5C5 14.8 12 21 12 21z"/><circle cx="12" cy="9.3" r="2.4"/></svg>`,
    alerts: `<svg viewBox="0 0 24 24" fill="none" stroke-width="2.0" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9a6 6 0 0 1 12 0c0 5 2 6 2 6H4s2-1 2-6z"/><path d="M10 19a2.2 2.2 0 0 0 4 0"/></svg>`,
    settings: `<svg viewBox="0 0 24 24" fill="none" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 13.5a1.7 1.7 0 0 0 .34 1.87l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.7 1.7 0 0 0-1.87-.34 1.7 1.7 0 0 0-1 1.55V19.5a2 2 0 1 1-4 0v-.09a1.7 1.7 0 0 0-1.1-1.55 1.7 1.7 0 0 0-1.87.34l-.06.06A2 2 0 1 1 4.2 15.4l.06-.06a1.7 1.7 0 0 0 .34-1.87 1.7 1.7 0 0 0-1.55-1H2.5a2 2 0 1 1 0-4h.09a1.7 1.7 0 0 0 1.55-1.1 1.7 1.7 0 0 0-.34-1.87l-.06-.06A2 2 0 1 1 6.57 2.6l.06.06a1.7 1.7 0 0 0 1.87.34H8.6a1.7 1.7 0 0 0 1-1.55V1.4a2 2 0 1 1 4 0v.09a1.7 1.7 0 0 0 1 1.55 1.7 1.7 0 0 0 1.87-.34l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.7 1.7 0 0 0-.34 1.87v.09a1.7 1.7 0 0 0 1.55 1H21.5a2 2 0 1 1 0 4h-.09a1.7 1.7 0 0 0-1.55 1.05z"/></svg>`,
    doc: `<svg viewBox="0 0 24 24" fill="none" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M6 2.8h8.2L19 7.6V21a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V3.8a1 1 0 0 1 1-1z"/><path d="M14 2.8V8h5"/><path d="M8.3 12.4h6.8M8.3 15.6h6.8M8.3 18.8h4"/></svg>`,
    check: `<svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4.5 12.5l5 5 10-11"/></svg>`,
    flagIcon: `<svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 21V4"/><path d="M6 4h11l-2.5 4L17 12H6"/></svg>`,
    x: `<svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 5l14 14M19 5L5 19"/></svg>`,
    sync: `<svg viewBox="0 0 24 24" fill="none" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12a8 8 0 0 1 13.6-5.7L20 8.5"/><path d="M20 4v4.5h-4.5"/><path d="M20 12a8 8 0 0 1-13.6 5.7L4 15.5"/><path d="M4 20v-4.5h4.5"/></svg>`,
    emblem: `<svg viewBox="0 0 32 32" fill="none"><circle cx="16" cy="16" r="14.5" stroke="#3A7CA5" stroke-width="1.6"/><path d="M16 16 L16.0 30.0 M16 16 L19.6 29.5 M16 16 L23.0 28.1 M16 16 L25.9 25.9 M16 16 L28.1 23.0 M16 16 L29.5 19.6 M16 16 L30.0 16.0 M16 16 L29.5 12.4 M16 16 L28.1 9.0 M16 16 L25.9 6.1 M16 16 L23.0 3.9 M16 16 L19.6 2.5 M16 16 L16.0 2.0 M16 16 L12.4 2.5 M16 16 L9.0 3.9 M16 16 L6.1 6.1 M16 16 L3.9 9.0 M16 16 L2.5 12.4 M16 16 L2.0 16.0 M16 16 L2.5 19.6 M16 16 L3.9 23.0 M16 16 L6.1 25.9 M16 16 L9.0 28.1 M16 16 L12.4 29.5" stroke="#3A7CA5" stroke-width="0.8"/><circle cx="16" cy="16" r="2.5" fill="#3A7CA5"/></svg>`,
  };

  function truncHash(h){ return h.slice(0,10) + '…' + h.slice(-6); }

  // Real SHA-256 via Web Crypto API (Async)
  async function hashFromRef(ref) {
    const msgUint8 = new TextEncoder().encode(ref);
    const hashBuffer = await crypto.subtle.digest('SHA-256', msgUint8);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  }

  function initClock(el){
    function tick(){
      const now = new Date();
      const opts = { hour:'2-digit', minute:'2-digit', second:'2-digit', hour12:false, timeZone:'Asia/Kolkata' };
      el.textContent = now.toLocaleTimeString('en-GB', opts) + ' IST';
    }
    tick();
    setInterval(tick, 1000);
  }

  const NAV = [
    { id:"dashboard",   href:"index.html",       label:"Dashboard",   icon:"dashboard" },
    { id:"screening",   href:"screening.html",   label:"Screening",   icon:"screening" },
    { id:"audit",       href:"audit.html",       label:"Audit Log",   icon:"audit" },
    { id:"checkpoints", href:"checkpoints.html", label:"Checkpoints", icon:"checkpoints" },
    { id:"alerts",      href:"alerts.html",      label:"Alerts",      icon:"alerts" },
    { id:"settings",    href:"settings.html",    label:"Settings",    icon:"settings" },
  ];

  function renderShell(activeId, opts){
    opts = opts || {};
    const cpName = opts.checkpost || OFFICER.checkpost;
    const cpSector = opts.sector || OFFICER.sector;
    const navHTML = NAV.map(n => `
      <a class="nav-item ${n.id===activeId?'active':''}" href="${n.href}">
        ${ICONS[n.icon]}<span>${n.label}</span>
      </a>`).join('');

    return `
      <div class="tricolour-rule"></div>
      <div class="app-body">
        <aside class="sidebar">
          <div class="sidebar-mark">
            <div class="sidebar-emblem">${ICONS.emblem}</div>
            <div class="tricolour" style="display:flex; flex-direction:column; height:28px; width:4px; margin-right:12px; border-radius:2px; overflow:hidden; opacity: 0.9;">
              <div style="flex:1; background:#FF9933;"></div>
              <div style="flex:1; background:#FFFFFF;"></div>
              <div style="flex:1; background:#138808;"></div>
            </div>
            <div class="sidebar-mark-text">
              <div class="name">PRAMAAN AI</div>
              <div class="sub">प्रमाण · SSB Document Screening</div>
            </div>
          </div>
          <nav class="nav-group">${navHTML}</nav>
          <div class="sidebar-foot">
            <div class="offline-chip is-live" id="syncChip">
              <span class="dot"></span><span id="syncLabel">Synced · local ledger current</span>
            </div>
          </div>
        </aside>
        <div style="flex:1; min-width:0; display:flex; flex-direction:column;">
          <header class="topbar">
            <div class="topbar-checkpost">
              <div>
                <div class="cp-name">${cpName}</div>
                <div class="cp-sector">${cpSector}</div>
              </div>
            </div>
            <div class="topbar-right">
              <div class="topbar-clock" id="clock"></div>
              <div class="topbar-divider"></div>
              <div class="topbar-officer">
                <div class="officer-badge">RB</div>
                <div class="who">
                  <div class="n">${OFFICER.name}</div>
                  <div class="id">${OFFICER.badge}</div>
                </div>
              </div>
            </div>
          </header>
          <main class="main" id="mainContent"></main>
        </div>
      </div>`;
  }

  
  function mount(activeId, opts){
    const sessionStr = sessionStorage.getItem('pramaan_session');
    const isLogin = window.location.pathname.endsWith('login.html');
    
    if (!sessionStr && !isLogin) {
      window.location.href = 'login.html';
      return null;
    }
    
    // Default fallback for layout if session is missing but somehow bypassing redirect
    const session = sessionStr ? JSON.parse(sessionStr) : { badgeId: OFFICER.badge, name: OFFICER.name, role: 'officer' };

    document.body.insertAdjacentHTML('afterbegin', '<div class="app-shell" id="appShell"></div>');
    const shell = document.getElementById('appShell');
    shell.innerHTML = renderShell(activeId, session, opts);
    initClock(document.getElementById('clock'));
    return document.getElementById('mainContent');
  }
return { STAGES, hashFromRef, OFFICER, ICONS, truncHash, mount };
})();

window.PRAMAAN = PRAMAAN;
