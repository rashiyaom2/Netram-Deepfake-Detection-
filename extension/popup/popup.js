/**
 * Netram AI Shield — Popup Controller
 * Instant Multi-Model Switcher (Railway, Render, Local PC), Real-time Zoom/Meet
 * Participant Fetching, Draggable Telemetry Inspection, and Forensic Breakdowns.
 */

const RING_CIRCUMFERENCE = 2 * Math.PI * 52;
let socket = null;
let auditLog = [];
let participants = {};
let selectedParticipantId = null;
let displayScore = 0;
let targetScore = 0;

const DEFAULT_CLOUD_URL = "ws://127.0.0.1:8765";
let configuredServerUrl = DEFAULT_CLOUD_URL;
let reconnectTimer = null;
let reconnectDelay = 2000;
let pollTabTimer = null;

const hasChrome = typeof chrome !== "undefined" && chrome.storage && chrome.storage.local;

document.addEventListener("DOMContentLoaded", () => {
  loadSettings();
  setupEnginePillListeners();
  setupListeners();
  startDisplayLoop();
  fetchLiveParticipantsFromTab();

  // Poll active tab every 1.5s for live Zoom / Meet updates
  pollTabTimer = setInterval(fetchLiveParticipantsFromTab, 1500);
});

/* ─── Engine URL normalizer ─── */
function normalizeWsUrl(url) {
  if (!url) return "ws://127.0.0.1:8765";
  url = url.trim();
  if (url.includes("127.0.0.1") || url.includes("localhost") || url.includes(":8765")) {
    url = url.replace(/^https?:\/\//i, "ws://").replace(/^wss:\/\//i, "ws://");
    if (!url.startsWith("ws://")) url = "ws://" + url;
    return url;
  }
  if (url.startsWith("https://")) url = "wss://" + url.slice(8);
  else if (url.startsWith("http://")) url = "ws://" + url.slice(7);
  if (!url.startsWith("ws://") && !url.startsWith("wss://")) url = "wss://" + url;
  return url;
}

/* ─── Settings Persistence & Model Active Pill State ─── */
function loadSettings() {
  if (!hasChrome) {
    updateEnginePillUI(DEFAULT_CLOUD_URL);
    connectEngine();
    return;
  }

  chrome.storage.local.get(
    ["overlayEnabled", "audioAlertEnabled", "autoChatNotice", "serverUrl", "termsAccepted", "latestVerdict"],
    (r) => {
      document.getElementById("terms-bar").style.display = r.termsAccepted ? "none" : "flex";
      if (r.overlayEnabled !== undefined) document.getElementById("toggle-overlay").checked = r.overlayEnabled;
      if (r.audioAlertEnabled !== undefined) document.getElementById("toggle-audio").checked = r.audioAlertEnabled;
      if (r.autoChatNotice !== undefined) document.getElementById("toggle-chat-notice").checked = r.autoChatNotice;

      const input = document.getElementById("input-server-url");
      if (r.serverUrl && r.serverUrl.trim()) {
        configuredServerUrl = normalizeWsUrl(r.serverUrl);
        if (input) input.value = configuredServerUrl;
      } else {
        configuredServerUrl = DEFAULT_CLOUD_URL;
        if (input) input.value = DEFAULT_CLOUD_URL;
      }

      updateEnginePillUI(configuredServerUrl);
      connectEngine();

      if (r.latestVerdict) {
        onVerdict(r.latestVerdict);
      }
    }
  );
}

/* ─── Model / Engine Selector Controller ─── */
function setupEnginePillListeners() {
  const pills = [
    { id: "pill-railway", url: "wss://netram-deepfake-detection.up.railway.app" },
    { id: "pill-render", url: "wss://netram-deepfake-detection.onrender.com" },
    { id: "pill-local", url: "ws://127.0.0.1:8765" }
  ];

  pills.forEach((p) => {
    const el = document.getElementById(p.id);
    if (!el) return;
    el.addEventListener("click", () => {
      switchEngine(p.url);
    });
  });
}

function switchEngine(url) {
  const normalized = normalizeWsUrl(url);
  configuredServerUrl = normalized;
  updateEnginePillUI(normalized);

  const input = document.getElementById("input-server-url");
  if (input) input.value = normalized;

  if (hasChrome) {
    chrome.storage.local.set({ serverUrl: normalized });
    // Notify active meeting tab to switch WebSocket stream immediately
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs && tabs[0]?.id) {
        chrome.tabs.sendMessage(tabs[0].id, { type: "SWITCH_SERVER", serverUrl: normalized });
      }
    });
  }

  // Reconnect popup WebSocket
  reconnectDelay = 2000;
  if (socket) {
    try { socket.close(); } catch (_) { }
    socket = null;
  }
  connectEngine();
}

function updateEnginePillUI(activeUrl) {
  const norm = normalizeWsUrl(activeUrl);
  const isRailway = norm.includes("railway.app");
  const isRender = norm.includes("render.com");
  const isLocal = norm.includes("127.0.0.1") || norm.includes("localhost") || norm.includes(":8765");

  document.getElementById("pill-railway")?.classList.toggle("active", isRailway);
  document.getElementById("pill-render")?.classList.toggle("active", isRender);
  document.getElementById("pill-local")?.classList.toggle("active", isLocal);

  const textDot = document.getElementById("conn-text");
  if (textDot) {
    if (isLocal) textDot.title = "Local PC Engine (ws://127.0.0.1:8765)";
    else if (isRender) textDot.title = "Render Cloud Engine";
    else textDot.title = "Railway GPU Cloud Engine";
  }
}

/* ─── Fetch Live Participants from Active Zoom / Meet Tab ─── */
function fetchLiveParticipantsFromTab() {
  if (!hasChrome || !chrome.tabs?.query) return;

  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (!tabs || !tabs[0]?.id) return;
    try {
      chrome.tabs.sendMessage(tabs[0].id, { type: "GET_PARTICIPANTS" }, (response) => {
        if (chrome.runtime.lastError || !response) return;

        if (response.participants && Array.isArray(response.participants)) {
          if (response.participants.length > 0) {
            response.participants.forEach((p) => {
              participants[p.id] = p;
            });
            renderParticipants();

            // Auto-focus latest active participant if none is chosen
            if (!selectedParticipantId || !participants[selectedParticipantId]) {
              selectedParticipantId = response.participants[0].id;
            }
            if (participants[selectedParticipantId]) {
              renderFocusedVerdict(participants[selectedParticipantId]);
            }
          }
        }
      });
    } catch (_) { }
  });
}

/* ─── WebSocket Connection & Telemetry Engine ─── */
function connectEngine() {
  const dot = document.getElementById("conn-dot");
  const text = document.getElementById("conn-text");
  const liveDot = document.getElementById("engine-live-dot");
  const liveText = document.getElementById("engine-live-text");
  const tb = document.getElementById("troubleshoot-box");

  if (socket && (socket.readyState === WebSocket.CONNECTING || socket.readyState === WebSocket.OPEN)) return;

  if (dot) dot.className = "conn-dot";
  if (text) text.textContent = "Connecting…";
  if (liveDot) liveDot.className = "engine-live-dot";
  if (liveText) liveText.textContent = "Connecting…";

  if (socket) {
    try { socket.close(); } catch (_) { }
    socket = null;
  }

  const url = normalizeWsUrl(configuredServerUrl || DEFAULT_CLOUD_URL);

  try {
    socket = new WebSocket(url);

    socket.onopen = () => {
      if (dot) dot.className = "conn-dot online";
      if (text) text.textContent = "Online";
      if (liveDot) liveDot.className = "engine-dot online";
      if (liveText) {
        const isLocal = url.includes("127.0.0.1") || url.includes(":8765");
        const isRender = url.includes("render.com");
        liveText.textContent = isLocal ? "Local Online" : isRender ? "Render Online" : "Railway Online";
      }
      if (tb) tb.classList.remove("show");
      if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
      reconnectDelay = 2000;
    };

    socket.onmessage = (e) => {
      try {
        const d = JSON.parse(e.data);
        if (d.type === "verdict" || d.type === "telemetry") onVerdict(d);
      } catch (_) { }
    };

    socket.onclose = () => {
      setOfflineState();
      scheduleReconnect();
    };

    socket.onerror = () => {
      setOfflineState();
      try { socket.close(); } catch (_) { }
      socket = null;
    };
  } catch (_) {
    setOfflineState();
    scheduleReconnect();
  }
}

function setOfflineState() {
  const dot = document.getElementById("conn-dot");
  const text = document.getElementById("conn-text");
  const liveDot = document.getElementById("engine-live-dot");
  const liveText = document.getElementById("engine-live-text");
  const tb = document.getElementById("troubleshoot-box");

  if (dot) dot.className = "conn-dot";
  if (text) text.textContent = "Offline";
  if (liveDot) liveDot.className = "engine-dot offline";
  if (liveText) liveText.textContent = "Offline";
  if (tb) tb.classList.add("show");
  socket = null;
}

function scheduleReconnect() {
  if (!reconnectTimer) {
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connectEngine();
      reconnectDelay = Math.min(reconnectDelay * 1.5, 12000);
    }, reconnectDelay);
  }
}

/* ─── Progressive SVG Ring Animation ─── */
function startDisplayLoop() {
  setInterval(() => {
    const diff = targetScore - displayScore;
    displayScore += diff * 0.15;
    if (Math.abs(diff) < 0.001) displayScore = targetScore;
    renderRing(displayScore);
  }, 100);
}

function renderRing(score) {
  const pct = Math.min(100, Math.max(0, score * 100));
  const offset = RING_CIRCUMFERENCE * (1 - score);
  const fill = document.getElementById("ring-fill");
  if (fill) {
    fill.setAttribute("stroke-dashoffset", offset.toFixed(2));
    let color, riskClass;
    if (pct >= 82) { color = "var(--red)"; riskClass = "risk-high"; }
    else if (pct >= 65) { color = "var(--orange)"; riskClass = "risk-high"; }
    else if (pct >= 45) { color = "var(--yellow)"; riskClass = "risk-mid"; }
    else if (pct >= 25) { color = "var(--blue)"; riskClass = ""; }
    else { color = "var(--green)"; riskClass = ""; }
    fill.setAttribute("stroke", color);
  }

  const ringPct = document.getElementById("ring-pct");
  if (ringPct) ringPct.textContent = pct.toFixed(0) + "%";

  const hero = document.getElementById("hero");
  if (hero) {
    hero.classList.remove("risk-mid", "risk-high");
    if (pct >= 65) hero.classList.add("risk-high");
    else if (pct >= 45) hero.classList.add("risk-mid");
  }
}

/* ─── Verdict Handler ─── */
function onVerdict(d) {
  auditLog.push(d);
  if (auditLog.length > 500) auditLog.shift();

  const pid = d.participant_id || "participant_1";
  participants[pid] = d;

  if (!selectedParticipantId || selectedParticipantId === pid || Object.keys(participants).length === 1) {
    selectedParticipantId = pid;
    renderFocusedVerdict(d);
  }

  renderParticipants();
}

function renderFocusedVerdict(d) {
  targetScore = d.score ?? 0;

  const level = (d.threat_level || "CLEAR").toLowerCase();
  const badge = document.getElementById("threat-badge");
  if (badge) {
    if (d.phone_detected) {
      badge.textContent = "PHONE REPLAY DETECTED";
      badge.className = "threat-badge critical";
    } else if (d.ar_filter_detected) {
      badge.textContent = "AR BEAUTY FILTER";
      badge.className = "threat-badge moderate";
    } else {
      badge.textContent = d.threat_level || "Clear";
      badge.className = "threat-badge " + level;
    }
  }

  const lbl = document.getElementById("threat-label");
  if (lbl) lbl.textContent = d.threat_label || "Awaiting analysis";

  const sub = document.getElementById("threat-sublabel");
  if (sub) {
    sub.textContent = d.caution_note || (d.confidence_tier ? `Detection confidence: ${d.confidence_tier}` : "Real-time monitoring active.");
  }

  const recLine = document.getElementById("recommendation-line");
  if (recLine) {
    if (d.recommendation) {
      recLine.style.display = "block";
      recLine.textContent = d.recommendation;
    } else {
      recLine.style.display = "none";
    }
  }

  setBar("spatial", d.p_spatial);
  setBar("freq", d.p_freq);
  setBar("temporal", d.p_temporal);
  setBar("liveness", d.p_liveness);
  setBar("jitter", d.jitter);

  const phoneRow = document.getElementById("row-phone");
  if (phoneRow) phoneRow.style.display = d.phone_detected ? "flex" : "none";

  const filterRow = document.getElementById("row-filter");
  if (filterRow) filterRow.style.display = d.ar_filter_detected ? "flex" : "none";

  if (d.latency_ms != null) {
    const latEl = document.getElementById("meta-latency");
    if (latEl) latEl.textContent = d.latency_ms.toFixed(0) + " ms";
  }
  if (d.frame_idx != null) {
    const frEl = document.getElementById("meta-frames");
    if (frEl) frEl.textContent = d.frame_idx;
  }
}

function setBar(id, val) {
  const v = Math.min(1, Math.max(0, val || 0));
  const pct = (v * 100).toFixed(1);
  const bar = document.getElementById("bar-" + id);
  const label = document.getElementById("val-" + id);
  if (bar) {
    bar.style.width = pct + "%";
    if (v > 0.6) bar.style.background = "var(--red)";
    else if (v > 0.35) bar.style.background = "var(--yellow)";
    else bar.style.background = "var(--green)";
  }
  if (label) label.textContent = pct + "%";
}

/* ─── Render Live Participants List ─── */
function renderParticipants() {
  const el = document.getElementById("participant-list");
  const countEl = document.getElementById("p-count");
  const keys = Object.keys(participants);

  if (countEl) countEl.textContent = keys.length;
  if (!el) return;

  if (!keys.length) {
    el.innerHTML = `
      <div class="empty-participants-wrap">
        <p class="empty-hint">No active video stream detected. Join a video meeting to begin real-time inspection.</p>
        <div class="empty-quick-links">
          <button type="button" class="quick-launch-chip" id="launch-meet-btn">Google Meet ↗</button>
          <button type="button" class="quick-launch-chip" id="launch-zoom-btn">Zoom Web ↗</button>
          <button type="button" class="quick-launch-chip" id="launch-testbed-btn">Sandbox ↗</button>
        </div>
      </div>`;
    setupLaunchListeners();
    return;
  }

  el.innerHTML = "";
  keys.forEach((pid) => {
    const d = participants[pid];
    const level = (d.threat_level || "clear").toLowerCase();
    const scorePct = Math.round((d.score || 0) * 100);
    const initials = esc(pid).slice(0, 2).toUpperCase();
    const isSelected = (selectedParticipantId === pid);

    let statusText = d.threat_level || "Clear";
    if (d.phone_detected) statusText = "Phone Replay";
    else if (d.ar_filter_detected) statusText = "AR Filter";

    const card = document.createElement("div");
    card.className = `p-card ${isSelected ? "active" : ""}`;
    card.title = `Click to inspect ${pid} — ${statusText} (${scorePct}%)`;
    card.innerHTML = `
      <span class="p-avatar ${level}">${initials}</span>
      <div class="p-card-info">
        <span class="p-card-name">${esc(pid)}</span>
        <span class="p-card-status ${level}">${statusText} <b class="p-card-score">${scorePct}%</b></span>
      </div>`;

    card.addEventListener("click", () => {
      selectedParticipantId = pid;
      renderFocusedVerdict(d);
      renderParticipants();
    });

    el.appendChild(card);
  });
}

function setupLaunchListeners() {
  document.getElementById("launch-meet-btn")?.addEventListener("click", () => {
    window.open("https://meet.google.com/new", "_blank");
  });
  document.getElementById("launch-zoom-btn")?.addEventListener("click", () => {
    window.open("https://app.zoom.us/wc", "_blank");
  });
  document.getElementById("launch-testbed-btn")?.addEventListener("click", () => {
    if (hasChrome && chrome.runtime?.getURL) {
      window.open(chrome.runtime.getURL("test_meeting.html"), "_blank");
    } else {
      window.open("../test_meeting.html", "_blank");
    }
  });
  document.getElementById("launch-onboarding-btn")?.addEventListener("click", openOnboarding);
}

function openOnboarding() {
  if (hasChrome && chrome.tabs?.create) {
    chrome.tabs.create({ url: chrome.runtime.getURL("onboarding/onboarding.html") });
  } else {
    window.open("../onboarding/onboarding.html", "_blank");
  }
}

/* ─── UI View Switching & Action Listeners ─── */
function setupListeners() {
  document.getElementById("btn-onboarding-top")?.addEventListener("click", openOnboarding);
  document.getElementById("btn-open-onboarding")?.addEventListener("click", openOnboarding);

  document.getElementById("btn-gear")?.addEventListener("click", () => {
    document.getElementById("view-home")?.classList.add("hidden-left");
    document.getElementById("view-settings")?.classList.remove("hidden-right");
    document.getElementById("footer-home").style.display = "none";
  });

  document.getElementById("btn-back")?.addEventListener("click", () => {
    document.getElementById("view-home")?.classList.remove("hidden-left");
    document.getElementById("view-settings")?.classList.add("hidden-right");
    document.getElementById("footer-home").style.display = "flex";
  });

  document.getElementById("toggle-overlay")?.addEventListener("change", (e) => hasChrome && chrome.storage.local.set({ overlayEnabled: e.target.checked }));
  document.getElementById("toggle-audio")?.addEventListener("change", (e) => hasChrome && chrome.storage.local.set({ audioAlertEnabled: e.target.checked }));
  document.getElementById("toggle-chat-notice")?.addEventListener("change", (e) => hasChrome && chrome.storage.local.set({ autoChatNotice: e.target.checked }));

  document.getElementById("btn-accept-terms")?.addEventListener("click", () => {
    document.getElementById("terms-bar").style.display = "none";
    if (hasChrome) chrome.storage.local.set({ termsAccepted: true, termsAcceptedTimestamp: new Date().toISOString() });
  });

  document.getElementById("btn-review-terms")?.addEventListener("click", openLegal);
  document.getElementById("btn-open-legal")?.addEventListener("click", openLegal);

  function openLegal() {
    if (hasChrome && chrome.tabs?.create) {
      chrome.tabs.create({ url: chrome.runtime.getURL("legal/terms.html") });
    } else {
      window.open("../legal/terms.html", "_blank");
    }
  }

  document.getElementById("btn-broadcast-chat")?.addEventListener("click", () => {
    if (hasChrome) {
      chrome.tabs?.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs && tabs[0]?.id) chrome.tabs.sendMessage(tabs[0].id, { type: "BROADCAST_CHAT_NOTICE" });
      });
    }
  });

  document.getElementById("btn-sync")?.addEventListener("click", () => {
    reconnectDelay = 2000;
    if (socket) { try { socket.close(); } catch (_) { } socket = null; }
    connectEngine();
    fetchLiveParticipantsFromTab();
  });

  document.getElementById("btn-export")?.addEventListener("click", exportLog);

  const btnSaveServer = document.getElementById("btn-save-server");
  const inputServer = document.getElementById("input-server-url");
  if (btnSaveServer && inputServer) {
    btnSaveServer.addEventListener("click", () => {
      let val = inputServer.value.trim();
      if (!val) return;
      switchEngine(val);
      btnSaveServer.textContent = "Saved";
      setTimeout(() => { btnSaveServer.textContent = "Save"; }, 1500);
    });
  }

  document.querySelectorAll(".server-hint-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const url = chip.getAttribute("data-url");
      if (url) switchEngine(url);
    });
  });

  document.getElementById("btn-fallback-local")?.addEventListener("click", () => {
    switchEngine("ws://127.0.0.1:8765");
  });

  document.getElementById("forensics-toggle")?.addEventListener("click", () => {
    document.getElementById("accordion-body")?.classList.toggle("open");
    document.getElementById("chevron")?.classList.toggle("open");
  });

  setupLaunchListeners();
}

function exportLog() {
  const blob = new Blob([JSON.stringify(auditLog, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `netram-audit-${new Date().toISOString().slice(0, 19).replace(/:/g, "-")}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function esc(s) {
  return String(s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
