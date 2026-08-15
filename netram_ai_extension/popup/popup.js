/**
 * Aegis Shield — Popup Controller
 * Handles WebSocket connection, progressive verdict display (1s smoothing),
 * collapsible forensics, and AI recommendation rendering.
 */

const RING_CIRCUMFERENCE = 2 * Math.PI * 52; // matches SVG r=52
let socket = null;
let auditLog = [];
let participants = {};
let displayScore = 0;        // smoothed display value (updated every second)
let targetScore = 0;         // latest server value
let animFrameId = null;

document.addEventListener("DOMContentLoaded", () => {
  loadSettings();
  connectEngine();
  setupListeners();
  startDisplayLoop();
});

const DEFAULT_CLOUD_URL = "wss://netram-deepfake-detection.onrender.com";
let configuredServerUrl = DEFAULT_CLOUD_URL;
let reconnectTimer = null;
let currentEndpointIdx = 0;
let reconnectDelay = 2000;

/* ─── URL Normalizer: auto-convert https/http to wss/ws ─── */
function normalizeWsUrl(url) {
  if (!url) return DEFAULT_CLOUD_URL;
  url = url.trim();
  if (url.startsWith("https://")) url = "wss://" + url.slice(8);
  else if (url.startsWith("http://")) url = "ws://" + url.slice(7);
  if (!url.startsWith("ws://") && !url.startsWith("wss://")) url = "wss://" + url;
  return url;
}

/* ─── Settings ─── */
function loadSettings() {
  if (chrome.storage?.local) {
    chrome.storage.local.get(["overlayEnabled", "audioAlertEnabled", "autoChatNotice", "serverUrl", "termsAccepted"], (r) => {
      const termsBanner = document.getElementById("terms-banner-card");
      if (termsBanner) {
        termsBanner.style.display = r.termsAccepted ? "none" : "flex";
      }

      if (r.overlayEnabled !== undefined && document.getElementById("toggle-overlay")) document.getElementById("toggle-overlay").checked = r.overlayEnabled;
      if (r.audioAlertEnabled !== undefined && document.getElementById("toggle-audio")) document.getElementById("toggle-audio").checked = r.audioAlertEnabled;
      if (r.autoChatNotice !== undefined && document.getElementById("toggle-chat-notice")) document.getElementById("toggle-chat-notice").checked = r.autoChatNotice;
      const input = document.getElementById("input-server-url");
      if (r.serverUrl && r.serverUrl.trim()) {
        configuredServerUrl = normalizeWsUrl(r.serverUrl);
        // Auto-fix stored URL if it was wrong
        if (configuredServerUrl !== r.serverUrl.trim()) {
          chrome.storage.local.set({ serverUrl: configuredServerUrl });
        }
        if (input) input.value = configuredServerUrl;
      } else {
        if (input) input.value = DEFAULT_CLOUD_URL;
      }
    });
  }
}

function setupListeners() {
  document.getElementById("toggle-overlay")?.addEventListener("change", (e) => {
    chrome.storage?.local?.set({ overlayEnabled: e.target.checked });
  });
  document.getElementById("toggle-audio")?.addEventListener("change", (e) => {
    chrome.storage?.local?.set({ audioAlertEnabled: e.target.checked });
  });
  document.getElementById("toggle-chat-notice")?.addEventListener("change", (e) => {
    chrome.storage?.local?.set({ autoChatNotice: e.target.checked });
  });

  // Mandatory Terms Acceptance button in Popup
  document.getElementById("btn-popup-accept-terms")?.addEventListener("click", () => {
    chrome.storage?.local?.set({
      termsAccepted: true,
      termsAcceptedTimestamp: new Date().toISOString()
    }, () => {
      const termsBanner = document.getElementById("terms-banner-card");
      if (termsBanner) termsBanner.style.display = "none";
    });
  });

  // Participant Terms & Legal Docs button
  document.getElementById("btn-open-legal-popup")?.addEventListener("click", () => {
    if (chrome.tabs?.create) {
      chrome.tabs.create({ url: chrome.runtime.getURL("legal/terms.html") });
    } else {
      window.open("../legal/terms.html", "_blank");
    }
    chrome.tabs?.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs && tabs[0]?.id) {
        chrome.tabs.sendMessage(tabs[0].id, { type: "OPEN_LEGAL_MODAL" });
      }
    });
  });

  // Broadcast Notice to Active Call Chat button
  const btnBroadcast = document.getElementById("btn-broadcast-chat-popup");
  btnBroadcast?.addEventListener("click", () => {
    chrome.tabs?.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs && tabs[0]?.id) {
        chrome.tabs.sendMessage(tabs[0].id, { type: "BROADCAST_CHAT_NOTICE" });
        btnBroadcast.textContent = "Announced ✓";
        setTimeout(() => { btnBroadcast.textContent = "💬 Broadcast to Call Chat"; }, 2000);
      }
    });
  });

  document.getElementById("btn-sync").addEventListener("click", () => {
    currentEndpointIdx = 0;
    reconnectDelay = 2000;
    if (socket) {
      try { socket.close(); } catch (_) { }
      socket = null;
    }
    connectEngine();
  });
  document.getElementById("btn-export").addEventListener("click", exportLog);

  // Server URL save button
  const btnSaveServer = document.getElementById("btn-save-server");
  const inputServer = document.getElementById("input-server-url");
  if (btnSaveServer && inputServer) {
    btnSaveServer.addEventListener("click", () => {
      let val = inputServer.value.trim();
      if (val) {
        val = normalizeWsUrl(val);
        inputServer.value = val;
        configuredServerUrl = val;
        chrome.storage?.local?.set({ serverUrl: val }, () => {
          btnSaveServer.textContent = "Saved ✓";
          setTimeout(() => { btnSaveServer.textContent = "Save"; }, 1500);
          currentEndpointIdx = 0;
          reconnectDelay = 2000;
          if (socket) {
            try { socket.close(); } catch (_) { }
            socket = null;
          }
          connectEngine();
        });
      }
    });
  }

  // Server hint chips
  document.querySelectorAll(".server-hint-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const url = chip.getAttribute("data-url");
      if (url && inputServer) {
        inputServer.value = url;
        btnSaveServer?.click();
      }
    });
  });

  // Direct offline fallback button
  document.getElementById("btn-fallback-local")?.addEventListener("click", () => {
    if (inputServer) {
      inputServer.value = "ws://127.0.0.1:8765";
      btnSaveServer?.click();
    }
  });

  // Forensics expand/collapse
  document.getElementById("expand-btn").addEventListener("click", () => {
    const body = document.getElementById("expand-body");
    const chev = document.getElementById("chevron");
    body.classList.toggle("open");
    chev.classList.toggle("open");
  });
}

/* ─── WebSocket with Configurable Endpoint & Fallback ─── */
function connectEngine() {
  const pill = document.getElementById("conn-pill");
  const text = document.getElementById("conn-text");
  const tb = document.getElementById("troubleshoot-box");

  if (socket && (socket.readyState === WebSocket.CONNECTING || socket.readyState === WebSocket.OPEN)) {
    return;
  }

  pill.className = "conn-pill offline";
  text.textContent = "Connecting…";
  if (socket) {
    try { socket.close(); } catch (_) { }
    socket = null;
  }

  // Only use cloud URL — no localhost fallback for cloud deployments
  const url = normalizeWsUrl(configuredServerUrl || DEFAULT_CLOUD_URL);

  try {
    socket = new WebSocket(url);
    socket.onopen = () => {
      pill.className = "conn-pill online";
      text.textContent = "Engine Active";
      if (tb) tb.style.display = "none";
      if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
      reconnectDelay = 2000; // reset backoff on success
    };
    socket.onmessage = (e) => {
      try {
        const d = JSON.parse(e.data);
        if (d.type === "verdict" || d.type === "telemetry") onVerdict(d);
      } catch (_) { }
    };
    socket.onclose = () => {
      pill.className = "conn-pill offline";
      text.textContent = "Offline";
      if (tb) tb.style.display = "block";
      socket = null;
      scheduleReconnect();
    };
    socket.onerror = () => {
      pill.className = "conn-pill offline";
      text.textContent = "Offline";
      if (tb) tb.style.display = "block";
      try { socket.close(); } catch (_) { }
      socket = null;
    };
  } catch (_) {
    pill.className = "conn-pill offline";
    text.textContent = "Offline";
    if (tb) tb.style.display = "block";
    socket = null;
    scheduleReconnect();
  }
}

function scheduleReconnect() {
  if (!reconnectTimer) {
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connectEngine();
      // Exponential backoff: 2s → 4s → 8s → max 15s
      reconnectDelay = Math.min(reconnectDelay * 1.5, 15000);
    }, reconnectDelay);
  }
}

/* ─── Progressive Display Loop (1 Hz update, smooth interpolation) ─── */
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
  fill.setAttribute("stroke-dashoffset", offset.toFixed(2));

  let color;
  if (pct >= 82) color = "var(--red)";
  else if (pct >= 65) color = "var(--orange)";
  else if (pct >= 45) color = "var(--amber)";
  else if (pct >= 25) color = "var(--blue)";
  else color = "var(--green)";
  fill.setAttribute("stroke", color);
  document.getElementById("ring-pct").textContent = pct.toFixed(0) + "%";
}

/* ─── Verdict Handler ─── */
function onVerdict(d) {
  auditLog.push(d);
  if (auditLog.length > 500) auditLog.shift();

  const pid = d.participant_id || "participant_1";
  participants[pid] = d;

  targetScore = d.score ?? 0;

  const level = (d.threat_level || "CLEAR").toLowerCase();
  const badge = document.getElementById("threat-badge");
  badge.textContent = d.threat_level || "CLEAR";
  badge.className = "threat-badge " + level;

  document.getElementById("threat-label").textContent = d.threat_label || "Awaiting Analysis";
  document.getElementById("threat-sublabel").textContent =
    d.confidence_tier ? `Detection Confidence: ${d.confidence_tier}` : "";

  const cautionText = document.getElementById("caution-text");
  if (cautionText && d.caution_note) {
    cautionText.textContent = d.caution_note;
  }

  const recCard = document.getElementById("recommendation-card");
  if (d.recommendation) {
    recCard.style.display = "";
    document.getElementById("recommendation-text").textContent = d.recommendation;
    if (d.dominant_signal && d.score > 0.2) {
      document.getElementById("dominant-row").style.display = "flex";
      document.getElementById("dominant-name").textContent = d.dominant_signal;
    } else {
      document.getElementById("dominant-row").style.display = "none";
    }
  }

  setBar("spatial", d.p_spatial);
  setBar("freq", d.p_freq);
  setBar("temporal", d.p_temporal);
  setBar("liveness", d.p_liveness);
  setBar("jitter", d.jitter);
  if (d.latency_ms != null) document.getElementById("meta-latency").textContent = d.latency_ms.toFixed(0) + " ms";
  if (d.frame_idx != null) document.getElementById("meta-frames").textContent = d.frame_idx;

  document.getElementById("p-count").textContent = Object.keys(participants).length;
  renderParticipants();
}

function setBar(id, val) {
  const v = Math.min(1, Math.max(0, val || 0));
  const pct = (v * 100).toFixed(1);
  const bar = document.getElementById("bar-" + id);
  const label = document.getElementById("val-" + id);
  if (bar) {
    bar.style.width = pct + "%";
    if (v > 0.6) bar.style.background = "var(--red)";
    else if (v > 0.35) bar.style.background = "var(--amber)";
    else bar.style.background = "var(--green)";
  }
  if (label) label.textContent = pct + "%";
}

function renderParticipants() {
  const el = document.getElementById("participant-list");
  const keys = Object.keys(participants);
  if (!keys.length) { el.innerHTML = '<p class="empty-hint">No active video call detected.</p>'; return; }
  el.innerHTML = "";
  keys.forEach((pid) => {
    const d = participants[pid];
    const level = (d.threat_level || "clear").toLowerCase();
    const row = document.createElement("div");
    row.className = "p-row";
    row.innerHTML = `
      <span class="p-dot ${level}"></span>
      <span class="p-name">${esc(pid)}</span>
      <span class="p-threat-tag">${d.threat_level || "—"}</span>`;
    el.appendChild(row);
  });
}

/* ─── Utilities ─── */
function exportLog() {
  const blob = new Blob([JSON.stringify(auditLog, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `aegis-audit-${new Date().toISOString().slice(0, 19).replace(/:/g, "-")}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }