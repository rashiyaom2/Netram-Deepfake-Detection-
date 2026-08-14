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

/* ─── Settings ─── */
function loadSettings() {
  if (chrome.storage?.local) {
    chrome.storage.local.get(["overlayEnabled", "audioAlertEnabled", "serverUrl"], (r) => {
      if (r.overlayEnabled !== undefined) document.getElementById("toggle-overlay").checked = r.overlayEnabled;
      if (r.audioAlertEnabled !== undefined) document.getElementById("toggle-audio").checked = r.audioAlertEnabled;
      const input = document.getElementById("input-server-url");
      if (r.serverUrl && r.serverUrl.trim()) {
        configuredServerUrl = r.serverUrl.trim();
        if (input) input.value = configuredServerUrl;
      } else {
        if (input) input.value = DEFAULT_CLOUD_URL;
      }
    });
  }
}

function setupListeners() {
  document.getElementById("toggle-overlay").addEventListener("change", (e) => {
    chrome.storage?.local?.set({ overlayEnabled: e.target.checked });
  });
  document.getElementById("toggle-audio").addEventListener("change", (e) => {
    chrome.storage?.local?.set({ audioAlertEnabled: e.target.checked });
  });
  document.getElementById("btn-sync").addEventListener("click", () => {
    currentEndpointIdx = 0;
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
      const val = inputServer.value.trim();
      if (val) {
        configuredServerUrl = val;
        chrome.storage?.local?.set({ serverUrl: val }, () => {
          btnSaveServer.textContent = "Saved ✓";
          setTimeout(() => { btnSaveServer.textContent = "Save"; }, 1500);
          currentEndpointIdx = 0;
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

  const endpoints = [];
  if (configuredServerUrl) endpoints.push(configuredServerUrl);
  if (!endpoints.includes(DEFAULT_CLOUD_URL)) endpoints.push(DEFAULT_CLOUD_URL);
  if (!endpoints.includes("ws://127.0.0.1:8765")) endpoints.push("ws://127.0.0.1:8765");
  if (!endpoints.includes("ws://localhost:8765")) endpoints.push("ws://localhost:8765");

  const url = endpoints[currentEndpointIdx % endpoints.length];

  try {
    socket = new WebSocket(url);
    socket.onopen = () => {
      pill.className = "conn-pill online";
      text.textContent = "Engine Active";
      if (tb) tb.style.display = "none";
      if (reconnectTimer) { clearInterval(reconnectTimer); reconnectTimer = null; }
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
      currentEndpointIdx++;
      scheduleReconnect();
    };
    socket.onerror = () => {
      pill.className = "conn-pill offline";
      text.textContent = "Offline";
      if (tb) tb.style.display = "block";
      try { socket.close(); } catch (_) { }
      socket = null;
      currentEndpointIdx++;
      scheduleReconnect();
    };
  } catch (_) {
    pill.className = "conn-pill offline";
    text.textContent = "Offline";
    if (tb) tb.style.display = "block";
    socket = null;
    currentEndpointIdx++;
    scheduleReconnect();
  }
}

function scheduleReconnect() {
  if (!reconnectTimer) {
    reconnectTimer = setInterval(() => {
      connectEngine();
    }, 2000);
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