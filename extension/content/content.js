/**
 * Aegis Shield — Enterprise Content Script with Draggable Siri Floating HUD
 * Injects into Google Meet / Zoom Web / Teams Web.
 * 
 * Features:
 * - Real-time video & WebRTC audio capture
 * - Draggable Siri-styled Glassmorphism HUD widget
 * - Floating badges with hoverable neural inspector popovers
 * - Smooth forensic metric updates & caution calibration notice
 */
(() => {
  const DEFAULT_CLOUD_URL = "wss://netram-deepfake-detection.onrender.com";
  let configuredServerUrl = DEFAULT_CLOUD_URL;
  let overlayEnabled = true;
  let endpointIdx = 0;
  const SAMPLE_MS = 300;       // ~3.3 FPS per participant
  const MAX_TILES = 4;

  let ws = null;
  let connected = false;
  let trackers = new Map();
  let loopTimer = null;
  const osc = document.createElement("canvas");
  const ctx = osc.getContext("2d", { willReadFrequently: true });

  let audioCtx = null;
  let siriHudEl = null;
  let activeVerdict = null;

  /* ── Load & sync settings ── */
  function loadSettings() {
    if (chrome.storage?.local) {
      chrome.storage.local.get(["serverUrl", "overlayEnabled"], (res) => {
        if (res.serverUrl && res.serverUrl.trim()) configuredServerUrl = res.serverUrl.trim();
        if (res.overlayEnabled !== undefined) overlayEnabled = res.overlayEnabled;
        updateHudVisibility();
      });
    }
  }

  if (chrome.storage?.onChanged) {
    chrome.storage.onChanged.addListener((changes, area) => {
      if (area === "local") {
        if (changes.serverUrl && changes.serverUrl.newValue) {
          configuredServerUrl = changes.serverUrl.newValue.trim();
          endpointIdx = 0;
          if (ws) {
            try { ws.close(); } catch (_) {}
            ws = null;
          }
          connect();
        }
        if (changes.overlayEnabled !== undefined) {
          overlayEnabled = changes.overlayEnabled.newValue;
          updateHudVisibility();
        }
      }
    });
  }

  function getAudioContext() {
    if (!audioCtx) {
      try {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      } catch (_) {}
    }
    return audioCtx;
  }

  /* ── Synth alert chime ── */
  function chime() {
    try {
      const ac = getAudioContext() || new (window.AudioContext || window.webkitAudioContext)();
      const o = ac.createOscillator(), g = ac.createGain();
      o.type = "sine"; o.frequency.setValueAtTime(660, ac.currentTime);
      o.frequency.exponentialRampToValueAtTime(1100, ac.currentTime + 0.12);
      g.gain.setValueAtTime(0.15, ac.currentTime);
      g.gain.exponentialRampToValueAtTime(0.01, ac.currentTime + 0.25);
      o.connect(g); g.connect(ac.destination); o.start(); o.stop(ac.currentTime + 0.3);
    } catch (_) {}
  }

  /* ── WebSocket with Auto-Reconnect & Configurable Endpoint ── */
  function connect() {
    if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) return;

    const endpoints = [];
    if (configuredServerUrl) endpoints.push(configuredServerUrl);
    if (!endpoints.includes(DEFAULT_CLOUD_URL)) endpoints.push(DEFAULT_CLOUD_URL);
    if (!endpoints.includes("ws://127.0.0.1:8765")) endpoints.push("ws://127.0.0.1:8765");
    if (!endpoints.includes("ws://localhost:8765")) endpoints.push("ws://localhost:8765");

    const url = endpoints[endpointIdx % endpoints.length];
    try {
      ws = new WebSocket(url);
      ws.onopen = () => { connected = true; updateSiriStatus("🟢 Active"); };
      ws.onmessage = (e) => {
        try {
          const d = JSON.parse(e.data);
          if (d.type === "verdict" || d.type === "telemetry") handleVerdict(d);
        } catch (_) {}
      };
      ws.onclose = () => { connected = false; ws = null; endpointIdx++; updateSiriStatus("⚪ Connecting…"); setTimeout(connect, 2500); };
      ws.onerror = () => { connected = false; try { ws.close(); } catch (_) {} ws = null; endpointIdx++; updateSiriStatus("⚪ Connecting…"); setTimeout(connect, 2500); };
    } catch (_) { connected = false; ws = null; endpointIdx++; updateSiriStatus("⚪ Connecting…"); setTimeout(connect, 2500); }
  }

  /* ── Visibility controller: only show HUD when camera/video is actively streaming ── */
  function updateHudVisibility() {
    if (!siriHudEl) return;
    const hasActiveCamera = trackers.size > 0;
    if (hasActiveCamera && overlayEnabled) {
      siriHudEl.classList.remove("aegis-hidden");
    } else {
      siriHudEl.classList.add("aegis-hidden");
    }
  }

  /* ── Video tile discovery & audio track attachment ── */
  function scan() {
    const allVideos = Array.from(document.querySelectorAll("video"));
    const videos = allVideos.filter(v => {
      const hasFrames = v.videoWidth > 0 && v.videoHeight > 0;
      const rect = v.getBoundingClientRect();
      const isVisible = rect.width > 20 && rect.height > 20;
      return (hasFrames || isVisible) && !v.paused;
    }).slice(0, MAX_TILES);

    videos.forEach((v, i) => {
      if (!trackers.has(v)) {
        const id = pid(v, i);
        const t = {
          id: id,
          video: v,
          badge: null,
          data: null,
          alerted: false,
          audioBuffer: [],
          audioSource: null
        };
        trackers.set(v, t);
        injectBadge(t);
        attachAudio(t);
      }
    });

    for (const [v, t] of trackers) {
      if (!videos.includes(v) || !document.body.contains(v)) {
        t.badge?.remove();
        trackers.delete(v);
      }
    }

    // Update floating HUD visibility based on active camera feeds
    updateHudVisibility();
  }

  function pid(v, i) {
    // Search upwards in DOM for real participant name in Google Meet / Zoom / Teams
    let current = v.parentElement;
    for (let depth = 0; depth < 10 && current && current !== document.body; depth++) {
      const nameEl = current.querySelector(
        "div.poVWob, div.d27c6d, div.j7304c, span.notranslate, div.notranslate, div.P3g2nd, [data-self-name], [jsname='r4nke'], .ZmdEae"
      );
      if (nameEl && nameEl.textContent.trim()) {
        const txt = nameEl.textContent.trim();
        if (txt.length >= 2 && !txt.includes("http") && !txt.includes("{")) {
          return txt.slice(0, 24);
        }
      }
      current = current.parentElement;
    }
    const mt = v.closest("[data-participant-id]") || v.closest("[data-requested-participant-id]");
    if (mt) {
      const attr = mt.getAttribute("data-participant-id") || mt.getAttribute("data-requested-participant-id");
      if (attr) return "User " + attr.slice(-4);
    }
    return "Participant " + (i + 1);
  }

  /* ── WebRTC Audio Ingestion Hook ── */
  function attachAudio(tracker) {
    try {
      const v = tracker.video;
      if (!v.srcObject || !(v.srcObject instanceof MediaStream)) return;
      const audioTracks = v.srcObject.getAudioTracks();
      if (!audioTracks || audioTracks.length === 0) return;

      const actx = getAudioContext();
      if (!actx) return;

      const stream = new MediaStream(audioTracks);
      const source = actx.createMediaStreamSource(stream);
      const processor = actx.createScriptProcessor(4096, 1, 1);

      processor.onaudioprocess = (evt) => {
        const inputData = evt.inputBuffer.getChannelData(0);
        for (let j = 0; j < inputData.length; j++) {
          tracker.audioBuffer.push(inputData[j]);
        }
        if (tracker.audioBuffer.length > 32000) {
          tracker.audioBuffer = tracker.audioBuffer.slice(-32000);
        }
      };

      source.connect(processor);
      processor.connect(actx.destination);
      tracker.audioSource = source;
    } catch (_) {}
  }

  /* ── Badge injection on Video Tiles ── */
  function injectBadge(t) {
    const parent = t.video.parentElement;
    if (!parent) return;
    parent.classList.add("aegis-participant-wrapper");
    const b = document.createElement("div");
    b.className = "aegis-badge clear";
    b.innerHTML = `
      <span class="aegis-dot"></span>
      <span class="aegis-label">Analyzing…</span>
      <div class="aegis-inspector">
        <div class="insp-header"><span class="insp-title">NEURAL INSPECTOR</span><span class="insp-lat">—</span></div>
        <div class="insp-row"><span>Threat Level</span><span class="iv iv-threat">—</span></div>
        <div class="insp-row"><span>Spatial ViT</span><span class="iv iv-spatial">—</span></div>
        <div class="insp-row"><span>Spectral CNN</span><span class="iv iv-freq">—</span></div>
        <div class="insp-row"><span>Temporal GRU</span><span class="iv iv-temporal">—</span></div>
        <div class="insp-row"><span>Liveness EAR</span><span class="iv iv-liveness">—</span></div>
        <div class="insp-row"><span>Voice Clone</span><span class="iv iv-voice">—</span></div>
        <div class="insp-rec" style="display:none;">
          <span class="insp-rec-title">AI Recommendation</span>
          <p class="insp-rec-text"></p>
        </div>
      </div>`;
    parent.appendChild(b);
    t.badge = b;
  }

  /* ── Siri-styled Draggable Floating HUD ── */
  function injectSiriHud() {
    if (document.getElementById("aegis-siri-hud")) return;
    const hud = document.createElement("div");
    hud.id = "aegis-siri-hud";
    hud.className = "aegis-hidden"; // Hidden until active camera is detected
    hud.innerHTML = `
      <div class="siri-header" id="aegis-siri-drag-handle">
        <div class="siri-orb-container">
          <div class="siri-orb"></div>
          <div class="siri-title-group">
            <span class="siri-title">Netram Neural HUD</span>
            <span class="siri-subtitle" id="siri-status-text">🟢 Active</span>
          </div>

        </div>
        <div class="siri-controls">
          <button class="siri-btn" id="siri-btn-min" title="Minimize/Expand">_</button>
        </div>
      </div>
      <div class="siri-body" id="siri-body">
        <div class="siri-caution-banner" id="siri-caution">
          <span class="siri-caution-icon">ℹ️</span>
          <span id="siri-caution-text">Initial baseline calibration in progress (3–5s). Forensic confidence accumulates smoothly.</span>
        </div>
        <div class="siri-hero">
          <svg class="siri-ring-svg" viewBox="0 0 52 52">
            <circle class="siri-ring-track" cx="26" cy="26" r="22"/>
            <circle class="siri-ring-fill" id="siri-ring-fill" cx="26" cy="26" r="22" stroke-dasharray="138.2" stroke-dashoffset="138.2"/>
            <text class="siri-ring-text" id="siri-ring-pct" x="26" y="27" text-anchor="middle" dominant-baseline="middle">0%</text>
          </svg>
          <div class="siri-status-info">
            <span class="siri-threat-pill clear" id="siri-threat-pill">CLEAR</span>
            <span class="siri-participant-name" id="siri-participant-name">Awaiting feed…</span>
          </div>
        </div>
        <div class="siri-metrics">
          <div class="siri-metric-row">
            <div class="siri-metric-header"><span>Spatial ViT (Artifacts)</span><span id="siri-v-spatial">0.0%</span></div>
            <div class="siri-bar-track"><div class="siri-bar-fill" id="siri-b-spatial"></div></div>
          </div>
          <div class="siri-metric-row">
            <div class="siri-metric-header"><span>Spectral Frequency CNN</span><span id="siri-v-freq">0.0%</span></div>
            <div class="siri-bar-track"><div class="siri-bar-fill" id="siri-b-freq"></div></div>
          </div>
          <div class="siri-metric-row">
            <div class="siri-metric-header"><span>Temporal Bi-GRU</span><span id="siri-v-temporal">0.0%</span></div>
            <div class="siri-bar-track"><div class="siri-bar-fill" id="siri-b-temporal"></div></div>
          </div>
          <div class="siri-metric-row">
            <div class="siri-metric-header"><span>Liveness & Blink EAR</span><span id="siri-v-liveness">0.0%</span></div>
            <div class="siri-bar-track"><div class="siri-bar-fill" id="siri-b-liveness"></div></div>
          </div>
        </div>
        <div class="siri-rec-box">
          <span class="siri-rec-label">AI Forensic Assessment</span>
          <p class="siri-rec-content" id="siri-rec-content">Nominal readings. No indicators of synthetic manipulation detected.</p>
        </div>
      </div>`;
    document.body.appendChild(hud);
    siriHudEl = hud;

    // Draggable Logic
    makeDraggable(hud, document.getElementById("aegis-siri-drag-handle"));

    // Minimize toggle
    document.getElementById("siri-btn-min").addEventListener("click", () => {
      hud.classList.toggle("minimized");
    });
  }

  function makeDraggable(el, handle) {
    let isDragging = false;
    let startX, startY, initLeft, initTop;

    handle.addEventListener("mousedown", (e) => {
      if (e.target.tagName === "BUTTON") return;
      isDragging = true;
      startX = e.clientX;
      startY = e.clientY;
      const rect = el.getBoundingClientRect();
      initLeft = rect.left;
      initTop = rect.top;
      el.style.right = "auto"; // Switch to left/top positioning
      el.style.left = initLeft + "px";
      el.style.top = initTop + "px";
      document.body.style.userSelect = "none";
    });

    window.addEventListener("mousemove", (e) => {
      if (!isDragging) return;
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;
      let newLeft = initLeft + dx;
      let newTop = initTop + dy;

      // Clamp within viewport
      newLeft = Math.max(10, Math.min(window.innerWidth - el.offsetWidth - 10, newLeft));
      newTop = Math.max(10, Math.min(window.innerHeight - el.offsetHeight - 10, newTop));

      el.style.left = newLeft + "px";
      el.style.top = newTop + "px";
    });

    window.addEventListener("mouseup", () => {
      isDragging = false;
      document.body.style.userSelect = "";
    });
  }

  function updateSiriStatus(txt) {
    const el = document.getElementById("siri-status-text");
    if (el) el.textContent = txt;
  }

  /* ── Verdict handler ── */
  function handleVerdict(d) {
    activeVerdict = d;

    // Update on-video badges
    for (const [, t] of trackers) {
      if (t.id !== d.participant_id && !t.id.includes(d.participant_id) && !d.participant_id.includes(t.id)) continue;
      if (!t.badge) continue;
      t.data = d;
      const level = (d.threat_level || "clear").toLowerCase();

      t.badge.className = "aegis-badge " + level;
      const label = t.badge.querySelector(".aegis-label");

      const labels = {
        clear: "Verified Authentic",
        calibrating: "Calibrating (3s)…",
        low: "Minor Irregularities",
        moderate: "⚠ Review Advised",
        high: "⚠ Probable Manipulation",
        critical: "🚨 Synthetic Media Detected",
      };
      if (label) label.textContent = labels[level] || "Analyzing…";

      const v = t.video;
      v.className = v.className.replace(/aegis-border-\w+/g, "").trim();
      v.classList.add("aegis-border-" + level);

      const q = (sel) => t.badge.querySelector(sel);
      const pct = (val) => ((val || 0) * 100).toFixed(1) + "%";
      if (q(".insp-lat")) q(".insp-lat").textContent = (d.latency_ms || 0).toFixed(0) + " ms";
      if (q(".iv-threat")) { q(".iv-threat").textContent = d.threat_level || "—"; q(".iv-threat").className = "iv iv-threat " + level; }
      if (q(".iv-spatial")) q(".iv-spatial").textContent = pct(d.p_spatial);
      if (q(".iv-freq")) q(".iv-freq").textContent = pct(d.p_freq);
      if (q(".iv-temporal")) q(".iv-temporal").textContent = pct(d.p_temporal);
      if (q(".iv-liveness")) q(".iv-liveness").textContent = pct(d.p_liveness);
      if (q(".iv-voice")) q(".iv-voice").textContent = d.p_voice_clone != null ? pct(d.p_voice_clone) : "N/A";

      const recWrap = q(".insp-rec");
      const recText = q(".insp-rec-text");
      if (recWrap && recText && d.recommendation) {
        recWrap.style.display = "";
        recText.textContent = d.recommendation;
      }

      if ((level === "critical" || level === "high") && !t.alerted) { chime(); t.alerted = true; }
      if (level === "clear" || level === "low" || level === "calibrating") t.alerted = false;
    }

    // Update Siri Draggable HUD
    if (siriHudEl) {
      const level = (d.threat_level || "clear").toLowerCase();
      const score = Math.min(1, Math.max(0, d.score || 0));
      const pctVal = (score * 100).toFixed(0);

      const pill = document.getElementById("siri-threat-pill");
      if (pill) {
        pill.className = "siri-threat-pill " + level;
        pill.textContent = d.threat_level || "CLEAR";
      }

      const pName = document.getElementById("siri-participant-name");
      if (pName) pName.textContent = d.participant_id || "Participant";

      // SVG Ring
      const circ = 2 * Math.PI * 22; // 138.2
      const offset = circ * (1 - score);
      const ringFill = document.getElementById("siri-ring-fill");
      if (ringFill) {
        ringFill.style.strokeDashoffset = offset.toFixed(1);
        let color = "#30d98b";
        if (score >= 0.82) color = "#f06060";
        else if (score >= 0.65) color = "#ff9f43";
        else if (score >= 0.45) color = "#f5a623";
        else if (score >= 0.25) color = "#4fc3f7";
        else if (level === "calibrating") color = "#af52de";
        ringFill.style.stroke = color;
      }

      const ringPct = document.getElementById("siri-ring-pct");
      if (ringPct) ringPct.textContent = pctVal + "%";

      // Caution Note
      const cautionText = document.getElementById("siri-caution-text");
      if (cautionText && d.caution_note) {
        cautionText.textContent = d.caution_note;
      }

      // Metric Bars
      const setMetric = (id, val) => {
        const v = Math.min(1, Math.max(0, val || 0));
        const p = (v * 100).toFixed(1);
        const txt = document.getElementById("siri-v-" + id);
        const bar = document.getElementById("siri-b-" + id);
        if (txt) txt.textContent = p + "%";
        if (bar) {
          bar.style.width = p + "%";
          if (v > 0.6) bar.style.background = "#f06060";
          else if (v > 0.35) bar.style.background = "#f5a623";
          else bar.style.background = "#30d98b";
        }
      };
      setMetric("spatial", d.p_spatial);
      setMetric("freq", d.p_freq);
      setMetric("temporal", d.p_temporal);
      setMetric("liveness", d.p_liveness);

      // AI Recommendation
      const recContent = document.getElementById("siri-rec-content");
      if (recContent && d.recommendation) {
        recContent.textContent = d.recommendation;
      }
    }
  }

  /* ── Capture & stream loop ── */
  function loop() {
    scan();
    if (!connected || !ws || ws.readyState !== 1) return;
    for (const [, t] of trackers) {
      const v = t.video;
      if (v.videoWidth <= 0 || v.videoHeight <= 0) continue;

      const vw = v.videoWidth || 640;
      const vh = v.videoHeight || 480;
      const aspect = vw / vh;
      let targetW = 640;
      let targetH = Math.round(640 / aspect);
      if (targetH > 480) { targetH = 480; targetW = Math.round(480 * aspect); }

      osc.width = targetW;
      osc.height = targetH;
      ctx.drawImage(v, 0, 0, targetW, targetH);

      let audioB64 = null;
      if (t.audioBuffer && t.audioBuffer.length >= 1600) {
        const floatArray = new Float32Array(t.audioBuffer.slice(-16000));
        const byteArray = new Uint8Array(floatArray.buffer);
        let binary = "";
        for (let k = 0; k < byteArray.length; k++) {
          binary += String.fromCharCode(byteArray[k]);
        }
        audioB64 = btoa(binary);
      }

      ws.send(JSON.stringify({
        type: "frame",
        participant_id: t.id,
        image: osc.toDataURL("image/jpeg", 0.85),
        audio: audioB64,
        timestamp: performance.now() / 1000,
      }));
    }
  }

  // Initialize
  loadSettings();
  injectSiriHud();
  connect();
  loopTimer = setInterval(loop, SAMPLE_MS);
  window.addEventListener("beforeunload", () => { clearInterval(loopTimer); ws?.close(); });
})();
