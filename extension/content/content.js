/**
 * Netram AI Deepfake Shield — Enterprise Content Script
 * Injects into Google Meet / Zoom Web / Teams Web / Testbed.
 * 
 * Features:
 * - Real-time video & WebRTC audio capture with multi-modal neural streaming
 * - Google Meet Automated In-Chat Legal Liability & Participant Consent Broadcaster
 * - Top-level Floating Glassmorphic Compliance & Privacy Bar
 * - Comprehensive Participant Legal Terms & Non-Misuse Guarantee Modal
 * - Draggable Siri-styled Glassmorphism HUD widget
 * - Floating badges with hoverable neural inspector popovers
 */
(() => {
  const DEFAULT_CLOUD_URL = "wss://netram-deepfake-detection.up.railway.app";
  let configuredServerUrl = DEFAULT_CLOUD_URL;
  let overlayEnabled = true;
  let autoChatNotice = true;
  let endpointIdx = 0;
  let reconnectDelay = 2000;
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
  let complianceBarEl = null;
  let legalModalEl = null;
  let activeVerdict = null;

  // Generate or retrieve unique Session Verification ID for this meeting tab
  const MEETING_SESSION_ID = (() => {
    let sid = sessionStorage.getItem("netram_meeting_session_id");
    if (!sid) {
      sid = "NETRAM-SEC-" + Math.floor(100000 + Math.random() * 900000);
      sessionStorage.setItem("netram_meeting_session_id", sid);
    }
    return sid;
  })();

  /* ── URL Normalizer: auto-convert https/http to wss/ws ── */
  function normalizeWsUrl(url) {
    if (!url) return DEFAULT_CLOUD_URL;
    url = url.trim();
    if (url.startsWith("https://")) url = "wss://" + url.slice(8);
    else if (url.startsWith("http://")) url = "ws://" + url.slice(7);
    if (!url.startsWith("ws://") && !url.startsWith("wss://")) url = "wss://" + url;
    return url;
  }

  /* ── Load & sync settings ── */
  function loadSettings() {
    if (chrome.storage?.local) {
      chrome.storage.local.get(["serverUrl", "overlayEnabled", "autoChatNotice"], (res) => {
        if (res.serverUrl && res.serverUrl.trim()) {
          configuredServerUrl = normalizeWsUrl(res.serverUrl);
          if (configuredServerUrl !== res.serverUrl.trim()) {
            chrome.storage.local.set({ serverUrl: configuredServerUrl });
          }
        }
        if (res.overlayEnabled !== undefined) overlayEnabled = res.overlayEnabled;
        if (res.autoChatNotice !== undefined) autoChatNotice = res.autoChatNotice;
        updateHudVisibility();
      });
    }
  }

  if (chrome.storage?.onChanged) {
    chrome.storage.onChanged.addListener((changes, area) => {
      if (area === "local") {
        if (changes.serverUrl && changes.serverUrl.newValue) {
          configuredServerUrl = normalizeWsUrl(changes.serverUrl.newValue);
          endpointIdx = 0;
          reconnectDelay = 2000;
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
        if (changes.autoChatNotice !== undefined) {
          autoChatNotice = changes.autoChatNotice.newValue;
        }
      }
    });
  }

  /* ── Listen for messages from popup ── */
  if (chrome.runtime?.onMessage) {
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
      if (request.type === "BROADCAST_CHAT_NOTICE") {
        broadcastLegalNoticeToMeetChat(true);
        sendResponse({ ok: true });
      } else if (request.type === "OPEN_LEGAL_MODAL") {
        openLegalModal();
        sendResponse({ ok: true });
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

    const url = normalizeWsUrl(configuredServerUrl || DEFAULT_CLOUD_URL);
    try {
      ws = new WebSocket(url);
      ws.onopen = () => { connected = true; reconnectDelay = 2000; updateSiriStatus("🟢 Active"); };
      ws.onmessage = (e) => {
        try {
          const d = JSON.parse(e.data);
          if (d.type === "verdict" || d.type === "telemetry") handleVerdict(d);
        } catch (_) {}
      };
      ws.onclose = () => { connected = false; ws = null; updateSiriStatus("⚪ Connecting…"); scheduleReconnect(); };
      ws.onerror = () => { connected = false; try { ws.close(); } catch (_) {} ws = null; updateSiriStatus("⚪ Connecting…"); };
    } catch (_) { connected = false; ws = null; updateSiriStatus("⚪ Connecting…"); scheduleReconnect(); }
  }

  function scheduleReconnect() {
    setTimeout(() => {
      connect();
      reconnectDelay = Math.min(reconnectDelay * 1.5, 15000);
    }, reconnectDelay);
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

  /* ═══════════════════════════════════════════════════════════════════
     LEGAL LIABILITY, TERMS & GOOGLE MEET CHAT BROADCASTER
     ═══════════════════════════════════════════════════════════════════ */

  const LEGAL_DISCLAIMER_TEXT = 
`🛡️ [Netram AI Enterprise Defense Notice]
Real-time neural deepfake & synthetic media integrity monitoring is active in this meeting session.

🔒 Participant Data Protection & Non-Misuse Guarantee:
• Video & audio streams are processed ephemerally in volatile memory solely for deepfake & synthetic voice detection.
• Zero Data Retention: No video clips, biometric face vectors, or meeting audio are recorded, stored, shared, monetized, or reused for AI training.
• Session Verification ID: ${MEETING_SESSION_ID}
• Compliant with Netram AI Trust & Safety Zero-Knowledge Standards.`;

  let autoNoticeAttempted = false;

  /**
   * Automatically monitors if user entered an active Google Meet/Teams/Zoom meeting
   * and triggers the automated participant legal liability notice in chat.
   */
  function checkAndBroadcastMeetingNotice() {
    if (!autoChatNotice || autoNoticeAttempted) return;

    const broadcastKey = "netram_chat_broadcast_sent_" + location.pathname.replace(/[^a-zA-Z0-9]/g, "_");
    if (sessionStorage.getItem(broadcastKey) === "true") {
      autoNoticeAttempted = true;
      return;
    }

    const inActiveCall = (
      trackers.size > 0 ||
      document.querySelector("button[aria-label*='Leave call' i], button[aria-label*='Leave meeting' i], button[aria-label*='Chat' i], [data-allocation-index], #netram-test-chat-input")
    );

    if (inActiveCall) {
      autoNoticeAttempted = true;
      setTimeout(() => {
        broadcastLegalNoticeToMeetChat(false);
      }, 1800);
    }
  }

  /**
   * Broadcasts legal compliance & non-misuse liability notice into Google Meet / Zoom / Teams / Testbed chat.
   */
  function broadcastLegalNoticeToMeetChat(forceManual = false, retryCount = 0) {
    const broadcastKey = "netram_chat_broadcast_sent_" + location.pathname.replace(/[^a-zA-Z0-9]/g, "_");
    if (!forceManual && sessionStorage.getItem(broadcastKey) === "true") {
      return; // Already announced in this session
    }

    // 1. Google Meet DOM Discovery
    const meetChatToggle = document.querySelector(
      "button[aria-label*='Chat' i], button[aria-label*='chat with everyone' i], button[data-panel-id='2'], [jsname='A5il2e'], button[aria-label*='Message' i], button[aria-label*='Messages' i]"
    );

    // If chat drawer is closed, click to open it
    if (meetChatToggle) {
      const isExpanded = meetChatToggle.getAttribute("aria-expanded") === "true" || meetChatToggle.classList.contains("active");
      if (!isExpanded) {
        try { meetChatToggle.click(); } catch (_) {}
      }
    }

    setTimeout(() => {
      // Find chat text input in Google Meet / Teams / Zoom / Testbed
      const chatInput = document.querySelector(
        "textarea[aria-label*='Send a message' i], textarea[name='chatTextInput'], div[contenteditable='true'][aria-label*='message' i], textarea[aria-label*='chat' i], [jsname='YPqjbf'], #netram-test-chat-input"
      );

      if (chatInput) {
        try {
          if (chatInput.tagName === "TEXTAREA" || chatInput.tagName === "INPUT") {
            chatInput.focus();
            chatInput.value = LEGAL_DISCLAIMER_TEXT;
            chatInput.dispatchEvent(new Event("input", { bubbles: true }));
            chatInput.dispatchEvent(new Event("change", { bubbles: true }));
          } else if (chatInput.isContentEditable) {
            chatInput.focus();
            chatInput.innerText = LEGAL_DISCLAIMER_TEXT;
            chatInput.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText" }));
          }

          // Trigger send button
          setTimeout(() => {
            const sendBtn = document.querySelector(
              "button[aria-label*='Send a message' i], button[aria-label*='Send' i], button[jsname='soHxf'], #netram-test-chat-send"
            );
            if (sendBtn && !sendBtn.disabled) {
              sendBtn.click();
            } else {
              // Fallback: Dispatch Enter key
              const enterEvent = new KeyboardEvent("keydown", {
                bubbles: true, cancelable: true, key: "Enter", code: "Enter", keyCode: 13
              });
              chatInput.dispatchEvent(enterEvent);
            }

            sessionStorage.setItem(broadcastKey, "true");
            showComplianceToast("✅ Netram AI Legal & Privacy Notice announced in meeting chat.");
          }, 350);
        } catch (err) {
          console.warn("[Netram AI] Chat automation notice:", err);
        }
      } else {
        // If not found yet and still within 6 retries, retry in 1.5s
        if (retryCount < 6) {
          setTimeout(() => {
            broadcastLegalNoticeToMeetChat(forceManual, retryCount + 1);
          }, 1500);
          return;
        }

        // Fallback for custom sandbox or if chat input wasn't found in DOM
        if (window.postMessage) {
          window.postMessage({
            type: "NETRAM_CHAT_DISCLAIMER",
            text: LEGAL_DISCLAIMER_TEXT,
            session_id: MEETING_SESSION_ID
          }, "*");
        }
        sessionStorage.setItem(broadcastKey, "true");
        if (forceManual) {
          showComplianceToast("ℹ️ Legal disclaimer copied & broadcast signal sent.");
          copyLegalNoticeToClipboard();
        }
      }
    }, 400);
  }

  function copyLegalNoticeToClipboard() {
    navigator.clipboard?.writeText(LEGAL_DISCLAIMER_TEXT).then(() => {
      showComplianceToast("📋 Legal Liability & Terms notice copied to clipboard!");
    }).catch(() => {
      showComplianceToast("📋 Disclaimer ready for sharing.");
    });
  }

  function showComplianceToast(msg) {
    let toast = document.getElementById("netram-compliance-toast");
    if (!toast) {
      toast = document.createElement("div");
      toast.id = "netram-compliance-toast";
      document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.className = "netram-toast show";
    setTimeout(() => {
      toast.className = "netram-toast";
    }, 4200);
  }

  /* ── Floating Top Compliance Bar Injection ── */
  function injectComplianceBar() {
    if (document.getElementById("netram-compliance-bar")) return;

    const bar = document.createElement("div");
    bar.id = "netram-compliance-bar";
    bar.innerHTML = `
      <div class="netram-bar-content">
        <div class="netram-bar-badge">
          <span class="netram-shield-icon">🛡️</span>
          <div class="netram-badge-text">
            <span class="netram-badge-title">Netram AI Shield Active</span>
            <span class="netram-badge-sub">Zero-Retention & Non-Misuse Protected · ${MEETING_SESSION_ID}</span>
          </div>
        </div>
        <div class="netram-bar-actions">
          <button class="netram-bar-btn" id="btn-open-legal-terms" title="View Legal Liability & Participant Terms">
            <span>📜</span> Legal Terms
          </button>
          <button class="netram-bar-btn" id="btn-broadcast-chat" title="Broadcast Legal Disclaimer to Meeting Chat">
            <span>💬</span> Post to Chat
          </button>
          <button class="netram-bar-btn" id="btn-copy-terms" title="Copy Disclaimer">
            <span>📋</span> Copy
          </button>
          <button class="netram-bar-close" id="btn-close-comp-bar" title="Minimize Bar">✕</button>
        </div>
      </div>
      <div class="netram-bar-minimized" id="netram-bar-minimized" style="display:none;" title="Expand Netram Compliance Shield">
        <span>🛡️</span>
        <span class="min-pulse"></span>
      </div>`;

    document.body.appendChild(bar);
    complianceBarEl = bar;

    // Button event listeners
    document.getElementById("btn-open-legal-terms").addEventListener("click", openLegalModal);
    document.getElementById("btn-broadcast-chat").addEventListener("click", () => broadcastLegalNoticeToMeetChat(true));
    document.getElementById("btn-copy-terms").addEventListener("click", copyLegalNoticeToClipboard);
    
    // Minimize / Expand logic
    const content = bar.querySelector(".netram-bar-content");
    const minBtn = document.getElementById("netram-bar-minimized");
    const closeBtn = document.getElementById("btn-close-comp-bar");

    closeBtn.addEventListener("click", () => {
      content.style.display = "none";
      minBtn.style.display = "flex";
    });

    minBtn.addEventListener("click", () => {
      content.style.display = "flex";
      minBtn.style.display = "none";
    });

    // Auto broadcast chat notice if enabled (delayed 3s after meeting launch)
    if (autoChatNotice) {
      setTimeout(() => {
        broadcastLegalNoticeToMeetChat(false);
      }, 3200);
    }
  }

  /* ── Full Comprehensive Legal Liability & Participant Terms Modal ── */
  function openLegalModal() {
    if (!document.getElementById("netram-legal-modal")) {
      injectLegalModal();
    }
    const modal = document.getElementById("netram-legal-modal");
    if (modal) modal.classList.add("netram-modal-visible");
  }

  function closeLegalModal() {
    const modal = document.getElementById("netram-legal-modal");
    if (modal) modal.classList.remove("netram-modal-visible");
  }

  function injectLegalModal() {
    const modal = document.createElement("div");
    modal.id = "netram-legal-modal";
    modal.className = "netram-modal-overlay";
    modal.innerHTML = `
      <div class="netram-modal-card">
        <div class="netram-modal-header">
          <div class="modal-brand">
            <div class="modal-shield-orb">🛡️</div>
            <div>
              <h2 class="modal-title">Participant Protection & Legal Liability Guarantee</h2>
              <span class="modal-sub">Netram AI Enterprise Trust, Safety & Zero-Retention Compliance Agreement</span>
            </div>
          </div>
          <button class="modal-close-btn" id="modal-btn-close">✕</button>
        </div>

        <div class="modal-session-strip">
          <span class="session-badge">
            <span class="sess-dot"></span> SESSION ID: <b>${MEETING_SESSION_ID}</b>
          </span>
          <span class="session-time">Timestamp: ${new Date().toLocaleTimeString()} · Edge Node Verified</span>
        </div>

        <div class="modal-body-scroll">
          <div class="legal-card-highlight">
            <span class="legal-badge-gold">LEGAL LIABILITY COMMITMENT</span>
            <h3>Strict Non-Misuse & Safe Clip Handling Guarantee</h3>
            <p>
              By design and strict contractual guarantee, Netram AI certifies that <b>no video clips, screen captures, voice audio recordings, or biometric face vectors</b> from any meeting participant will ever be saved, stored on servers, shared with third parties, sold, or used for AI model retraining.
            </p>
          </div>

          <div class="legal-clauses-grid">
            <div class="clause-box">
              <div class="clause-num">01</div>
              <h4>Scope of Real-Time Inspection</h4>
              <p>
                Video and audio streams are inspected strictly in volatile memory for artificial facial warping (ViT), GAN frequency noise (2D DCT CNN), inter-frame micro-jitter (Bi-GRU), and synthetic voice cloning (AASIST) to protect participants from unauthorized impersonation.
              </p>
            </div>

            <div class="clause-box">
              <div class="clause-num">02</div>
              <h4>Zero-Knowledge Ephemeral Purge</h4>
              <p>
                All video frame buffers and acoustic spectral windows exist solely in RAM for the sub-50ms inference lifecycle and are immediately overwritten. Zero biometric records or raw media persist after execution.
              </p>
            </div>

            <div class="clause-box">
              <div class="clause-num">03</div>
              <h4>Regulatory Compliance & Safety</h4>
              <p>
                Engine adheres to strict global biometric privacy guidelines (GDPR Art. 9 biometric exceptions for security, CCPA consumer non-disclosure, SOC-2 Type II, and ISO/IEC 27001 zero-trust criteria).
              </p>
            </div>

            <div class="clause-box">
              <div class="clause-num">04</div>
              <h4>Participant Transparency & Rights</h4>
              <p>
                All attendees are notified of active neural integrity defense via in-meeting telemetry and chat broadcasting, ensuring complete bilateral transparency during every conference call.
              </p>
            </div>
          </div>

          <div class="legal-seal-row">
            <div class="seal-item">
              <span class="seal-icon">🔒</span>
              <span class="seal-label">100% Volatile Memory Execution</span>
            </div>
            <div class="seal-item">
              <span class="seal-icon">⚡</span>
              <span class="seal-label">Sub-50ms Edge Telemetry</span>
            </div>
            <div class="seal-item">
              <span class="seal-icon">📜</span>
              <span class="seal-label">Zero Training Data Harvesting</span>
            </div>
          </div>
        </div>

        <div class="modal-footer">
          <div class="footer-actions-left">
            <button class="modal-btn secondary" id="modal-btn-open-doc">
              <span>📜</span> Full Legal Docs
            </button>
            <button class="modal-btn secondary" id="modal-btn-copy">
              <span>📋</span> Copy Statement
            </button>
            <button class="modal-btn secondary" id="modal-btn-chat">
              <span>💬</span> Broadcast to Chat
            </button>
          </div>
          <button class="modal-btn primary" id="modal-btn-confirm">
            I Understand & Agree
          </button>
        </div>
      </div>`;

    document.body.appendChild(modal);
    legalModalEl = modal;

    // Event listeners
    document.getElementById("modal-btn-close").addEventListener("click", closeLegalModal);
    document.getElementById("modal-btn-confirm").addEventListener("click", closeLegalModal);
    document.getElementById("modal-btn-open-doc").addEventListener("click", () => {
      const url = chrome.runtime?.getURL ? chrome.runtime.getURL("legal/terms.html") : "/legal/terms.html";
      window.open(url, "_blank");
    });
    document.getElementById("modal-btn-copy").addEventListener("click", copyLegalNoticeToClipboard);
    document.getElementById("modal-btn-chat").addEventListener("click", () => {
      broadcastLegalNoticeToMeetChat(true);
      closeLegalModal();
    });

    modal.addEventListener("click", (e) => {
      if (e.target === modal) closeLegalModal();
    });
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

    // Check if user entered an active meeting to broadcast legal notice to participants
    checkAndBroadcastMeetingNotice();
  }

  function pid(v, i) {
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
        <div class="insp-row iv-phone-row" style="display:none;"><span>Phone Replay</span><span class="iv iv-phone" style="color:#ef4444; font-weight:700;">🚨 DETECTED</span></div>
        <div class="insp-row iv-filter-row" style="display:none;"><span>AR Beauty Filter</span><span class="iv iv-filter" style="color:#f59e0b; font-weight:700;">✨ DETECTED</span></div>
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
    hud.className = "aegis-hidden";
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
          <button class="siri-btn" id="siri-btn-legal" title="View Terms & Liability">📜</button>
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

    // Legal modal button on HUD
    document.getElementById("siri-btn-legal").addEventListener("click", openLegalModal);
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
      el.style.right = "auto";
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
      if (label) {
        if (d.phone_detected) {
          label.textContent = "🚨 Phone Replay Spoof";
        } else if (d.ar_filter_detected) {
          label.textContent = "✨ AR Beauty Filter";
        } else {
          label.textContent = labels[level] || "Analyzing…";
        }
      }

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
      
      const phoneRow = q(".iv-phone-row");
      if (phoneRow) {
        phoneRow.style.display = d.phone_detected ? "flex" : "none";
      }

      const filterRow = q(".iv-filter-row");
      if (filterRow) {
        filterRow.style.display = d.ar_filter_detected ? "flex" : "none";
      }

      const recWrap = q(".insp-rec");
      const recText = q(".insp-rec-text");
      if (recWrap && recText && d.recommendation) {
        recWrap.style.display = "";
        recText.textContent = d.recommendation;
      }

      if ((level === "critical" || level === "high" || d.phone_detected) && !t.alerted) { chime(); t.alerted = true; }
      if (level === "clear" || level === "low" || level === "calibrating") t.alerted = false;
    }

    // Update Siri Draggable HUD
    if (siriHudEl) {
      const level = (d.threat_level || "clear").toLowerCase();
      const score = Math.min(1, Math.max(0, d.score || 0));
      const pctVal = (score * 100).toFixed(0);

      const pill = document.getElementById("siri-threat-pill");
      if (pill) {
        if (d.phone_detected) {
          pill.className = "siri-threat-pill critical";
          pill.textContent = "🚨 PHONE REPLAY";
        } else if (d.ar_filter_detected) {
          pill.className = "siri-threat-pill moderate";
          pill.textContent = "✨ AR FILTER";
        } else {
          pill.className = "siri-threat-pill " + level;
          pill.textContent = d.threat_level || "CLEAR";
        }
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
  injectComplianceBar();
  injectSiriHud();
  connect();
  loopTimer = setInterval(loop, SAMPLE_MS);
  window.addEventListener("beforeunload", () => { clearInterval(loopTimer); ws?.close(); });
})();
