/**
 * Netram AI — Onboarding Logic (5-Slide Flow)
 * Interactive carousel, mini-HUD drag simulation, live testbed video streams, engine probe.
 */
(() => {
  let currentSlide = 0;
  const slides = document.querySelectorAll(".slide");
  const tabs = document.querySelectorAll(".step-tab");

  let activeWebcamStream = null;
  const simAnomalies = { jitter: false, freeze: false, desync: false };

  function goToSlide(idx) {
    if (idx < 0 || idx >= slides.length) return;
    slides.forEach((s, i) => {
      s.classList.remove("active", "prev");
      if (i === idx) s.classList.add("active");
      else if (i < idx) s.classList.add("prev");
    });
    tabs.forEach((t, i) => t.classList.toggle("active", i === idx));
    currentSlide = idx;
  }

  document.querySelectorAll(".next-btn").forEach((btn) => {
    btn.addEventListener("click", () => goToSlide(parseInt(btn.getAttribute("data-next"), 10)));
  });
  document.querySelectorAll(".prev-btn").forEach((btn) => {
    btn.addEventListener("click", () => goToSlide(parseInt(btn.getAttribute("data-prev"), 10)));
  });
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => goToSlide(parseInt(tab.getAttribute("data-slide"), 10)));
  });

  const hasChrome = typeof chrome !== "undefined" && chrome.storage && chrome.storage.local;

  /* ── Terms gating ── */
  const cbTerms = document.getElementById("onboarding-cb-terms");
  const launchBtn = document.getElementById("btn-launch-meet");
  const openTestbedBtn = document.getElementById("btn-open-testbed");

  function updateTermsState() {
    const accepted = cbTerms ? cbTerms.checked : false;
    if (launchBtn) launchBtn.disabled = !accepted;
    if (openTestbedBtn) openTestbedBtn.disabled = !accepted;
    if (hasChrome) {
      chrome.storage.local.set({ termsAccepted: accepted, termsAcceptedTimestamp: accepted ? new Date().toISOString() : null });
    }
  }
  if (cbTerms) cbTerms.addEventListener("change", updateTermsState);

  if (hasChrome) {
    chrome.storage.local.get(["termsAccepted"], (res) => {
      if (res.termsAccepted) {
        if (cbTerms) cbTerms.checked = true;
        if (launchBtn) launchBtn.disabled = false;
        if (openTestbedBtn) openTestbedBtn.disabled = false;
      }
    });
  }

  const skipBtn = document.getElementById("btn-skip");
  if (skipBtn) skipBtn.addEventListener("click", () => goToSlide(4));

  if (launchBtn) {
    launchBtn.addEventListener("click", () => {
      if (!cbTerms?.checked) { alert("Please accept the Terms of Service & Participant Protection Guarantee to continue."); return; }
      window.open("https://meet.google.com/new", "_blank");
    });
  }
  if (openTestbedBtn) {
    openTestbedBtn.addEventListener("click", () => {
      if (!cbTerms?.checked) { alert("Please accept the Terms of Service & Participant Protection Guarantee to continue."); return; }
      if (hasChrome && chrome.runtime?.getURL) window.open(chrome.runtime.getURL("test_meeting.html"), "_blank");
      else window.open("/test_meeting.html", "_blank");
    });
  }

  /* ── HUD drag demo ── */
  const demoHud = document.getElementById("demo-hud");
  const demoHandle = document.getElementById("demo-hud-handle");
  if (demoHud && demoHandle) {
    let isDragging = false, startX, startY, initLeft, initTop;
    demoHandle.addEventListener("mousedown", (e) => {
      isDragging = true; startX = e.clientX; startY = e.clientY;
      const rect = demoHud.getBoundingClientRect();
      initLeft = rect.left; initTop = rect.top;
      demoHud.style.position = "fixed";
      demoHud.style.left = initLeft + "px";
      demoHud.style.top = initTop + "px";
      demoHud.style.cursor = "grabbing";
    });
    window.addEventListener("mousemove", (e) => {
      if (!isDragging) return;
      demoHud.style.left = (initLeft + (e.clientX - startX)) + "px";
      demoHud.style.top = (initTop + (e.clientY - startY)) + "px";
    });
    window.addEventListener("mouseup", () => { isDragging = false; if (demoHud) demoHud.style.cursor = "grab"; });
  }

  /* ── Sandbox synthetic streams ── */
  function createSlideStream(canvasId, videoId, persona) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const video = document.getElementById(videoId);
    if (!video) return;
    let t = Math.random() * 100;

    function render() {
      t += 0.045;
      const w = canvas.width, h = canvas.height, cx = w / 2, cy = h / 2;
      ctx.fillStyle = persona === "bob" ? "#100a1a" : "#080b14";
      ctx.fillRect(0, 0, w, h);

      const isStatic = (persona === "charlie") || simAnomalies.freeze;
      const isDeepfake = (persona === "bob") || simAnomalies.jitter;
      const swayX = isStatic ? 0 : Math.sin(t * 0.8) * 6;
      const swayY = isStatic ? 0 : Math.cos(t * 0.9) * 3;
      const jx = isDeepfake ? (Math.random() - 0.5) * 5 : 0;
      const jy = isDeepfake ? (Math.random() - 0.5) * 5 : 0;
      const fx = cx + swayX + jx, fy = cy + swayY + jy;

      ctx.beginPath(); ctx.ellipse(fx, fy + 140, 110, 70, 0, 0, Math.PI * 2); ctx.fillStyle = "#181d2c"; ctx.fill();
      ctx.fillStyle = "#c59b7b"; ctx.fillRect(fx - 22, fy + 50, 44, 45);
      ctx.beginPath(); ctx.ellipse(fx, fy, 65, 85, 0, 0, Math.PI * 2); ctx.fillStyle = persona === "bob" ? "#c48b71" : "#e0ac69"; ctx.fill();

      const blink = isStatic ? false : (Math.sin(t * 1.7) > 0.93);
      const eh = blink ? 2 : 9;
      ctx.beginPath(); ctx.ellipse(fx - 24, fy - 10, 11, eh, 0, 0, Math.PI * 2); ctx.ellipse(fx + 24, fy - 10, 11, eh, 0, 0, Math.PI * 2); ctx.fillStyle = "#ffffff"; ctx.fill();
      ctx.beginPath(); ctx.arc(fx - 24, fy - 10, blink ? 1 : 4, 0, Math.PI * 2); ctx.arc(fx + 24, fy - 10, blink ? 1 : 4, 0, Math.PI * 2); ctx.fillStyle = "#1e293b"; ctx.fill();

      const mo = isStatic ? 0 : Math.abs(Math.sin(t * 2.5)) * 10;
      ctx.beginPath(); ctx.ellipse(fx, fy + 36, 16, 3 + mo, 0, 0, Math.PI * 2); ctx.fillStyle = "#833838"; ctx.fill();

      if (isDeepfake) {
        ctx.fillStyle = "rgba(242,104,92,0.08)"; ctx.fillRect(fx - 50, fy - 45, 100, 95);
        ctx.strokeStyle = "rgba(242,104,92,0.3)"; ctx.lineWidth = 1; ctx.strokeRect(fx - 50, fy - 45, 100, 95);
      }
      requestAnimationFrame(render);
    }
    render();
    try { video.srcObject = canvas.captureStream(30); } catch (_) { }
  }

  async function toggleWebcam() {
    const btn = document.getElementById("ob-btn-camera");
    const v1 = document.getElementById("ob-v1");
    if (!v1) return;
    if (activeWebcamStream) {
      activeWebcamStream.getTracks().forEach((t) => t.stop());
      activeWebcamStream = null;
      btn.textContent = "Enable live webcam";
      createSlideStream("ob-c1", "ob-v1", "alice");
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 360 }, audio: false });
        activeWebcamStream = stream;
        v1.srcObject = stream;
        btn.textContent = "Switch to simulated feed";
      } catch (err) {
        alert("Camera permission denied or camera not available.");
      }
    }
  }

  function setupThreatBtn(id, key) {
    const b = document.getElementById(id);
    if (!b) return;
    b.addEventListener("click", () => {
      simAnomalies[key] = !simAnomalies[key];
      b.classList.toggle("active", simAnomalies[key]);
    });
  }

  function probeEngine() {
    const dot = document.getElementById("conn-dot");
    const label = document.getElementById("conn-status-label");
    if (!dot || !label) return;
    let targetUrl = "wss://netram-deepfake-detection.up.railway.app";

    function testSocket(url) {
      try {
        const ws = new WebSocket(url);
        ws.onopen = () => { dot.className = "conn-status-dot online"; label.textContent = "Cloud engine connected & ready"; try { ws.close(); } catch (_) { } };
        ws.onerror = () => { dot.className = "conn-status-dot"; label.textContent = "Cloud engine starting up — please wait a moment and retry"; };
      } catch (_) { dot.className = "conn-status-dot"; label.textContent = "Could not reach cloud engine — check your internet connection"; }
    }
    if (hasChrome) chrome.storage.local.get(["serverUrl"], (res) => testSocket(res.serverUrl?.trim() || targetUrl));
    else testSocket(targetUrl);
  }

  createSlideStream("ob-c1", "ob-v1", "alice");
  createSlideStream("ob-c2", "ob-v2", "bob");
  createSlideStream("ob-c3", "ob-v3", "charlie");
  createSlideStream("ob-c4", "ob-v4", "david");

  const camBtn = document.getElementById("ob-btn-camera");
  if (camBtn) camBtn.addEventListener("click", toggleWebcam);

  setupThreatBtn("ob-sim-jitter", "jitter");
  setupThreatBtn("ob-sim-freeze", "freeze");
  setupThreatBtn("ob-sim-desync", "desync");

  probeEngine();
  setInterval(probeEngine, 3000);
})();
