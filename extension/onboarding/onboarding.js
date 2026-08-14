/**
 * Netram AI — Onboarding Logic (5-Slide Enhanced Flow)
 * Interactive carousel, mini-HUD drag simulation, live testbed video streams, and engine probe.
 */
(() => {
  let currentSlide = 0;
  const slides = document.querySelectorAll(".slide");
  const dots = document.querySelectorAll(".step-dot");

  let activeWebcamStream = null;
  const simAnomalies = {
    jitter: false,
    freeze: false,
    desync: false,
  };

  function goToSlide(idx) {
    if (idx < 0 || idx >= slides.length) return;
    slides.forEach((s, i) => {
      s.classList.remove("active", "prev");
      if (i === idx) s.classList.add("active");
      else if (i < idx) s.classList.add("prev");
    });
    dots.forEach((d, i) => {
      d.classList.toggle("active", i === idx);
    });
    currentSlide = idx;
  }

  // Next / Prev button listeners
  document.querySelectorAll(".next-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const nextIdx = parseInt(btn.getAttribute("data-next"), 10);
      goToSlide(nextIdx);
    });
  });

  document.querySelectorAll(".prev-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const prevIdx = parseInt(btn.getAttribute("data-prev"), 10);
      goToSlide(prevIdx);
    });
  });

  // Step dot click listeners
  dots.forEach((dot) => {
    dot.addEventListener("click", () => {
      const idx = parseInt(dot.getAttribute("data-slide"), 10);
      goToSlide(idx);
    });
  });

  // Skip & Launch Meet buttons
  const skipBtn = document.getElementById("btn-skip");
  if (skipBtn) {
    skipBtn.addEventListener("click", () => {
      goToSlide(4);
    });
  }

  const launchBtn = document.getElementById("btn-launch-meet");
  if (launchBtn) {
    launchBtn.addEventListener("click", () => {
      window.open("https://meet.google.com/new", "_blank");
    });
  }

  const openTestbedBtn = document.getElementById("btn-open-testbed");
  if (openTestbedBtn) {
    openTestbedBtn.addEventListener("click", () => {
      if (chrome.runtime?.getURL) {
        window.open(chrome.runtime.getURL("test_meeting.html"), "_blank");
      } else {
        window.open("/test_meeting.html", "_blank");
      }
    });
  }

  // ── Interactive Mini HUD Drag Simulation in Slide 2 ──
  const demoHud = document.getElementById("demo-hud");
  const demoHandle = document.getElementById("demo-hud-handle");
  if (demoHud && demoHandle) {
    let isDragging = false;
    let startX, startY, initLeft, initTop;

    demoHandle.addEventListener("mousedown", (e) => {
      isDragging = true;
      startX = e.clientX;
      startY = e.clientY;
      const rect = demoHud.getBoundingClientRect();
      initLeft = rect.left;
      initTop = rect.top;
      demoHud.style.position = "absolute";
      demoHud.style.left = initLeft + "px";
      demoHud.style.top = initTop + "px";
      demoHud.style.cursor = "grabbing";
    });

    window.addEventListener("mousemove", (e) => {
      if (!isDragging) return;
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;
      demoHud.style.left = (initLeft + dx) + "px";
      demoHud.style.top = (initTop + dy) + "px";
    });

    window.addEventListener("mouseup", () => {
      isDragging = false;
      if (demoHud) demoHud.style.cursor = "grab";
    });
  }

  // ── Slide 3: Synthetic Photorealistic Video Grid Generator ──
  function createSlideStream(canvasId, videoId, persona) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const video = document.getElementById(videoId);
    if (!video) return;

    let t = Math.random() * 100;

    function render() {
      t += 0.045;
      const w = canvas.width, h = canvas.height;
      const cx = w / 2, cy = h / 2;

      // Dark background
      ctx.fillStyle = persona === 'bob' ? '#130c1e' : '#0a0f1d';
      ctx.fillRect(0, 0, w, h);

      const isStatic = (persona === 'charlie') || simAnomalies.freeze;
      const isDeepfake = (persona === 'bob') || simAnomalies.jitter;

      const swayX = isStatic ? 0 : Math.sin(t * 0.8) * 6;
      const swayY = isStatic ? 0 : Math.cos(t * 0.9) * 3;
      const jx = isDeepfake ? (Math.random() - 0.5) * 5 : 0;
      const jy = isDeepfake ? (Math.random() - 0.5) * 5 : 0;

      const fx = cx + swayX + jx;
      const fy = cy + swayY + jy;

      // Shoulders
      ctx.beginPath();
      ctx.ellipse(fx, fy + 140, 110, 70, 0, 0, Math.PI * 2);
      ctx.fillStyle = '#1e293b';
      ctx.fill();

      // Neck
      ctx.fillStyle = '#c59b7b';
      ctx.fillRect(fx - 22, fy + 50, 44, 45);

      // Head
      ctx.beginPath();
      ctx.ellipse(fx, fy, 65, 85, 0, 0, Math.PI * 2);
      ctx.fillStyle = persona === 'bob' ? '#c48b71' : '#e0ac69';
      ctx.fill();

      // Eyes & Blinking
      const blink = isStatic ? false : (Math.sin(t * 1.7) > 0.93);
      const eh = blink ? 2 : 9;

      ctx.beginPath();
      ctx.ellipse(fx - 24, fy - 10, 11, eh, 0, 0, Math.PI * 2);
      ctx.ellipse(fx + 24, fy - 10, 11, eh, 0, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();

      ctx.beginPath();
      ctx.arc(fx - 24, fy - 10, blink ? 1 : 4, 0, Math.PI * 2);
      ctx.arc(fx + 24, fy - 10, blink ? 1 : 4, 0, Math.PI * 2);
      ctx.fillStyle = '#1e293b';
      ctx.fill();

      // Mouth
      const mo = isStatic ? 0 : Math.abs(Math.sin(t * 2.5)) * 10;
      ctx.beginPath();
      ctx.ellipse(fx, fy + 36, 16, 3 + mo, 0, 0, Math.PI * 2);
      ctx.fillStyle = '#833838';
      ctx.fill();

      // Deepfake Seam Box & Noise Matrix
      if (isDeepfake) {
        ctx.fillStyle = 'rgba(239, 68, 68, 0.08)';
        ctx.fillRect(fx - 50, fy - 45, 100, 95);
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.25)';
        ctx.lineWidth = 1;
        ctx.strokeRect(fx - 50, fy - 45, 100, 95);
      }

      requestAnimationFrame(render);
    }

    render();
    try {
      const stream = canvas.captureStream(30);
      video.srcObject = stream;
    } catch (_) {}
  }

  // Toggle real webcam in Sandbox
  async function toggleWebcam() {
    const btn = document.getElementById("ob-btn-camera");
    const v1 = document.getElementById("ob-v1");
    if (!v1) return;

    if (activeWebcamStream) {
      activeWebcamStream.getTracks().forEach(t => t.stop());
      activeWebcamStream = null;
      btn.textContent = "📷 Enable Live Webcam";
      createSlideStream("ob-c1", "ob-v1", "alice");
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 480, height: 360 },
          audio: false
        });
        activeWebcamStream = stream;
        v1.srcObject = stream;
        btn.textContent = "Switch to Simulated Feed";
      } catch (err) {
        alert("Camera permission denied or camera not available.");
      }
    }
  }

  // Threat Simulator Buttons in Slide 3
  function setupThreatBtn(id, key) {
    const b = document.getElementById(id);
    if (!b) return;
    b.addEventListener("click", () => {
      simAnomalies[key] = !simAnomalies[key];
      b.classList.toggle("active", simAnomalies[key]);
    });
  }

  // ── Slide 4: Live WebSocket Connection Probe ──
  function probeEngine() {
    const dot = document.getElementById("conn-dot");
    const label = document.getElementById("conn-status-label");
    if (!dot || !label) return;

    let targetUrl = "ws://127.0.0.1:8765";
    if (chrome.storage?.local) {
      chrome.storage.local.get(["serverUrl"], (res) => {
        if (res.serverUrl && res.serverUrl.trim()) targetUrl = res.serverUrl.trim();
        testSocket(targetUrl);
      });
    } else {
      testSocket(targetUrl);
    }

    function testSocket(url) {
      try {
        const ws = new WebSocket(url);
        ws.onopen = () => {
          dot.className = "conn-status-dot online";
          label.textContent = `🟢 Engine Connected & Ready (${url})`;
          try { ws.close(); } catch (_) {}
        };
        ws.onerror = () => {
          dot.className = "conn-status-dot";
          label.textContent = `⚪ Engine Offline (${url}) — Run 'start_server.bat' to activate`;
        };
      } catch (_) {
        dot.className = "conn-status-dot";
        label.textContent = `⚪ Engine Offline — Run 'start_server.bat' to activate`;
      }
    }
  }

  // Initialize Slide 3 streams
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
