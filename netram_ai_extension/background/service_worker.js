/**
 * Netram AI Deepfake Shield - Background Service Worker (Manifest V3)
 * Provides privileged WebSocket proxying for HTTPS pages (bypasses Chrome mixed-content restrictions),
 * state persistence, and badge status updates.
 */

let ws = null;
let currentServerUrl = "ws://127.0.0.1:8765";
let activePorts = new Set();
let lastVerdicts = new Map();
let isConnecting = false;

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

// Load saved server URL
chrome.storage.local.get(["serverUrl"], (res) => {
  if (res.serverUrl) {
    currentServerUrl = normalizeWsUrl(res.serverUrl);
  }
  connectWebSocket();
});

chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.serverUrl && changes.serverUrl.newValue) {
    currentServerUrl = normalizeWsUrl(changes.serverUrl.newValue);
    if (ws) {
      try { ws.close(); } catch (_) {}
      ws = null;
    }
    connectWebSocket();
  }
});

function connectWebSocket() {
  if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) return;
  if (isConnecting) return;
  isConnecting = true;

  try {
    ws = new WebSocket(currentServerUrl);
    ws.onopen = () => {
      isConnecting = false;
      broadcast({ type: "WS_STATUS", connected: true, serverUrl: currentServerUrl });
    };
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.participant_id) {
          lastVerdicts.set(data.participant_id, data);
        }
        broadcast(data);
      } catch (_) {}
    };
    ws.onclose = () => {
      isConnecting = false;
      ws = null;
      broadcast({ type: "WS_STATUS", connected: false, serverUrl: currentServerUrl });
      setTimeout(connectWebSocket, 2000);
    };
    ws.onerror = () => {
      isConnecting = false;
      try { ws.close(); } catch (_) {}
      ws = null;
      broadcast({ type: "WS_STATUS", connected: false, serverUrl: currentServerUrl });
    };
  } catch (e) {
    isConnecting = false;
    ws = null;
    setTimeout(connectWebSocket, 2000);
  }
}

function broadcast(msg) {
  for (const port of activePorts) {
    try {
      port.postMessage(msg);
    } catch (_) {
      activePorts.delete(port);
    }
  }
}

// Handle long-lived connection ports from content script and popup
chrome.runtime.onConnect.addListener((port) => {
  if (port.name === "netram_tunnel") {
    activePorts.add(port);

    // Send immediate initial status
    const isConnected = ws !== null && ws.readyState === WebSocket.OPEN;
    port.postMessage({ type: "WS_STATUS", connected: isConnected, serverUrl: currentServerUrl });

    // Send any cached verdicts
    for (const verdict of lastVerdicts.values()) {
      try { port.postMessage(verdict); } catch (_) {}
    }

    port.onMessage.addListener((msg) => {
      if (msg.type === "SEND_FRAME" || msg.type === "frame") {
        const payload = msg.payload || msg;
        if (ws && ws.readyState === WebSocket.OPEN) {
          try {
            ws.send(JSON.stringify(payload));
          } catch (_) {}
        } else if (!ws || ws.readyState === WebSocket.CLOSED) {
          connectWebSocket();
        }
      } else if (msg.type === "SWITCH_SERVER") {
        if (msg.serverUrl) {
          currentServerUrl = normalizeWsUrl(msg.serverUrl);
          chrome.storage.local.set({ serverUrl: currentServerUrl });
          if (ws) {
            try { ws.close(); } catch (_) {}
            ws = null;
          }
          connectWebSocket();
        }
      } else if (msg.type === "RESET_PARTICIPANT") {
        if (ws && ws.readyState === WebSocket.OPEN) {
          try {
            ws.send(JSON.stringify({ type: "reset_participant", participant_id: msg.participant_id }));
          } catch (_) {}
        }
      }
    });

    port.onDisconnect.addListener(() => {
      activePorts.delete(port);
    });
  }
});

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.set({
    overlayEnabled: true,
    audioAlertEnabled: true,
    serverUrl: "ws://127.0.0.1:8765",
  });
  connectWebSocket();
});
