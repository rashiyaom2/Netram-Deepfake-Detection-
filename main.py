"""
Netram AI Deepfake Detection Backend — Production Entry Point (Railway / Cloud Deployment)

Railway Railpack & Nixpacks automatically execute main.py on startup.
Listens on 0.0.0.0:$PORT and supports:
1. HTTP GET / and /health (Railway health checks & uptime monitors)
2. WebSocket wss:// connections (Real-time Chrome Extension frame streaming)
"""
import os
import sys
import logging
from extension_server import DeepfakeExtensionServer, run_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("railway_main")

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8765"))
    
    logger.info("=" * 60)
    logger.info("🛡️  NETRAM AI — REAL-TIME DEEPFAKE DETECTION BACKEND")
    logger.info(f"⚡ Binding on Host: {host} | Port: {port}")
    logger.info("📡 Dual-Protocol: HTTP Health Checks (/health) + WebSocket (/ for Extension)")
    logger.info("=" * 60)
    
    run_server(host=host, port=port)
