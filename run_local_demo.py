"""
===============================================================================
🛡️ NETRAM AI — ZERO-DEPENDENCY OFFLINE LOCAL DEMO LAUNCHER
===============================================================================
Launches the full Netram AI Deepfake & Presentation Attack Detection System
locally without any internet or cloud connection required:

1. Starts the Netram AI Multi-Modal Inference Engine (WebSocket on 127.0.0.1:8765)
2. Starts the Local Web Demo Server (HTTP on 127.0.0.1:3000)
3. Automatically opens your default web browser to the interactive meeting room:
   http://127.0.0.1:3000/test_meeting.html

Includes:
- Multi-Participant Simulation (Alice authentic, Bob deepfake, Charlie frozen, David verified)
- Handheld Smartphone & Display Screen Replay Spoof Detector (Live toggle)
- Snapchat / Instagram / AR Beauty Filter Detector (Live toggle)
- Real-time Audio AASIST Voice Cloning Detector
- Automated In-Call Participant Legal Notice Broadcaster
- Draggable Siri-styled Neural Forensics HUD

Usage:
  python run_local_demo.py
===============================================================================
"""
import sys
import os
import time
import threading
import webbrowser
import http.server
import socketserver
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ANSI Color codes for modern terminal output
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("NetramLocalDemo")


class LocalDemoHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the project root directory with proper CORS and MIME types."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def log_message(self, format, *args):
        # Suppress noisy HTTP asset requests
        pass


def run_http_server(host="127.0.0.1", port=3000):
    """Runs a local HTTP server for test_meeting.html and extension assets."""
    # Find available port if 3000 is taken
    for p in range(port, port + 20):
        try:
            httpd = socketserver.TCPServer((host, p), LocalDemoHTTPHandler)
            logger.info(f"🌐 Local Web Server listening on http://{host}:{p}")
            return httpd, p
        except OSError:
            continue
    raise RuntimeError("Could not find open port for HTTP server (ports 3000-3020 in use).")


def run_websocket_server(host="127.0.0.1", port=8765):
    """Runs the full Netram AI extension WebSocket server in an asyncio event loop."""
    import asyncio
    import websockets
    from extension_server import handle_client

    async def main_ws():
        async with websockets.serve(handle_client, host, port, max_size=10 * 1024 * 1024):
            logger.info(f"⚡ Netram Inference WebSocket Server listening on ws://{host}:{port}")
            await asyncio.Future()  # run forever

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main_ws())
    except Exception as e:
        logger.error(f"WebSocket engine encountered error: {e}")


def main():
    print(f"\n{CYAN}{BOLD}" + "=" * 76 + f"{RESET}")
    print(f"{CYAN}{BOLD}  🛡️  NETRAM AI — REAL-TIME OFFLINE DEMO LAUNCHER  🛡️{RESET}")
    print(f"{CYAN}{BOLD}" + "=" * 76 + f"{RESET}\n")

    print(f"{BOLD}[1/3]{RESET} Initializing Netram AI Multi-Modal Inference Engine...")
    ws_thread = threading.Thread(
        target=run_websocket_server,
        args=("127.0.0.1", 8765),
        daemon=True,
        name="NetramInferenceEngine"
    )
    ws_thread.start()
    time.sleep(1.0)

    print(f"{BOLD}[2/3]{RESET} Initializing Local Web Server...")
    httpd, actual_port = run_http_server("127.0.0.1", 3000)
    http_thread = threading.Thread(
        target=httpd.serve_forever,
        daemon=True,
        name="NetramHTTPServer"
    )
    http_thread.start()

    demo_url = f"http://127.0.0.1:{actual_port}/test_meeting.html"
    legal_url = f"http://127.0.0.1:{actual_port}/netram_ai_extension/legal/terms.html"

    print(f"{BOLD}[3/3]{RESET} Launching Interactive Test Meeting in Browser...")
    time.sleep(0.5)
    webbrowser.open(demo_url)

    print(f"\n{GREEN}{BOLD}✨ NETRAM AI LOCAL DEMO IS FULLY LIVE & OPERATIONAL! ✨{RESET}\n")
    print(f"  {BOLD}• Meeting Testbed URL:{RESET}  {CYAN}{demo_url}{RESET}")
    print(f"  {BOLD}• Legal & Privacy Portal:{RESET} {CYAN}{legal_url}{RESET}")
    print(f"  {BOLD}• WebSocket Endpoint:{RESET}    {CYAN}ws://127.0.0.1:8765{RESET}")
    print(f"  {BOLD}• Mode:{RESET}                  {GREEN}100% Offline / Local Inference (Zero Internet Needed){RESET}")
    print()
    print(f"{YELLOW}{BOLD}🎮 Live Interactive Demo Controls in Browser:{RESET}")
    print(f"  {BOLD}[📱 Phone Replay]{RESET}     Hold simulated smartphone in front of camera (detects bezel & moiré)")
    print(f"  {BOLD}[✨ Snapchat Filter]{RESET}  Applies porcelain skin airbrushing, cheek blush & eye warping")
    print(f"  {BOLD}[Mesh Jitter]{RESET}        Simulates GAN boundary distortion on Bob")
    print(f"  {BOLD}[Static Freeze]{RESET}      Simulates presentation freeze on Charlie")
    print(f"  {BOLD}[Lip Desync]{RESET}         Simulates audio-visual dub desynchronization")
    print(f"  {BOLD}[Audio Spoof]{RESET}        Simulates TTS voice cloning (AASIST flagged)")
    print(f"  {BOLD}[📢 Post Legal Notice]{RESET} Broadcasts enterprise non-misuse disclaimer to meeting chat")
    print(f"\n{CYAN}Press {BOLD}Ctrl+C{RESET}{CYAN} in this terminal to shut down the local demo servers anytime.{RESET}\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Stopping Netram AI local demo servers...{RESET}")
        httpd.shutdown()
        print(f"{GREEN}✓ Netram AI local servers stopped safely.{RESET}\n")


if __name__ == "__main__":
    main()
