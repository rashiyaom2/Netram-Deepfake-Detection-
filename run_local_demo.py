"""
===============================================================================
🛡️ NETRAM AI — LOCAL INFERENCE ENGINE LAUNCHER
===============================================================================
Starts the local Netram AI Multi-Modal Deepfake & Presentation Attack Engine
on ws://127.0.0.1:8765.

Runs purely in the terminal — does not open any browser windows or demo HTML pages.
The Chrome Extension (or any external client) connects directly to:
  ws://127.0.0.1:8765

Usage:
  python run_local_demo.py
===============================================================================
"""
import sys
import logging
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from extension_server import run_server

# ANSI Color codes for modern terminal output
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("NetramEngine")


def main():
    print(f"\n{CYAN}{BOLD}" + "=" * 70 + f"{RESET}")
    print(f"{CYAN}{BOLD}   [NETRAM AI] -- LOCAL INFERENCE ENGINE{RESET}")
    print(f"{CYAN}{BOLD}" + "=" * 70 + f"{RESET}\n")

    print(f"  {BOLD}• WebSocket Endpoint:{RESET}    {GREEN}ws://127.0.0.1:8765{RESET}")
    print(f"  {BOLD}• HTTP Health Check:{RESET}     {GREEN}http://127.0.0.1:8765/health{RESET}")
    print(f"  {BOLD}• Mode:{RESET}                  {CYAN}100% Offline / Local Neural Processing{RESET}")
    print(f"  {BOLD}• Active Detectors:{RESET}      Spatial ViT · 2D DCT CNN · Bi-GRU · Phone Replay · AR Filters · AASIST Audio")
    print(f"\n{YELLOW}Engine is listening for Chrome Extension connections... (Press Ctrl+C to stop){RESET}\n")

    try:
        run_server(host="127.0.0.1", port=8765)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Netram AI local inference engine stopped.{RESET}\n")


if __name__ == "__main__":
    main()
