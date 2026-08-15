"""
Netram AI Backend Entry Point Alias (for Python WSGI/ASGI/Railpack detection)
"""
from main import *

if __name__ == "__main__":
    import os
    from extension_server import run_server
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8765"))
    run_server(host=host, port=port)
