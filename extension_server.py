"""
Native High-Throughput WebSocket Deepfake Inference Server.
Mirrors run_local_demo.py for backward compatibility with Docker, Procfile, and tests.
"""
from run_local_demo import (
    DeepfakeExtensionServer,
    run_server,
    main,
)

if __name__ == "__main__":
    main()
