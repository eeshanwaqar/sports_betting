"""
Run API Script - Start the FastAPI prediction server.

Usage:
    python scripts/run_api.py                  # Default: 0.0.0.0:8000
    python scripts/run_api.py --port 8080      # Custom port
    python scripts/run_api.py --reload          # Auto-reload for development
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import uvicorn
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("scripts.run_api")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="EPL Predictor API Server")
    parser.add_argument("--host", type=str, default=None, help="Bind host")
    parser.add_argument("--port", type=int, default=None, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    return parser.parse_args()


def main() -> None:
    """Start the API server."""
    args = parse_args()
    config = load_config(args.config)

    host = args.host or config.api.host
    port = args.port or config.api.port
    reload = args.reload or config.api.debug

    print("=" * 60)
    print("EPL BETTING PREDICTOR - API SERVER")
    print("=" * 60)
    print(f"  Host     : {host}")
    print(f"  Port     : {port}")
    print(f"  Reload   : {reload}")
    print(f"  Docs     : http://localhost:{port}/docs")
    print("=" * 60)

    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
