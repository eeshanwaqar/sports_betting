"""
Run Frontend - Serve the web frontend on port 3000.

A lightweight HTTP server that serves the web/ directory.
The frontend calls the FastAPI backend on port 8000.

Usage:
    python scripts/run_frontend.py              # Default: localhost:3000
    python scripts/run_frontend.py --port 3001  # Custom port
"""

import argparse
import os
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial

# Resolve paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"


class FrontendHandler(SimpleHTTPRequestHandler):
    """
    Custom handler that serves the SPA.

    - /static/* -> web/static/*
    - Everything else -> web/templates/index.html (SPA routing)
    """

    def __init__(self, *args, web_dir: Path, **kwargs):
        self.web_dir = web_dir
        super().__init__(*args, directory=str(web_dir), **kwargs)

    def do_GET(self):
        # Serve static files normally
        if self.path.startswith("/static/"):
            super().do_GET()
            return

        # For everything else, serve index.html (SPA client-side routing)
        self.path = "/templates/index.html"
        super().do_GET()

    def log_message(self, format, *args):
        """Cleaner log format. Safely handles Unicode in request logs (Windows cp1252)."""
        msg = f"  [frontend] {self.address_string()} - {format % args}\n"
        try:
            sys.stdout.write(msg)
        except UnicodeEncodeError:
            safe_msg = msg.encode("ascii", errors="replace").decode("ascii")
            sys.stdout.write(safe_msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EPL Predictor Frontend Server")
    parser.add_argument("--port", type=int, default=3000, help="Port (default: 3000)")
    parser.add_argument("--host", type=str, default="localhost", help="Host (default: localhost)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not WEB_DIR.exists():
        print(f"ERROR: Web directory not found: {WEB_DIR}")
        sys.exit(1)

    handler = partial(FrontendHandler, web_dir=WEB_DIR)
    server = HTTPServer((args.host, args.port), handler)

    print("=" * 60)
    print("EPL BETTING PREDICTOR - FRONTEND SERVER")
    print("=" * 60)
    print(f"  URL      : http://{args.host}:{args.port}")
    print(f"  Serving  : {WEB_DIR}")
    print(f"  API      : http://localhost:8000")
    print(f"  API Docs : http://localhost:8000/docs")
    print("=" * 60)
    print("\n  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Frontend server stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
