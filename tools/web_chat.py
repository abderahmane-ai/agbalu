#!/usr/bin/env python3
"""AƔBALU: Lightweight Web Chat UI & Reverse Proxy Server for Jugurtha-4B.

Serves the modern chat interface on port 8080 and transparently proxies /v1/*
requests directly to the vLLM backend (with full SSE streaming support).

Usage:
    # Run directly on HPC (bridges to local vLLM on port 8000):
    python tools/web_chat.py

    # Or run locally on your Mac pointing to HPC Tailscale IP:
    python tools/web_chat.py --vllm-url http://100.98.139.97:8000/v1 --port 3000
"""

from __future__ import annotations

import argparse
import http.server
import socketserver
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_PORT = 8080
DEFAULT_VLLM_URL = "http://127.0.0.1:8000/v1"
STATIC_DIR = Path(__file__).resolve().parent / "chat_ui"


class JugurthaHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler serving the static UI and proxying /v1/* endpoints to vLLM."""

    def __init__(self, *args: object, vllm_url: str = DEFAULT_VLLM_URL, **kwargs: object) -> None:
        self.vllm_url = vllm_url.rstrip("/")
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        sys.stderr.write(f"[WebChat] {self.address_string()} - {format % args}\n")

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, X-Requested-With",
        )
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.startswith("/v1/"):
            self.proxy_request("GET")
        else:
            super().do_GET()

    def do_POST(self) -> None:
        if self.path.startswith("/v1/"):
            self.proxy_request("POST")
        else:
            self.send_error(404, "Not Found")

    def proxy_request(self, method: str) -> None:
        target_path = self.path[len("/v1") :]
        target_url = f"{self.vllm_url}{target_path}"

        parsed = urllib.parse.urlparse(target_url)
        if parsed.scheme not in ("http", "https"):
            self.send_error(400, "Invalid URL scheme")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in ("host", "content-length")
        }

        req = urllib.request.Request(  # noqa: S310
            target_url,
            data=body,
            headers=headers,
            method=method,
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
                self.send_response(resp.status)
                for header, val in resp.headers.items():
                    hl = header.lower()
                    if hl not in (
                        "transfer-encoding",
                        "content-length",
                        "access-control-allow-origin",
                    ):
                        self.send_header(header, val)
                self.end_headers()

                while True:
                    chunk = resp.read(512)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()

        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for header, val in e.headers.items():
                if header.lower() not in ("transfer-encoding", "content-length"):
                    self.send_header(header, val)
            self.end_headers()
            self.wfile.write(e.read())
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            err_msg = (
                f'{{"error": {{"message": "Failed to connect to vLLM at {self.vllm_url}: {e}", '
                f'"type": "bad_gateway", "code": 502}}}}'
            )
            self.wfile.write(err_msg.encode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jugurtha-4B Web Chat & API Proxy")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to serve UI on")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")  # noqa: S104
    parser.add_argument(
        "--vllm-url",
        default=DEFAULT_VLLM_URL,
        help="Target vLLM /v1 base URL",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not STATIC_DIR.exists():
        sys.stderr.write(f"ERROR: UI directory not found at {STATIC_DIR}\n")
        sys.exit(1)

    class CustomServer(socketserver.TCPServer):
        allow_reuse_address = True

    class CustomHandler(JugurthaHTTPHandler):
        def __init__(self, *h_args: object, **h_kwargs: object) -> None:
            super().__init__(*h_args, vllm_url=args.vllm_url, **h_kwargs)

    with CustomServer((args.host, args.port), CustomHandler) as httpd:
        sys.stdout.write("=" * 66 + "\n")
        sys.stdout.write("  AƔBALU: Jugurtha-4B Web Chat Interface & Proxy\n")
        sys.stdout.write("=" * 66 + "\n")
        sys.stdout.write(f"  Web Interface:  http://{args.host}:{args.port}\n")
        sys.stdout.write(f"  vLLM Proxy:     http://{args.host}:{args.port}/v1\n")
        sys.stdout.write(f"  Target Backend: {args.vllm_url}\n")
        sys.stdout.write(f"  Serving Dir:    {STATIC_DIR}\n")
        sys.stdout.write("=" * 66 + "\n")
        sys.stdout.write("Press Ctrl+C to stop the web server.\n\n")
        sys.stdout.flush()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            sys.stdout.write("\nShutting down web server...\n")


if __name__ == "__main__":
    main()
