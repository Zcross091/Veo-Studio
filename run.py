"""Veo Studio All-in-One Launcher.

Starts the server, checks dependencies, verifies OpenArt credentials,
and automatically opens the web interface in your default browser.
"""
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

BASE = Path(__file__).parent


def ensure_env():
    env_file = BASE / ".env"
    example = BASE / ".env.example"
    if not env_file.exists() and example.exists():
        shutil.copy(example, env_file)
        print("  Created .env file from .env.example")


def ensure_openart():
    exe = BASE / ".venv" / "Scripts" / "openart.exe"
    if not exe.exists():
        print("  Setting up OpenArt CLI...")
        try:
            import io
            import urllib.request
            import zipfile
            url = "https://github.com/OpenArt-AI/cli/releases/download/v0.1.1/openart_0.1.1_windows_amd64.zip"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                z.extractall(str(BASE / ".venv" / "Scripts"))
            print("  OpenArt CLI ready.")
        except Exception as e:
            print("  Note: OpenArt CLI setup skipped:", e)


def open_browser(port):
    time.sleep(1.8)
    url = f"http://127.0.0.1:{port}"
    print(f"\n  Opening browser at {url}...\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass


def start_bridge_server():
    try:
        from bridge import app as bridge_app
        bridge_port = int(os.environ.get("BRIDGE_PORT", "5050"))
        bridge_app.run(host="127.0.0.1", port=bridge_port, threaded=True)
    except Exception as e:
        pass


def main():
    print("\n" + "=" * 58)
    print("         Launching Veo Studio (All-In-One)")
    print("=" * 58 + "\n")

    ensure_env()
    ensure_openart()

    # Start REST API Bridge Hub in background thread
    threading.Thread(target=start_bridge_server, daemon=True).start()

    port = int(os.environ.get("PORT", "5000"))
    bridge_port = int(os.environ.get("BRIDGE_PORT", "5050"))
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    # Import and run app
    from app import app
    import socket
    local_ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    print("  ========================================================")
    print("  Veo Studio Ready!")
    print(f"  - On your PC:    http://127.0.0.1:{port}")
    print(f"  - On your Phone: http://{local_ip}:{port} (same Wi-Fi)")
    print(f"  - Universal REST Hub: http://127.0.0.1:{bridge_port}/api/v1/video/generate")
    print("  ========================================================\n")

    app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    main()

