"""Unified Video Provider Bridge for Veo Studio.

Exposes a SINGLE API endpoint to generate videos using:
- Kling AI (Text-to-Video & Image-to-Video)
- Luma Dream Machine (Ray / Generations API)
- Hailuo / MiniMax (Video-01)

Features:
- Single unified submission endpoint: POST /api/v1/video/generate
- Single unified polling endpoint: GET /api/v1/video/status/<task_id>
- Provider 'auto' mode that tries the best available provider
- Seamless integration with Veo Studio's multi-scene pipeline
"""
import base64
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from flask import Flask, jsonify, request

app = Flask(__name__)

TASKS = {}


# ---------- Environment & Configuration ----------
def get_env_credentials():
    return {
        "kling_ak": os.environ.get("KLING_ACCESS_KEY") or os.environ.get("KLING_AK"),
        "kling_sk": os.environ.get("KLING_SECRET_KEY") or os.environ.get("KLING_SK"),
        "kling_token": os.environ.get("KLING_TOKEN") or os.environ.get("KLING_COOKIE"),
        "luma_key": os.environ.get("LUMA_API_KEY") or os.environ.get("LUMA_KEY") or os.environ.get("LUMA_TOKEN"),
        "hailuo_key": os.environ.get("MINIMAX_API_KEY") or os.environ.get("HAILUO_API_KEY") or os.environ.get("HAILUO_TOKEN"),
    }


# ---------- 1. Kling AI Adapter ----------
def kling_create_task(prompt, aspect="16:9", duration=5, reference_image=None, creds=None):
    """Submit a task to Kling AI (Official API or authenticated session)."""
    token = (creds or {}).get("kling_token")
    ak = (creds or {}).get("kling_ak")
    sk = (creds or {}).get("kling_sk")

    auth_header = None
    if token:
        auth_header = f"Bearer {token}" if not token.startswith("Bearer ") else token
    elif ak and sk:
        auth_header = f"Basic {base64.b64encode(f'{ak}:{sk}'.encode()).decode()}"
    else:
        raise ValueError("Missing Kling credentials. Set KLING_ACCESS_KEY & KLING_SECRET_KEY or KLING_TOKEN in .env.")

    # Model endpoint
    is_i2v = bool(reference_image)
    endpoint = (
        "https://api.klingai.com/v1/videos/image2video"
        if is_i2v
        else "https://api.klingai.com/v1/videos/text2video"
    )

    payload = {
        "model": "kling-v1",
        "prompt": prompt,
        "aspect_ratio": "16:9" if aspect == "16:9" else "9:16",
        "duration": str(duration) if str(duration) in ("5", "10") else "5",
        "mode": "std",
    }
    if is_i2v:
        payload["image"] = reference_image

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": auth_header,
            "Content-Type": "application/json",
            "User-Agent": "VeoStudioBridge/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        code = res.get("code")
        if code != 0 and code != 200:
            raise RuntimeError(f"Kling API error ({code}): {res.get('message')}")
        task_id = (res.get("data") or {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"Kling returned no task_id: {res}")
        return f"kling_{task_id}"


def kling_check_status(task_id, creds=None):
    raw_id = task_id.replace("kling_", "", 1)
    token = (creds or {}).get("kling_token")
    ak = (creds or {}).get("kling_ak")
    sk = (creds or {}).get("kling_sk")

    auth_header = f"Bearer {token}" if token else f"Basic {base64.b64encode(f'{ak}:{sk}'.encode()).decode()}" if ak else None
    url = f"https://api.klingai.com/v1/videos/text2video/{raw_id}"

    req = urllib.request.Request(url, headers={"Authorization": auth_header, "User-Agent": "VeoStudioBridge/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        data = res.get("data") or {}
        st = data.get("task_status")
        if st in ("succeed", "completed"):
            works = data.get("task_result", {}).get("videos", [])
            v_url = works[0].get("url") if works else None
            return {"status": "completed", "progress": 100, "video_url": v_url}
        elif st in ("failed", "cancelled"):
            return {"status": "failed", "error": data.get("task_status_msg", "Generation failed")}
        elif st == "processing":
            return {"status": "in_progress", "progress": 50}
        return {"status": "queued", "progress": 0}


# ---------- 2. Luma Dream Machine Adapter ----------
def luma_create_task(prompt, aspect="16:9", reference_image=None, creds=None):
    """Submit a task to Luma Dream Machine."""
    key = (creds or {}).get("luma_key")
    if not key:
        raise ValueError("Missing Luma key. Set LUMA_API_KEY in .env.")

    payload = {
        "prompt": prompt,
        "aspect_ratio": "16:9" if aspect == "16:9" else "9:16",
    }
    if reference_image:
        payload["keyframes"] = {
            "frame0": {"type": "image", "url": reference_image}
        }

    url = "https://api.lumalabs.ai/dream-machine/v1/generations"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "VeoStudioBridge/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        gen_id = res.get("id")
        if not gen_id:
            raise RuntimeError(f"Luma returned no generation ID: {res}")
        return f"luma_{gen_id}"


def luma_check_status(task_id, creds=None):
    raw_id = task_id.replace("luma_", "", 1)
    key = (creds or {}).get("luma_key")
    url = f"https://api.lumalabs.ai/dream-machine/v1/generations/{raw_id}"

    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", "User-Agent": "VeoStudioBridge/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        state = res.get("state")
        if state == "completed":
            v_url = (res.get("assets") or {}).get("video")
            return {"status": "completed", "progress": 100, "video_url": v_url}
        elif state == "failed":
            return {"status": "failed", "error": res.get("failure_reason", "Luma generation failed")}
        elif state == "dreaming":
            return {"status": "in_progress", "progress": 50}
        return {"status": "queued", "progress": 0}


# ---------- 3. Hailuo / MiniMax Adapter ----------
def hailuo_create_task(prompt, reference_image=None, creds=None):
    """Submit a task to Hailuo / MiniMax Video-01."""
    key = (creds or {}).get("hailuo_key")
    if not key:
        raise ValueError("Missing Hailuo/MiniMax key. Set MINIMAX_API_KEY in .env.")

    payload = {
        "model": "video-01",
        "prompt": prompt,
    }
    if reference_image:
        payload["first_frame_image"] = reference_image

    url = "https://api.minimax.chat/v1/video_generation"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "VeoStudioBridge/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        task_id = res.get("task_id")
        if not task_id:
            base_resp = res.get("base_resp", {})
            err = base_resp.get("status_msg") or res
            raise RuntimeError(f"MiniMax returned error: {err}")
        return f"hailuo_{task_id}"


def hailuo_check_status(task_id, creds=None):
    raw_id = task_id.replace("hailuo_", "", 1)
    key = (creds or {}).get("hailuo_key")
    url = f"https://api.minimax.chat/v1/query/video_generation?task_id={raw_id}"

    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", "User-Agent": "VeoStudioBridge/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        st = res.get("status")
        if st == "Success":
            file_id = res.get("file_id")
            # Query download URL
            dl_url = f"https://api.minimax.chat/v1/files/retrieve?file_id={file_id}"
            dl_req = urllib.request.Request(dl_url, headers={"Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(dl_req, timeout=20) as dl_resp:
                dl_data = json.loads(dl_resp.read().decode("utf-8"))
                v_url = (dl_data.get("file") or {}).get("download_url")
                return {"status": "completed", "progress": 100, "video_url": v_url}
        elif st in ("Fail", "Failed"):
            return {"status": "failed", "error": "MiniMax generation failed"}
        elif st == "Processing":
            return {"status": "in_progress", "progress": 50}
        return {"status": "queued", "progress": 0}


# ---------- 4. OpenArt AI Adapter ----------
def openart_cli_path():
    p = Path(__file__).parent / ".venv" / "Scripts" / "openart.exe"
    if p.exists():
        return str(p)
    return shutil.which("openart") or "openart"


def openart_has_auth():
    if os.environ.get("OPENART_TOKEN"):
        return True
    cred = Path.home() / ".openart" / "cli-credentials.json"
    return cred.exists()


def openart_generate_video(prompt, aspect="16:9", duration=5, reference_image=None, out_path=None):
    """Generate video using official OpenArt CLI."""
    cli = openart_cli_path()
    if not openart_has_auth():
        raise RuntimeError("OpenArt is not authenticated. Run '.\\.venv\\Scripts\\openart.exe login' in terminal to link your account.")

    cmd = [cli, "generate", "video", prompt, "--json"]
    if aspect in ("16:9", "9:16"):
        cmd.extend(["--aspect-ratio", aspect])
    if reference_image:
        cmd.extend(["--image", str(reference_image)])
    if out_path:
        cmd.extend(["-o", str(out_path)])

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        err = p.stderr.strip() or p.stdout.strip()
        raise RuntimeError(f"OpenArt error: {err}")

    try:
        data = json.loads(p.stdout)
        url = data.get("url") or (data.get("video") or {}).get("url") or (data.get("result") or {}).get("url")
        return url or str(out_path)
    except Exception:
        if out_path and Path(out_path).exists():
            return str(out_path)
        raise RuntimeError(f"OpenArt finished but produced unexpected output: {p.stdout}")


# ---------- Unified API Endpoints ----------
@app.get("/health")
def health():
    creds = get_env_credentials()
    return jsonify({
        "status": "healthy",
        "providers": {
            "kling": bool(creds.get("kling_token") or creds.get("kling_ak")),
            "luma": bool(creds.get("luma_key")),
            "hailuo": bool(creds.get("hailuo_key")),
            "openart": openart_has_auth(),
        }
    })


@app.post("/api/v1/video/generate")
def generate_video():
    """Single unified endpoint for all 3 video providers."""
    data = request.get_json(force=True, silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify(error="Prompt is required"), 400

    provider = (data.get("provider") or "auto").strip().lower()
    aspect = data.get("aspect", "16:9")
    duration = int(data.get("duration", 5))
    reference_image = data.get("reference_image")
    creds = get_env_credentials()

    # Determine provider
    chosen_provider = provider
    if provider == "auto":
        if openart_has_auth():
            chosen_provider = "openart"
        elif creds.get("kling_token") or creds.get("kling_ak"):
            chosen_provider = "kling"
        elif creds.get("luma_key"):
            chosen_provider = "luma"
        elif creds.get("hailuo_key"):
            chosen_provider = "hailuo"
        else:
            return jsonify(error="No credentials configured for OpenArt, Kling, Luma, or Hailuo. Set keys in .env or run openart login."), 400

    try:
        if chosen_provider == "openart":
            v_url = openart_generate_video(prompt, aspect, duration, reference_image)
            task_id = f"openart_{uuid.uuid4().hex[:8]}"
            TASKS[task_id] = {
                "id": task_id,
                "provider": "openart",
                "status": "completed",
                "progress": 100,
                "video_url": v_url,
                "error": None,
            }
            return jsonify({
                "task_id": task_id,
                "provider": "openart",
                "status": "completed",
                "video_url": v_url,
                "poll_url": f"/api/v1/video/status/{task_id}",
            })
        elif chosen_provider == "kling":
            task_id = kling_create_task(prompt, aspect, duration, reference_image, creds)
        elif chosen_provider == "luma":
            task_id = luma_create_task(prompt, aspect, reference_image, creds)
        elif chosen_provider == "hailuo":
            task_id = hailuo_create_task(prompt, reference_image, creds)
        else:
            return jsonify(error=f"Unsupported provider: {provider}"), 400

        task_record = {
            "id": task_id,
            "provider": chosen_provider,
            "created_at": time.time(),
            "status": "queued",
            "progress": 0,
            "video_url": None,
            "error": None,
        }
        TASKS[task_id] = task_record

        return jsonify({
            "task_id": task_id,
            "provider": chosen_provider,
            "status": "queued",
            "poll_url": f"/api/v1/video/status/{task_id}",
        })

    except Exception as e:
        return jsonify(error=str(e)), 500


@app.get("/api/v1/video/status/<task_id>")
def get_task_status(task_id):
    """Single unified polling endpoint."""
    creds = get_env_credentials()
    try:
        if task_id.startswith("kling_"):
            res = kling_check_status(task_id, creds)
        elif task_id.startswith("luma_"):
            res = luma_check_status(task_id, creds)
        elif task_id.startswith("hailuo_"):
            res = hailuo_check_status(task_id, creds)
        else:
            return jsonify(error="Unknown task ID format"), 404

        if task_id in TASKS:
            TASKS[task_id].update(res)

        return jsonify({
            "task_id": task_id,
            **res,
        })
    except Exception as e:
        return jsonify(task_id=task_id, status="failed", error=str(e)), 500


if __name__ == "__main__":
    port = int(os.environ.get("BRIDGE_PORT", "5050"))
    print(f"\n  ========================================================")
    print(f"  Unified Free-Tier Video Bridge")
    print(f"  Listening on http://127.0.0.1:{port}")
    print(f"  Single Endpoint: POST http://127.0.0.1:{port}/api/v1/video/generate")
    print(f"  ========================================================\n")
    app.run(host="127.0.0.1", port=port, threaded=True)

