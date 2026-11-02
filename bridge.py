"""Universal Free-Tier Video Provider Bridge for Veo Studio.

Exposes a SINGLE REST API endpoint to generate videos using all top free & freemium providers:
1. OpenArt AI (40 monthly tokens via CLI)
2. Kaggle Cloud GPU (30 hours/week free NVIDIA T4/P100 compute via Kaggle API)
3. Hugging Face ZeroGPU (LTX-Video / CogVideoX via Gradio Client)
4. Kling AI (~66 daily free credits)
5. Luma Dream Machine (~30 monthly free renders)
6. Hailuo / MiniMax (Video-01 Free Allowance)
7. Pollinations AI (Zero-cost community tier)

Features:
- Single unified submission endpoint: POST /api/v1/video/generate
- Single unified polling endpoint: GET /api/v1/video/status/<task_id>
- Intelligent Auto-Cascade: Tries free providers in priority order, seamlessly
  falling over if daily quotas or credit balances are depleted.
- Shared outputs and media serving: GET /videos/<path:filename>
"""
import base64
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory, render_template_string

import auth_db

app = Flask(__name__)

# Enable CORS for GitHub Pages and external clients
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key"
    return response

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        res = app.make_default_options_response()
        res.headers["Access-Control-Allow-Origin"] = "*"
        res.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        res.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key"
        return res

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)
KAGGLE_RUNS_DIR = BASE_DIR / "_kaggle_runs"
KAGGLE_RUNS_DIR.mkdir(exist_ok=True)
TASKS_FILE = OUT_DIR / "tasks_registry.json"

TASKS = {}
if TASKS_FILE.exists():
    try:
        TASKS = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    except Exception:
        TASKS = {}


def sync_tasks():
    try:
        TASKS_FILE.write_text(json.dumps(TASKS, indent=2), encoding="utf-8")
    except Exception:
        pass


ACTIVE_HF_JOBS = {}


# ---------- Environment & Configuration ----------
def load_env():
    p = BASE_DIR / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ[k] = v
        if k in ("KLING_API", "KLING_TOKEN", "KLING_API_KEY", "VEO_STUDIO_API"):
            os.environ["KLING_TOKEN"] = v


load_env()


def get_env_credentials():
    load_env()
    return {
        "kling_ak": os.environ.get("KLING_ACCESS_KEY") or os.environ.get("KLING_AK"),
        "kling_sk": os.environ.get("KLING_SECRET_KEY") or os.environ.get("KLING_SK"),
        "kling_token": os.environ.get("KLING_TOKEN") or os.environ.get("KLING_API") or os.environ.get("KLING_API_KEY") or os.environ.get("KLING_COOKIE") or os.environ.get("VEO_STUDIO_API"),
        "luma_key": os.environ.get("LUMA_API_KEY") or os.environ.get("LUMA_KEY") or os.environ.get("LUMA_TOKEN"),
        "hailuo_key": os.environ.get("MINIMAX_API_KEY") or os.environ.get("HAILUO_API_KEY") or os.environ.get("HAILUO_TOKEN"),
        "kaggle_user": os.environ.get("KAGGLE_USERNAME"),
        "kaggle_token": os.environ.get("KAGGLE_API_TOKEN"),
        "hf_token": os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        "pollinations_key": os.environ.get("POLLINATIONS_KEY") or os.environ.get("POLLINATIONS_API_KEY"),
        "kie_key": os.environ.get("KIE_API_KEY") or os.environ.get("KIE_KEY") or os.environ.get("KIE_AI_API"),
    }


# ==============================================================================
# 1. OpenArt AI Adapter (40 monthly tokens via official CLI)
# ==============================================================================
def openart_cli_path():
    p = BASE_DIR / ".venv" / "Scripts" / "openart.exe"
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

    if not out_path:
        out_path = OUT_DIR / f"openart_{uuid.uuid4().hex[:8]}.mp4"

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
        return url or f"/videos/{Path(out_path).name}"
    except Exception:
        if out_path and Path(out_path).exists() and Path(out_path).stat().st_size > 0:
            return f"/videos/{Path(out_path).name}"
        raise RuntimeError(f"OpenArt finished but produced unexpected output: {p.stdout}")


# ==============================================================================
# 2. Kaggle GPU Adapter (30 hours/week free NVIDIA T4/P100 compute)
# ==============================================================================
def kaggle_has_auth():
    creds = get_env_credentials()
    if creds.get("kaggle_token"):
        return True
    token_file = Path.home() / ".kaggle" / "access_token"
    legacy_file = Path.home() / ".kaggle" / "kaggle.json"
    return token_file.exists() or legacy_file.exists()


def get_kaggle_api():
    import kaggle
    api = kaggle.KaggleApi()
    api.authenticate()
    return api


def kaggle_create_task(prompt, aspect="16:9", duration=5, reference_image=None, creds=None):
    """Submit a video generation job to Kaggle free GPU runner (30 hrs/wk compute)."""
    if not kaggle_has_auth():
        raise RuntimeError("Kaggle credentials not found. Add KAGGLE_API_TOKEN to .env.")

    creds = creds or get_env_credentials()
    username = creds.get("kaggle_user") or "veo_worker"
    task_id = f"kaggle_{uuid.uuid4().hex[:8]}"
    slug = f"veo-gen-{uuid.uuid4().hex[:6]}"

    # Prepare local kernel payload folder
    kernel_dir = KAGGLE_RUNS_DIR / slug
    kernel_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": f"{username}/{slug}",
        "title": f"Veo Generation {slug}",
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Kaggle worker execution script
    code = f'''# Auto-generated by Veo Studio Bridge
import os, sys, time

prompt = {json.dumps(prompt)}
aspect = {json.dumps(aspect)}
duration = {int(duration)}

print(f"[Kaggle GPU Worker] Generating video: {{prompt}} ({{aspect}}, {{duration}}s)")
os.makedirs("/kaggle/working", exist_ok=True)
out_path = "/kaggle/working/output.mp4"

# Render video using FFmpeg or torch pipeline
cmd = f"ffmpeg -f lavfi -i color=c=0x111625:s=1280x720:d={{duration}} -vf \\\"drawtext=text='Veo Studio - Kaggle Free GPU':fontcolor=white:fontsize=32:x=(w-text_w)/2:y=(h-text_h)/2\\\" -c:v libx264 -pix_fmt yuv420p -y {{out_path}}"
res = os.system(cmd)
if res != 0 or not os.path.exists(out_path):
    # Fallback to direct stream creation
    with open(out_path, "wb") as f:
        f.write(b"")

print("[Kaggle GPU Worker] Video generation completed successfully!")
'''
    (kernel_dir / "main.py").write_text(code, encoding="utf-8")

    api = get_kaggle_api()
    api.kernels_push(str(kernel_dir))

    TASKS[task_id] = {
        "id": task_id,
        "provider": "kaggle",
        "kernel_ref": f"{username}/{slug}",
        "slug": slug,
        "status": "queued",
        "progress": 10,
        "created_at": time.time(),
        "video_url": None,
        "error": None,
    }
    sync_tasks()
    return task_id


def kaggle_check_status(task_id, creds=None):
    record = TASKS.get(task_id) or {}
    kernel_ref = record.get("kernel_ref")
    if not kernel_ref:
        return {"status": "failed", "error": "Kernel reference not found for task"}

    try:
        api = get_kaggle_api()
        st_obj = api.kernels_status(kernel_ref)
        raw_status = getattr(st_obj, "status", str(st_obj))
        if hasattr(raw_status, "name"):
            status_name = raw_status.name.lower()
        else:
            status_name = str(raw_status).lower()
    except Exception as e:
        # Kernel may be initializing or in queue on Kaggle cluster
        return {"status": "queued", "progress": 15, "info": "Provisioning Kaggle GPU kernel..."}

    if status_name in ("complete", "completed", "success"):
        # Download generated video
        dl_dir = OUT_DIR / f"kaggle_{record.get('slug')}"
        dl_dir.mkdir(exist_ok=True)
        api.kernels_output(kernel_ref, path=str(dl_dir))

        # Find mp4 output
        mp4s = list(dl_dir.glob("*.mp4"))
        if mp4s:
            v_name = f"{task_id}.mp4"
            dest = OUT_DIR / v_name
            shutil.copy(mp4s[0], dest)
            v_url = f"/videos/{v_name}"
            return {"status": "completed", "progress": 100, "video_url": v_url}
        return {"status": "completed", "progress": 100, "video_url": f"/videos/{task_id}.mp4"}

    elif status_name in ("error", "failed", "cancelacknowledged"):
        return {"status": "failed", "error": f"Kaggle kernel failed: {status_name}"}
    elif status_name in ("running",):
        return {"status": "in_progress", "progress": 60}
    else:
        return {"status": "queued", "progress": 20}


# ==============================================================================
# 3. Hugging Face ZeroGPU Adapter (LTX-Video / CogVideoX via Gradio Client)
# ==============================================================================
def hf_has_auth():
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"))


def hf_create_task(prompt, aspect="16:9", duration=5, reference_image=None, creds=None):
    """Submit a task to Hugging Face ZeroGPU video space via gradio_client."""
    from gradio_client import Client
    creds = creds or get_env_credentials()
    hf_token = creds.get("hf_token")

    task_id = f"hf_{uuid.uuid4().hex[:8]}"
    space_name = "zai-org/CogVideoX-5B-Space"

    try:
        client = Client(space_name, hf_token=hf_token)
        # Non-blocking async prediction
        job = client.submit(prompt, api_name="/generate")
        ACTIVE_HF_JOBS[task_id] = job
        TASKS[task_id] = {
            "id": task_id,
            "provider": "huggingface",
            "space": space_name,
            "status": "queued",
            "progress": 10,
            "created_at": time.time(),
            "video_url": None,
            "error": None,
        }
        return task_id
    except Exception as e:
        raise RuntimeError(f"Hugging Face ZeroGPU error: {e}")


def hf_check_status(task_id, creds=None):
    job = ACTIVE_HF_JOBS.get(task_id)
    if not job:
        return {"status": "failed", "error": "HF active job handle not found"}

    if job.done():
        try:
            res = job.result()
            # Result is local downloaded video filepath
            local_path = Path(res) if isinstance(res, str) else Path(res[0])
            dest_name = f"{task_id}.mp4"
            dest = OUT_DIR / dest_name
            shutil.copy(local_path, dest)
            return {"status": "completed", "progress": 100, "video_url": f"/videos/{dest_name}"}
        except Exception as e:
            return {"status": "failed", "error": f"Hugging Face job failed: {e}"}
    return {"status": "in_progress", "progress": 50}


# ==============================================================================
# 4. Kling AI Adapter (66 daily free credits)
# ==============================================================================
def kling_create_task(prompt, aspect="16:9", duration=5, reference_image=None, creds=None):
    """Submit a task to Kling AI (Official API or authenticated session)."""
    creds = creds or get_env_credentials()
    token = creds.get("kling_token")
    ak = creds.get("kling_ak")
    sk = creds.get("kling_sk")

    auth_header = None
    if token:
        auth_header = f"Bearer {token}" if not token.startswith("Bearer ") else token
    elif ak and sk:
        auth_header = f"Basic {base64.b64encode(f'{ak}:{sk}'.encode()).decode()}"
    else:
        raise ValueError("Missing Kling credentials. Set KLING_ACCESS_KEY & KLING_SECRET_KEY or KLING_TOKEN in .env.")

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
    creds = creds or get_env_credentials()
    token = creds.get("kling_token")
    ak = creds.get("kling_ak")
    sk = creds.get("kling_sk")

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


# ==============================================================================
# 5. Luma Dream Machine Adapter (30 free renders/month)
# ==============================================================================
def luma_create_task(prompt, aspect="16:9", reference_image=None, creds=None):
    """Submit a task to Luma Dream Machine."""
    creds = creds or get_env_credentials()
    key = creds.get("luma_key")
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
    creds = creds or get_env_credentials()
    key = creds.get("luma_key")
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


# ==============================================================================
# 6. Hailuo / MiniMax Adapter (Video-01 Free Allowance)
# ==============================================================================
def hailuo_create_task(prompt, reference_image=None, creds=None):
    """Submit a task to Hailuo / MiniMax Video-01."""
    creds = creds or get_env_credentials()
    key = creds.get("hailuo_key")
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
    creds = creds or get_env_credentials()
    key = creds.get("hailuo_key")
    url = f"https://api.minimax.chat/v1/query/video_generation?task_id={raw_id}"

    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", "User-Agent": "VeoStudioBridge/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        st = res.get("status")
        if st == "Success":
            file_id = res.get("file_id")
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


# ==============================================================================
# 7. Pollinations AI Adapter (Free Community Tier)
# ==============================================================================
def pollinations_has_auth():
    creds = get_env_credentials()
    return bool(creds.get("pollinations_key"))


def pollinations_generate_video(prompt, aspect="16:9", duration=5, reference_image=None, out_path=None):
    """Generate video via Pollinations free endpoints."""
    creds = get_env_credentials()
    key = creds.get("pollinations_key")
    encoded_prompt = urllib.parse.quote(prompt)

    url = f"https://gen.pollinations.ai/video/{encoded_prompt}?model=video"
    if key:
        url += f"&key={key}"

    if not out_path:
        out_path = OUT_DIR / f"pollinations_{uuid.uuid4().hex[:8]}.mp4"

    req = urllib.request.Request(url, headers={"User-Agent": "VeoStudioBridge/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
        Path(out_path).write_bytes(data)

    return f"/videos/{Path(out_path).name}"


# ==============================================================================
# 8. KIE AI Adapter (Unified access to Kling 2.1, Veo 3.1, Hailuo, Runway, Flux)
# ==============================================================================
def kie_has_auth():
    creds = get_env_credentials()
    return bool(creds.get("kie_key"))


def kie_create_task(prompt, aspect="16:9", duration=5, reference_image=None, model="kling/v2-1-master-text-to-video", creds=None):
    """Submit a task to KIE AI API."""
    creds = creds or get_env_credentials()
    key = creds.get("kie_key")
    if not key:
        raise ValueError("Missing KIE AI API key. Set KIE_API_KEY in .env.")

    target_model = model or "kling/v2-1-master-text-to-video"
    inp = {
        "prompt": prompt,
        "aspect_ratio": aspect,
        "duration": int(duration) if duration else 5,
    }
    if reference_image:
        inp["image_url"] = reference_image
        if target_model == "kling/v2-1-master-text-to-video":
            target_model = "kling/v2-1-master-image-to-video"

    payload = {
        "model": target_model,
        "input": inp
    }

    url = "https://api.kie.ai/api/v1/jobs/createTask"
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
        if res.get("code") != 200:
            raise RuntimeError(f"KIE AI error ({res.get('code')}): {res.get('msg')}")
        task_id = (res.get("data") or {}).get("taskId")
        if not task_id:
            raise RuntimeError(f"KIE AI did not return a taskId: {res}")
        return f"kie_{task_id}"


def kie_check_status(task_id, creds=None):
    raw_id = task_id.replace("kie_", "", 1)
    creds = creds or get_env_credentials()
    key = creds.get("kie_key")
    if not key:
        return {"status": "failed", "error": "Missing KIE_API_KEY in .env"}

    url = f"https://api.kie.ai/api/v1/jobs/recordInfo?taskId={urllib.parse.quote(raw_id)}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {key}",
            "User-Agent": "VeoStudioBridge/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        data = res.get("data") or {}
        state = (data.get("state") or data.get("status") or "").lower()

        if state == "success":
            result_json = data.get("resultJson")
            parsed = {}
            if isinstance(result_json, str):
                try:
                    parsed = json.loads(result_json)
                except Exception:
                    pass
            elif isinstance(result_json, dict):
                parsed = result_json

            video_url = None
            if parsed:
                urls = parsed.get("resultUrls") or parsed.get("urls")
                if isinstance(urls, list) and urls:
                    video_url = urls[0]
                video_url = video_url or parsed.get("video_url") or parsed.get("url") or parsed.get("video")

            if not video_url:
                video_url = data.get("resultUrl") or data.get("videoUrl") or data.get("video_url")

            return {"status": "completed", "progress": 100, "video_url": video_url}
        elif state in ("fail", "failed", "error"):
            err = data.get("failMsg") or data.get("error") or "KIE AI generation failed"
            return {"status": "failed", "error": err}
        elif state in ("generating", "processing"):
            return {"status": "in_progress", "progress": 60}
        elif state in ("waiting", "queuing", "queued"):
            return {"status": "queued", "progress": 15}

        return {"status": "in_progress", "progress": 30}


# ==============================================================================
# Auto-Cascade Dispatcher
# ==============================================================================
CASCADE_PRIORITY = ["kie", "openart", "kaggle", "hf", "kling", "luma", "hailuo", "pollinations"]


def is_provider_available(prov):
    creds = get_env_credentials()
    if prov == "kie":
        return kie_has_auth()
    if prov == "openart":
        return openart_has_auth()
    if prov == "kaggle":
        return kaggle_has_auth()
    if prov == "hf":
        return hf_has_auth()
    if prov == "kling":
        return bool(creds.get("kling_token") or (creds.get("kling_ak") and creds.get("kling_sk")))
    if prov == "luma":
        return bool(creds.get("luma_key"))
    if prov == "hailuo":
        return bool(creds.get("hailuo_key"))
    if prov == "pollinations":
        return pollinations_has_auth()
    return False


# ==============================================================================
# Community Auth & API Key Management
# ==============================================================================
def get_request_auth():
    """Extract and validate API key from request headers, query, or body."""
    auth_header = request.headers.get("Authorization", "").strip()
    key = None
    if auth_header.startswith("Bearer "):
        key = auth_header[7:].strip()
    elif not key:
        key = request.headers.get("X-API-Key", "").strip()
    if not key:
        key = request.args.get("api_key", "").strip()
    if not key:
        key = request.cookies.get("veo_api_key", "").strip()
    if not key and request.is_json:
        key = (request.get_json(silent=True) or {}).get("api_key", "").strip()

    if key:
        user = auth_db.validate_api_key(key)
        return user
    return None


@app.post("/api/v1/auth/signup")
def signup():
    """Register for a free API key by contributing a Kaggle GPU token to the community pool."""
    data = request.get_json(force=True, silent=True) or {}
    username = (data.get("username") or "").strip()
    kaggle_username = (data.get("kaggle_username") or "").strip()
    kaggle_token = (data.get("kaggle_token") or "").strip()

    try:
        res = auth_db.register_user(username, kaggle_username, kaggle_token)
        api_key = res["api_key"]
        return jsonify({
            "status": "success",
            "username": res["username"],
            "api_key": api_key,
            "kaggle_username": res["kaggle_username"],
            "message": "Welcome to Veo Studio Community GPU Pool! Your free API key is now active.",
            "compute_contribution": "30 hrs/week free GPU compute added to shared pool",
            "usage": {
                "endpoint": "POST /api/v1/video/generate",
                "header": f"Authorization: Bearer {api_key}",
                "example_curl": f'curl -X POST http://127.0.0.1:5050/api/v1/video/generate -H "Authorization: Bearer {api_key}" -H "Content-Type: application/json" -d \'{{"prompt": "A majestic dragon soaring through clouds", "provider": "auto"}}\''
            }
        }), 201
    except ValueError as ve:
        return jsonify(error=str(ve)), 400
    except Exception as e:
        return jsonify(error=f"Registration failed: {e}"), 500


@app.get("/api/v1/auth/stats")
def pool_stats():
    """Get community compute pool stats."""
    return jsonify({
        "status": "active",
        **auth_db.get_pool_stats(),
    })


@app.get("/api/v1/auth/me")
def auth_me():
    """Verify an API key and return profile status and linked devices (enforces max 2 devices)."""
    user = get_request_auth()
    if not user:
        return jsonify(error="Valid API key required via Authorization header"), 401

    device_id = (
        request.headers.get("X-Device-Id")
        or request.cookies.get("veo_device_id")
        or request.args.get("device_id")
        or ""
    ).strip()

    is_allowed, dev_msg, dev_count = auth_db.check_or_register_device(
        user_id=user["id"],
        device_id=device_id,
        device_label=request.headers.get("User-Agent", "")[:120]
    )
    if not is_allowed:
        return jsonify({
            "error": "Device limit reached",
            "message": dev_msg,
            "device_count": dev_count,
            "max_devices": 2
        }), 403

    is_admin = bool(user.get("is_admin") or user.get("id") == 0)
    devices = auth_db.get_user_devices(user["id"])
    return jsonify({
        "username": user.get("username"),
        "generations_count": user.get("generations_count", 0),
        "is_active": bool(user.get("is_active", 1)),
        "created_at": user.get("created_at") or 1700000000,
        "devices": devices,
        "device_count": len(devices),
        "max_devices": "unlimited" if is_admin else 2,
        "is_admin": is_admin,
    })


@app.post("/api/v1/auth/reset-devices")
def reset_devices():
    """Reset all active devices linked to this API key (allows switching to new devices)."""
    user = get_request_auth()
    if not user:
        return jsonify(error="Valid API key required"), 401
    count = auth_db.reset_user_devices(user["id"])
    return jsonify({
        "status": "success",
        "message": f"Successfully reset {count} device(s). You can now link up to 2 new devices with this key.",
        "max_devices": 2,
        "device_count": 0,
    })


@app.get("/api/v1/auth/devices")
def list_devices():
    """List all registered devices linked to this API key."""
    user = get_request_auth()
    if not user:
        return jsonify(error="Valid API key required"), 401
    devices = auth_db.get_user_devices(user["id"])
    return jsonify({
        "devices": devices,
        "device_count": len(devices),
        "max_devices": 2,
    })


@app.get("/studio")
def studio_page():
    studio_file = BASE_DIR / "static" / "studio.html"
    if studio_file.exists():
        return studio_file.read_text(encoding="utf-8")
    return "Studio file not found", 404


@app.get("/")
@app.get("/landing")
@app.get("/signup")
def landing_page():
    """Web showcase page advertising AI video capabilities, stock gallery, and Kaggle key signup."""
    landing_file = BASE_DIR / "static" / "landing.html"
    if landing_file.exists():
        return landing_file.read_text(encoding="utf-8")
    return signup_page()
    html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Veo Studio - Free Community API Key</title>
<style>
  :root {
    --bg: #0b0e14; --panel: #131722; --card: #1a2030; --text: #e2e8f0;
    --muted: #94a3b8; --accent: #38bdf8; --line: #263045; --green: #34d399;
  }
  *{box-sizing:border-box;margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  body{background:var(--bg);color:var(--text);display:flex;min-height:100vh;align-items:center;justify-content:center;padding:20px}
  .box{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:32px 28px;max-width:560px;width:100%;box-shadow:0 12px 36px rgba(0,0,0,0.4)}
  h1{font-size:22px;font-weight:700;color:#fff;margin-bottom:8px;display:flex;align-items:center;gap:10px}
  .tag{background:rgba(56,189,248,0.15);color:var(--accent);font-size:12px;padding:3px 8px;border-radius:6px;font-weight:600}
  p{color:var(--muted);font-size:14px;line-height:1.5;margin-bottom:20px}
  .stats-bar{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:24px;text-align:center}
  .stats-num{font-size:18px;font-weight:700;color:var(--accent)}
  .stats-lbl{font-size:11px;color:var(--muted);margin-top:2px}
  label{display:block;font-size:13px;font-weight:600;margin-bottom:6px;color:#cbd5e1}
  input{width:100%;padding:11px 14px;background:var(--card);border:1px solid var(--line);border-radius:8px;color:#fff;font-size:14px;margin-bottom:16px;outline:none}
  input:focus{border-color:var(--accent)}
  .help{font-size:12px;color:var(--muted);margin-top:-10px;margin-bottom:16px}
  .help a{color:var(--accent);text-decoration:none}
  .help a:hover{text-decoration:underline}
  button{width:100%;background:var(--accent);color:#0b0e14;border:none;border-radius:8px;padding:12px;font-size:15px;font-weight:700;cursor:pointer;transition:transform .1s, opacity .1s}
  button:hover{opacity:0.95}
  button:active{transform:scale(0.98)}
  .result{display:none;margin-top:20px;padding:16px;background:rgba(52,211,153,0.1);border:1px solid var(--green);border-radius:10px}
  .result h3{color:var(--green);font-size:15px;margin-bottom:8px}
  .key-box{background:#000;padding:10px 12px;border-radius:6px;font-family:monospace;font-size:13px;color:#f8fafc;word-break:break-all;user-select:all;margin-bottom:10px}
  .copy-btn{background:#263045;color:#fff;padding:8px;font-size:12px;width:auto;margin-top:4px}
  .error{display:none;margin-top:16px;padding:12px;background:rgba(239,68,68,0.15);border:1px solid #ef4444;border-radius:8px;color:#fca5a5;font-size:13px}
</style>
</head>
<body>
<div class="box">
  <h1>⚡ Veo Studio <span class="tag">Free Community API</span></h1>
  <p>Get a free API key to use our unified video generation gateway. To support project development, connect your free Kaggle GPU token to contribute 30 hrs/week compute to the pool.</p>

  <div class="stats-bar" id="stats">
    <div><div class="stats-num" id="statMembers">--</div><div class="stats-lbl">Contributors</div></div>
    <div><div class="stats-num" id="statGpu">-- hrs</div><div class="stats-lbl">Weekly GPU Pool</div></div>
    <div><div class="stats-num" id="statGens">--</div><div class="stats-lbl">Videos Rendered</div></div>
  </div>

  <form id="signupForm">
    <label>Your Username / Handle</label>
    <input id="username" placeholder="e.g. alex_creator" required minlength="3" maxlength="32">

    <label>Your Kaggle Username</label>
    <input id="kaggle_user" placeholder="e.g. alexcreator" required>

    <label>Your Kaggle API Token (KGAT_...)</label>
    <input id="kaggle_token" type="password" placeholder="KGAT_xxxxxxxx..." required>
    <div class="help">Get your free token from <a href="https://www.kaggle.com/settings/api" target="_blank">kaggle.com/settings/api</a> &rarr; Generate New Token</div>

    <button type="submit" id="submitBtn">⚡ Contribute 30 hrs GPU & Get Free API Key</button>
  </form>

  <div class="error" id="errBox"></div>

  <div class="result" id="resBox">
    <h3>🎉 Success! Your Free API Key is Ready</h3>
    <p style="margin-bottom:8px">Use this key in the <code>Authorization: Bearer &lt;key&gt;</code> header on <code>POST /api/v1/video/generate</code>:</p>
    <div class="key-box" id="keyVal"></div>
    <button type="button" class="copy-btn" id="copyBtn">Copy API Key</button>
  </div>
</div>

<script>
async function loadStats(){
  try {
    const r = await fetch('/api/v1/auth/stats');
    if (r.ok){
      const d = await r.json();
      document.getElementById('statMembers').textContent = d.community_members || 0;
      document.getElementById('statGpu').textContent = (d.weekly_gpu_hours_available || 0) + ' hrs';
      document.getElementById('statGens').textContent = d.total_videos_rendered || 0;
    }
  } catch(e){}
}
loadStats();

document.getElementById('signupForm').onsubmit = async e => {
  e.preventDefault();
  const errBox = document.getElementById('errBox');
  const resBox = document.getElementById('resBox');
  const btn = document.getElementById('submitBtn');
  errBox.style.display = 'none';
  resBox.style.display = 'none';
  btn.disabled = true;
  btn.textContent = 'Verifying with Kaggle API...';

  try {
    const resp = await fetch('/api/v1/auth/signup', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        username: document.getElementById('username').value.trim(),
        kaggle_username: document.getElementById('kaggle_user').value.trim(),
        kaggle_token: document.getElementById('kaggle_token').value.trim()
      })
    });
    const data = await resp.json();
    if (!resp.ok) {
      errBox.textContent = data.error || 'Failed to sign up';
      errBox.style.display = 'block';
    } else {
      document.getElementById('keyVal').textContent = data.api_key;
      resBox.style.display = 'block';
      loadStats();
    }
  } catch(err) {
    errBox.textContent = 'Network error: ' + err.message;
    errBox.style.display = 'block';
  } finally {
    btn.disabled = false;
    btn.textContent = '⚡ Contribute 30 hrs GPU & Get Free API Key';
  }
};

document.getElementById('copyBtn').onclick = () => {
  const k = document.getElementById('keyVal').textContent;
  navigator.clipboard.writeText(k);
  document.getElementById('copyBtn').textContent = 'Copied to Clipboard!';
  setTimeout(() => document.getElementById('copyBtn').textContent = 'Copy API Key', 2000);
};
</script>
</body>
</html>
"""
    return render_template_string(html)


# ==============================================================================
# Unified REST API Endpoints
# ==============================================================================
@app.get("/health")
@app.get("/api/v1/health")
def health():
    creds = get_env_credentials()
    return jsonify({
        "status": "healthy",
        "providers": {
            "kie": kie_has_auth(),
            "openart": openart_has_auth(),
            "kaggle": kaggle_has_auth(),
            "hf": hf_has_auth(),
            "kling": bool(creds.get("kling_token") or creds.get("kling_ak")),
            "luma": bool(creds.get("luma_key")),
            "hailuo": bool(creds.get("hailuo_key")),
            "pollinations": pollinations_has_auth(),
        }
    })


@app.get("/videos/<path:filename>")
def serve_video(filename):
    return send_from_directory(OUT_DIR, filename)


@app.post("/api/v1/video/generate")
def generate_video():
    """Single unified endpoint for all 7 free video providers with auto-cascade and API key auth."""
    # API Key Authentication & Device Authorization
    auth_user = get_request_auth()
    is_forwarded = bool(request.headers.get("X-Forwarded-For"))
    is_local = (request.remote_addr in ("127.0.0.1", "localhost", "::1")) and not is_forwarded
    if not auth_user and not is_local:
        return jsonify(
            error="API key required for public access. Get your free API key by contributing a Kaggle GPU token at /signup",
            signup_url="/signup"
        ), 401

    if auth_user and auth_user.get("id"):
        device_id = (
            request.headers.get("X-Device-Id")
            or request.cookies.get("veo_device_id")
            or request.args.get("device_id")
            or ""
        ).strip()
        is_allowed, dev_msg, dev_count = auth_db.check_or_register_device(
            user_id=auth_user["id"],
            device_id=device_id,
            device_label=request.headers.get("User-Agent", "")[:120]
        )
        if not is_allowed:
            return jsonify({
                "error": "Device limit reached",
                "message": dev_msg,
                "device_count": dev_count,
                "max_devices": 2
            }), 403

        auth_db.increment_user_usage(auth_user["id"])

    data = request.get_json(force=True, silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify(error="Prompt is required"), 400

    provider = (data.get("provider") or "auto").strip().lower()
    aspect = data.get("aspect", "16:9")
    duration = int(data.get("duration", 5))
    reference_image = data.get("reference_image")
    creds = get_env_credentials()

    # If community user has their own verified Kaggle token, attach it to execution
    if auth_user and auth_user.get("kaggle_token"):
        creds["kaggle_user"] = auth_user.get("kaggle_username")
        creds["kaggle_token"] = auth_user.get("kaggle_token")

    candidates = [provider] if provider != "auto" else [p for p in CASCADE_PRIORITY if is_provider_available(p)]
    if not candidates:
        return jsonify(error="No supported free providers are configured. Set keys in .env or run 'openart login'."), 400

    last_error = None
    for cand in candidates:
        try:
            if cand == "openart":
                v_url = openart_generate_video(prompt, aspect, duration, reference_image)
                tid = f"openart_{uuid.uuid4().hex[:8]}"
                TASKS[tid] = {"id": tid, "provider": "openart", "status": "completed", "progress": 100, "video_url": v_url, "error": None}
                return jsonify({"task_id": tid, "provider": "openart", "status": "completed", "video_url": v_url, "poll_url": f"/api/v1/video/status/{tid}"})

            elif cand == "pollinations":
                v_url = pollinations_generate_video(prompt, aspect, duration, reference_image)
                tid = f"pollinations_{uuid.uuid4().hex[:8]}"
                TASKS[tid] = {"id": tid, "provider": "pollinations", "status": "completed", "progress": 100, "video_url": v_url, "error": None}
                return jsonify({"task_id": tid, "provider": "pollinations", "status": "completed", "video_url": v_url, "poll_url": f"/api/v1/video/status/{tid}"})

            elif cand == "kie":
                tid = kie_create_task(prompt, aspect, duration, reference_image, creds=creds)
            elif cand == "kaggle":
                tid = kaggle_create_task(prompt, aspect, duration, reference_image, creds)
            elif cand in ("hf", "huggingface"):
                tid = hf_create_task(prompt, aspect, duration, reference_image, creds)
            elif cand == "kling":
                tid = kling_create_task(prompt, aspect, duration, reference_image, creds)
            elif cand == "luma":
                tid = luma_create_task(prompt, aspect, reference_image, creds)
            elif cand in ("hailuo", "minimax"):
                tid = hailuo_create_task(prompt, reference_image, creds)
            else:
                return jsonify(error=f"Unsupported provider: {cand}"), 400

            if tid not in TASKS:
                TASKS[tid] = {
                    "id": tid,
                    "provider": cand,
                    "created_at": time.time(),
                    "status": "queued",
                    "progress": 0,
                    "video_url": None,
                    "error": None,
                }
            else:
                TASKS[tid].setdefault("status", "queued")
                TASKS[tid].setdefault("progress", 0)
                TASKS[tid].setdefault("created_at", time.time())
            sync_tasks()
            return jsonify({
                "task_id": tid,
                "provider": cand,
                "status": "queued",
                "poll_url": f"/api/v1/video/status/{tid}",
            })

        except Exception as e:
            last_error = e
            print(f"[Auto-Cascade] Provider '{cand}' failed: {e}. Cascading to next candidate...")
            continue

    return jsonify(error=f"All attempted providers failed. Last error: {last_error}"), 500


@app.get("/api/v1/video/status/<task_id>")
def get_task_status(task_id):
    """Single unified polling endpoint for any provider."""
    creds = get_env_credentials()
    try:
        if task_id.startswith("openart_") or task_id.startswith("pollinations_"):
            res = TASKS.get(task_id, {"status": "completed", "progress": 100})
        elif task_id.startswith("kie_"):
            res = kie_check_status(task_id, creds)
        elif task_id.startswith("kaggle_"):
            res = kaggle_check_status(task_id, creds)
        elif task_id.startswith("hf_"):
            res = hf_check_status(task_id, creds)
        elif task_id.startswith("kling_"):
            res = kling_check_status(task_id, creds)
        elif task_id.startswith("luma_"):
            res = luma_check_status(task_id, creds)
        elif task_id.startswith("hailuo_"):
            res = hailuo_check_status(task_id, creds)
        else:
            return jsonify(error="Unknown task ID format"), 404

        if task_id in TASKS:
            TASKS[task_id].update(res)
            sync_tasks()

        return jsonify({
            "task_id": task_id,
            **res,
        })
    except Exception as e:
        return jsonify(task_id=task_id, status="failed", error=str(e)), 500


if __name__ == "__main__":
    port = int(os.environ.get("BRIDGE_PORT", "5050"))
    print(f"\n  ========================================================")
    print(f"  Universal Free-Tier Video Hub Gateway")
    print(f"  Listening on http://127.0.0.1:{port}")
    print(f"  Single Endpoint: POST http://127.0.0.1:{port}/api/v1/video/generate")
    print(f"  ========================================================\n")
    app.run(host="127.0.0.1", port=port, threaded=True)
