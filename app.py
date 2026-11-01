"""Veo Studio - Local web UI that turns a prompt into a 1-3 minute animated AI video.

Supports:
- Google Veo 3.1 & Veo 3.1 FL (First/Last frame chaining) via GenRelay
- Kling 1.0, MiniMax Video-01 & Luma Dream Machine via Fal.ai
- Gemini Omni 1.1 Flash via Google Direct Interactions API
- Gemini 3.5 Flash Lite & 3.8 Flash for multi-scene director story planning
"""
import base64
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import imageio_ffmpeg
from flask import Flask, abort, jsonify, request, send_from_directory
from google import genai
from google.genai import types

BASE = Path(__file__).parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

DEFAULT_PLANNER_MODEL = "gemini-3.5-flash-lite"
PLANNER_FALLBACKS = ["gemini-3.8-flash", "gemini-2.5-flash"]
CLIP_SECONDS = 8
MAX_CLIPS = 24
MAX_SECONDS = 180
FFMPEG = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()


def load_env():
    p = BASE / ".env"
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
        # Normalize alternative variable names
        if k in ("FAL.AI_API", "FAL_KEY", "FAL_API_KEY", "FAL_API"):
            os.environ["FAL_KEY"] = v
        if k in ("GENRELAY_API", "GENRELAY_API_KEY"):
            os.environ["GENRELAY_API"] = v
        if k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            os.environ["GEMINI_API_KEY"] = v


load_env()

DEFAULT_VIDEO_MODEL = (
    "veo_3_1" if (os.environ.get("GENRELAY_API") or os.environ.get("GENRELAY_API_KEY"))
    else "fal-ai/minimax-video/text-to-video" if (os.environ.get("FAL_KEY"))
    else "gemini-omni-1.1-flash"
)

app = Flask(__name__, static_folder="static", static_url_path="/static")
JOBS = {}


class Cancelled(Exception):
    pass


def sleep_interruptible(seconds, check_fn):
    """Sleep in short slices so cancellation responds immediately."""
    end = time.time() + seconds
    while time.time() < end:
        if check_fn:
            check_fn()
        time.sleep(min(0.5, max(0.05, end - time.time())))


def clips_for(seconds):
    return max(1, min(MAX_CLIPS, math.ceil(seconds / CLIP_SECONDS)))


# ---------- FFmpeg helpers ----------
def run_ff(args, check_rc=True):
    p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
                       capture_output=True, text=True)
    if check_rc and p.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + p.stderr[-300:])
    return p


def duration(path):
    p = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", p.stderr)
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 0.0


def last_frame(video, out):
    for ss in ("-0.15", "-1"):
        out.unlink(missing_ok=True)
        run_ff(["-sseof", ss, "-i", str(video), "-frames:v", "1", "-q:v", "2", str(out)], check_rc=False)
        if out.exists() and out.stat().st_size > 0:
            return
    raise RuntimeError("Could not extract the last frame of the clip")


def join(clips, out):
    if len(clips) == 1:
        shutil.copy(clips[0], out)
        return
    lst = out.with_suffix(".txt")
    lst.write_text(
        "".join("file '%s'\n" % c.resolve().as_posix().replace("'", "'\\''") for c in clips),
        encoding="utf-8",
    )
    try:
        out.unlink(missing_ok=True)
        r = run_ff(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)], check_rc=False)
        if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            run_ff(["-f", "concat", "-safe", "0", "-i", str(lst), "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)])
    finally:
        lst.unlink(missing_ok=True)


# ---------- Network & Upload Helpers ----------
def download_url(url, dest_path, check_fn):
    """Download a file with interruptible checks."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=45) as resp, open(dest_path, "wb") as f:
        while True:
            if check_fn:
                check_fn()
            chunk = resp.read(65536)
            if not chunk:
                break
            f.write(chunk)
    if not dest_path.exists() or dest_path.stat().st_size == 0:
        raise RuntimeError("Downloaded video is empty")


def upload_frame_for_chaining(frame_path, log_fn=None):
    """Uploads the extracted last-frame to a temporary direct image host for veo_3_1-fl / fal."""
    try:
        data = frame_path.read_bytes()
        boundary = "----WebKitFormBoundary" + uuid.uuid4().hex[:16]
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="reqtype"\r\n\r\nfileupload\r\n'
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="fileToUpload"; filename="frame.jpg"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode("utf-8") + data + f"\r\n--{boundary}--\r\n".encode("utf-8")
        req = urllib.request.Request(
            "https://catbox.moe/user/api.php",
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "Mozilla/5.0",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            url = resp.read().decode("utf-8").strip()
            if url.startswith("http"):
                return url
    except Exception as e:
        if log_fn:
            log_fn(f"Frame upload notice: {e}")
    return None


# ---------- GenRelay API Helpers ----------
def call_genrelay_create_task(api_key, model, prompt, aspect, resolution, frame_url=None):
    """Submit a task to GenRelay Unified Media API."""
    size = "1280x720" if aspect == "16:9" else "720x1280"
    payload = {
        "model": model,
        "prompt": prompt,
        "seconds": "8",
        "size": size,
        "tier": resolution if resolution in ("720p", "1080p") else "720p",
    }
    if frame_url and model == "veo_3_1-fl":
        payload["reference_images"] = [frame_url]

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://genrelay.ai/v1/videos",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "VeoStudio/2.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            task_id = res.get("task_id") or res.get("id")
            if not task_id:
                raise RuntimeError(f"GenRelay returned no task_id: {res}")
            return task_id
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="ignore")
        try:
            data = json.loads(raw)
            err_msg = data.get("message") or raw
            if data.get("code") == "insufficient_user_quota" or "quota" in err_msg.lower():
                raise RuntimeError(f"GenRelay balance exhausted ({err_msg}). Please top up credits at https://genrelay.ai to continue.")
            raise RuntimeError(f"GenRelay error ({e.code}): {err_msg}")
        except json.JSONDecodeError:
            raise RuntimeError(f"GenRelay error ({e.code}): {raw}")


def poll_genrelay_task(api_key, task_id, check_fn, log_fn, job, clip_idx, total_clips):
    """Poll GenRelay until task is completed or failed."""
    url = f"https://genrelay.ai/v1/videos/{task_id}"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": "VeoStudio/2.0"},
    )
    for _ in range(240):  # up to 16 minutes
        check_fn()
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                status = data.get("status")
                progress = data.get("progress", 0)
                if status == "in_progress":
                    job["message"] = f"Generating clip {clip_idx + 1} of {total_clips} ({progress}%)..."
                elif status == "completed":
                    video_url = (
                        data.get("video_url")
                        or data.get("url")
                        or (data.get("metadata") or {}).get("video_url")
                        or data.get("output_url")
                    )
                    if not video_url:
                        raise RuntimeError("GenRelay task completed but returned no video URL")
                    return video_url
                elif status == "failed":
                    err = (
                        (data.get("error") or {}).get("message")
                        or (data.get("metadata") or {}).get("url")
                        or "Task marked failed by GenRelay"
                    )
                    raise RuntimeError(f"GenRelay video generation failed: {err}")
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise RuntimeError(f"GenRelay authentication error: {e.read().decode('utf-8', 'ignore')}")
            # Other HTTP errors may be transient network hiccups, continue polling
        sleep_interruptible(4, check_fn)
    raise TimeoutError("GenRelay video generation timed out")


# ---------- Fal.ai API Helpers ----------
def call_fal_video(api_key, model, prompt, aspect, frame_url=None, check_fn=None, log_fn=None, job=None, clip_idx=0, total_clips=1):
    """Generate a video clip using Fal.ai Queue API."""
    actual_model = model
    payload = {"prompt": prompt}

    if "kling" in model:
        payload["aspect_ratio"] = aspect
        payload["duration"] = "5"
        if frame_url:
            actual_model = "fal-ai/kling-video/v1/standard/image-to-video"
            payload["image_url"] = frame_url
    elif "minimax" in model:
        if frame_url:
            actual_model = "fal-ai/minimax-video/image-to-video"
            payload["image_url"] = frame_url
    elif "luma" in model:
        payload["aspect_ratio"] = aspect
        if frame_url:
            payload["keyframes"] = {"frame0": {"type": "image", "url": frame_url}}
    else:
        if aspect in ("16:9", "9:16"):
            payload["aspect_ratio"] = aspect

    queue_url = f"https://queue.fal.run/{actual_model}"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        queue_url,
        data=body,
        headers={
            "Authorization": f"Key {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "VeoStudio/2.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            status_url = data.get("status_url")
            response_url = data.get("response_url")
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="ignore")
        try:
            err_data = json.loads(raw)
            detail = err_data.get("detail") or raw
            if "Exhausted balance" in detail or "locked" in detail:
                raise RuntimeError(f"Fal.ai balance exhausted: {detail} Top up credits at https://fal.ai/dashboard/billing")
            raise RuntimeError(f"Fal.ai API error ({e.code}): {detail}")
        except json.JSONDecodeError:
            raise RuntimeError(f"Fal.ai API error ({e.code}): {raw}")

    # Poll status
    for _ in range(240):  # up to 16 minutes
        if check_fn:
            check_fn()
        try:
            s_req = urllib.request.Request(status_url, headers={"Authorization": f"Key {api_key}"})
            with urllib.request.urlopen(s_req, timeout=20) as s_resp:
                s_data = json.loads(s_resp.read().decode("utf-8"))
                st = s_data.get("status")
                if st == "IN_PROGRESS":
                    if job:
                        job["message"] = f"Generating clip {clip_idx + 1} of {total_clips} (rendering on Fal.ai)..."
                elif st == "COMPLETED":
                    break
        except urllib.error.HTTPError:
            pass
        sleep_interruptible(4, check_fn)

    # Fetch final video URL
    r_req = urllib.request.Request(response_url, headers={"Authorization": f"Key {api_key}"})
    with urllib.request.urlopen(r_req, timeout=30) as r_resp:
        r_data = json.loads(r_resp.read().decode("utf-8"))
        video_obj = r_data.get("video") or {}
        v_url = video_obj.get("url")
        if not v_url:
            raise RuntimeError(f"Fal.ai returned no video URL: {r_data}")
        return v_url


# ---------- Planning & Retry Helpers ----------
def with_retry(fn, log, check_fn=None, tries=3):
    for attempt in range(1, tries + 1):
        if check_fn:
            check_fn()
        try:
            return fn()
        except Exception as e:
            if check_fn:
                check_fn()
            msg = str(e)
            zero_quota = "limit: 0" in msg or "free_tier" in msg.lower()
            fatal = zero_quota or (
                any(t in msg for t in ("400", "401", "403", "404", "NOT_FOUND", "PERMISSION",
                                       "API key", "INVALID_ARGUMENT"))
                and "RESOURCE_EXHAUSTED" not in msg
            )
            if attempt == tries or fatal:
                raise
            wait = 20 * attempt if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) else 8 * attempt
            log(f"API notice: {msg[:120]} - retrying in {wait}s")
            sleep_interruptible(wait, check_fn)


def friendly(msg):
    """Add a plain-language explanation to raw API errors."""
    if "Exhausted balance" in msg or "fal.ai/dashboard/billing" in msg:
        return "Fal.ai credit balance is exhausted ($0.00). Top up your credits at https://fal.ai/dashboard/billing to generate videos with Fal.ai. " + msg
    if "insufficient_user_quota" in msg or "quota: ＄" in msg or "quota: $" in msg:
        return "GenRelay credit balance exhausted. Top up your account at https://genrelay.ai to continue generating. " + msg
    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
        return (
            "Google refused the request: quota limit 0 on free tier. Video models require billing enabled on Google AI Studio, "
            "or use the recommended 'veo_3_1 (via GenRelay)' which does not require Google Cloud billing. Raw error: " + msg
        )
    if "404" in msg or "NOT_FOUND" in msg:
        return "Model not found or retired. Select 'veo_3_1' in Settings. Raw error: " + msg
    return msg


def plan_scenes(client, planner_model, idea, n, log, check_fn=None):
    """Ask Gemini to split the idea into n consecutive narrative shots."""
    if check_fn:
        check_fn()
    ask = (
        f"You are an expert film director. Split this video idea into exactly {n} "
        f"consecutive shots of about 8-10 seconds each. Each shot must continue "
        f"directly from the previous one with the same characters, setting and "
        f"art style, and together they must tell a complete story with a "
        f"beginning, middle and end. Return ONLY a JSON array of {n} strings. "
        f"Each string is a self-contained cinematic prompt for a video generation model, "
        f"describing the action, camera movement, character appearance, and mood.\n\nIdea: {idea}"
    )
    for model in [planner_model] + [m for m in PLANNER_FALLBACKS if m != planner_model]:
        if check_fn:
            check_fn()
        try:
            r = client.models.generate_content(
                model=model,
                contents=ask,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            arr = json.loads(r.text)
            if isinstance(arr, list):
                arr = [x for x in arr if isinstance(x, str) and x.strip()]
                if arr:
                    while len(arr) < n:
                        arr.append(f"Continue the story smoothly toward its conclusion. {idea}")
                    log(f"Scenes planned with {model}")
                    return arr[:n]
        except Exception as e:
            log(f"Planner {model} notice: {str(e)[:90]}")
    log("Using direct continuation prompts instead of a planner")
    return [idea] + [f"Continue the same story smoothly toward the ending. {idea}"] * (n - 1)


def save_omni_video(client, out, path, check_fn):
    """Save video from Google Interactions API."""
    uri = getattr(out, "uri", None)
    if uri:
        name = uri.split("/")[-1]
        for _ in range(120):
            if check_fn:
                check_fn()
            info = client.files.get(name=f"files/{name}")
            state = getattr(info.state, "name", str(info.state))
            if state == "ACTIVE":
                break
            if state == "FAILED":
                raise RuntimeError("Google reported the generated video file as FAILED")
            sleep_interruptible(4, check_fn)
        try:
            client.files.download(file=uri, destination=str(path))
        except TypeError:
            data = client.files.download(file=uri)
            path.write_bytes(data)
    elif getattr(out, "data", None):
        path.write_bytes(base64.b64decode(out.data))
    else:
        raise RuntimeError("Google response had no video data")
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError("Video download produced an empty file")


# ---------- Main Job Runner ----------
def run_job(job, gemini_key, genrelay_key, fal_key):
    def log(m):
        job["log"].append(time.strftime("%H:%M:%S  ") + m)

    def check():
        if job.get("cancel") or job.get("status") == "cancelled":
            raise Cancelled()

    try:
        target = job["seconds"]
        n = job["total"]
        log(f"Target ~{target}s -> up to {n} clips ({job['resolution']}, {job['aspect']}) using {job['model']}")

        # 1. Plan scenes with Gemini (or fallback)
        job["message"] = "Planning scenes with director AI..."
        scenes = None
        if gemini_key:
            try:
                g_client = genai.Client(api_key=gemini_key, http_options=types.HttpOptions(timeout=60_000))
                scenes = plan_scenes(g_client, job["planner"], job["prompt"], n, log, check_fn=check)
            except Exception as e:
                log(f"Gemini scene planning notice: {e}")

        if not scenes:
            scenes = [job["prompt"]] + [
                f"Continue the story smoothly with consistent characters and atmosphere. {job['prompt']}"
            ] * (n - 1)

        job["scenes"] = scenes
        check()

        clips, total_secs, frame = [], 0.0, None
        checkpoint = OUT / f"{job['id']}_so_far.mp4"

        is_genrelay = job["model"].startswith("veo_") or job["model"].startswith("genrelay_")
        is_fal = job["model"].startswith("fal-ai/")

        for i, scene in enumerate(scenes):
            check()
            style = job["style"]
            full_prompt = f"{style}. {scene}" if style else scene
            job["message"] = f"Generating clip {i + 1} of {n}..."
            log(f"Clip {i + 1}/{n}: {scene[:90]}")
            path = OUT / f"{job['id']}_clip{i + 1:02d}.mp4"

            if is_genrelay:
                # --- GENRELAY PATH (Veo 3.1 & Veo 3.1 FL) ---
                if not genrelay_key:
                    raise RuntimeError("No GenRelay API key found. Add GENRELAY_API to .env or Settings.")

                if frame is None:
                    tid = call_genrelay_create_task(
                        genrelay_key, "veo_3_1", full_prompt, job["aspect"], job["resolution"]
                    )
                    v_url = poll_genrelay_task(genrelay_key, tid, check, log, job, i, n)
                    download_url(v_url, path, check)
                else:
                    frame_url = upload_frame_for_chaining(frame, log)
                    try:
                        if frame_url:
                            tid = call_genrelay_create_task(
                                genrelay_key, "veo_3_1-fl", full_prompt, job["aspect"], job["resolution"], frame_url=frame_url
                            )
                        else:
                            tid = call_genrelay_create_task(
                                genrelay_key, "veo_3_1", full_prompt, job["aspect"], job["resolution"]
                            )
                        v_url = poll_genrelay_task(genrelay_key, tid, check, log, job, i, n)
                        download_url(v_url, path, check)
                    except Exception as ex:
                        log(f"Chained FL clip attempt notice: {ex}. Retrying with direct continuity...")
                        tid = call_genrelay_create_task(
                            genrelay_key, "veo_3_1", f"Continuing smoothly from previous shot: {full_prompt}", job["aspect"], job["resolution"]
                        )
                        v_url = poll_genrelay_task(genrelay_key, tid, check, log, job, i, n)
                        download_url(v_url, path, check)

            elif is_fal:
                # --- FAL.AI PATH (MiniMax, Kling, Luma) ---
                if not fal_key:
                    raise RuntimeError("No Fal.ai API key found. Add FAL.AI_API or FAL_KEY to .env or Settings.")

                frame_url = upload_frame_for_chaining(frame, log) if frame else None
                v_url = call_fal_video(
                    fal_key, job["model"], full_prompt, job["aspect"],
                    frame_url=frame_url, check_fn=check, log_fn=log, job=job, clip_idx=i, total_clips=n
                )
                download_url(v_url, path, check)

            else:
                # --- GOOGLE DIRECT PATH (Gemini Omni 1.1 Flash) ---
                if not gemini_key:
                    raise RuntimeError("No Google Gemini API key found. Add GEMINI_API_KEY to .env or Settings.")
                client = genai.Client(api_key=gemini_key, http_options=types.HttpOptions(timeout=180_000))
                if not hasattr(client, "interactions"):
                    raise RuntimeError("Your google-genai package is too old. Run: pip install -U google-genai")

                if frame is None:
                    parts = full_prompt
                else:
                    b64 = base64.b64encode(frame.read_bytes()).decode("utf-8")
                    parts = [
                        {"type": "image", "data": b64, "mime_type": "image/jpeg"},
                        {
                            "type": "text",
                            "text": (
                                "Use the attached image as the exact first frame of the video and "
                                "continue the action seamlessly, keeping characters and art style consistent. "
                                + full_prompt
                            ),
                        },
                    ]

                def call(p=parts):
                    return client.interactions.create(
                        model=job["model"],
                        input=p,
                        response_format={
                            "type": "video",
                            "aspect_ratio": job["aspect"],
                            "resolution": job["resolution"],
                            "delivery": "uri",
                        },
                    )

                interaction = with_retry(call, log, check_fn=check)
                out = getattr(interaction, "output_video", None)
                if out is None:
                    raise RuntimeError("No video returned (blocked by safety filter or refused by model)")
                save_omni_video(client, out, path, check)

            # Process clip
            d = duration(path)
            clips.append(path)
            total_secs += d
            job["done_steps"] = i + 1
            join(clips, checkpoint)
            job["video_name"] = checkpoint.name
            log(f"Clip {i + 1} done ({d:.1f}s). Total runtime so far: {total_secs:.0f}s")

            if total_secs >= target or i == len(scenes) - 1:
                break

            # Extract last frame for seamless continuity in next shot
            frame = OUT / f"{job['id']}_frame.jpg"
            last_frame(path, frame)

        final = OUT / f"video_{job['id']}.mp4"
        shutil.copy(checkpoint, final)
        job["video_name"] = final.name
        job["done_steps"] = job["total"]
        job["status"] = "done"
        job["message"] = f"Done - {total_secs:.0f}s video ready"
        log("Video generation finished successfully!")

    except Cancelled:
        job["status"] = "cancelled"
        job["message"] = "Cancelled by user" + (" (partial video saved)" if job.get("video_name") else "")
        log("Job stopped (cancelled)")
    except Exception as e:
        msg = friendly(str(e))
        job["status"] = "error"
        job["message"] = f"Failed: {msg[:1200]}" + (" (partial video saved)" if job.get("video_name") else "")
        log("ERROR: " + msg[:2000])


def public(job):
    d = {k: v for k, v in job.items() if k not in ("cancel",)}
    d["video_url"] = f"/videos/{job['video_name']}" if job.get("video_name") else None
    return d


# ---------- Web Routes ----------
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/config")
def config():
    has_gemini = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    has_genrelay = bool(os.environ.get("GENRELAY_API") or os.environ.get("GENRELAY_API_KEY"))
    has_fal = bool(os.environ.get("FAL_KEY") or os.environ.get("FAL.AI_API") or os.environ.get("FAL_API_KEY"))
    return jsonify(
        has_key=has_gemini or has_genrelay or has_fal,
        has_gemini=has_gemini,
        has_genrelay=has_genrelay,
        has_fal=has_fal,
        model=DEFAULT_VIDEO_MODEL,
        planner=DEFAULT_PLANNER_MODEL,
    )


@app.get("/api/jobs")
def list_jobs():
    return jsonify([public(j) for j in sorted(JOBS.values(), key=lambda j: j["created"])])


@app.get("/api/jobs/<jid>")
def get_job(jid):
    if jid not in JOBS:
        abort(404)
    return jsonify(public(JOBS[jid]))


@app.post("/api/jobs/<jid>/cancel")
def cancel(jid):
    if jid not in JOBS:
        abort(404)
    job = JOBS[jid]
    job["cancel"] = True
    job["status"] = "cancelled"
    job["message"] = "Cancelled by user" + (" (partial video saved)" if job.get("video_name") else "")
    job["log"].append(time.strftime("%H:%M:%S  ") + "Cancelled by user")
    return jsonify(ok=True)


@app.post("/api/jobs")
def create_job():
    data = request.get_json(force=True, silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify(error="Please write a prompt first"), 400

    gemini_key = (
        (data.get("gemini_key") or data.get("api_key") or "").strip()
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or ""
    )
    genrelay_key = (
        (data.get("genrelay_key") or "").strip()
        or os.environ.get("GENRELAY_API")
        or os.environ.get("GENRELAY_API_KEY")
        or ""
    )
    fal_key = (
        (data.get("fal_key") or "").strip()
        or os.environ.get("FAL_KEY")
        or os.environ.get("FAL.AI_API")
        or os.environ.get("FAL_API_KEY")
        or ""
    )

    model = (data.get("model") or DEFAULT_VIDEO_MODEL).strip()
    if model.startswith("veo_") and not genrelay_key:
        return jsonify(error="Veo 3.1 requires a GenRelay API key. Add GENRELAY_API to your .env or Settings."), 400
    if model.startswith("fal-ai/") and not fal_key:
        return jsonify(error="This model requires a Fal.ai API key. Add FAL.AI_API to .env or Settings."), 400
    if not model.startswith("veo_") and not model.startswith("fal-ai/") and not gemini_key:
        return jsonify(error="Gemini Omni requires a GEMINI_API_KEY. Add it to .env or Settings."), 400

    if any(j["status"] == "running" for j in JOBS.values()):
        return jsonify(error="A video is already generating. Cancel it or wait for it to finish."), 409

    try:
        seconds = int(data.get("seconds", 60))
    except (TypeError, ValueError):
        seconds = 60
    seconds = max(8, min(seconds, MAX_SECONDS))

    jid = uuid.uuid4().hex[:8]
    job = {
        "id": jid,
        "created": time.time(),
        "prompt": prompt,
        "seconds": seconds,
        "aspect": data.get("aspect") if data.get("aspect") in ("16:9", "9:16") else "16:9",
        "resolution": data.get("resolution") if data.get("resolution") in ("360p", "720p", "1080p") else "720p",
        "model": model,
        "planner": (data.get("planner") or DEFAULT_PLANNER_MODEL).strip(),
        "style": (data.get("style") or "").strip(),
        "total": clips_for(seconds),
        "done_steps": 0,
        "status": "running",
        "message": "Starting video generation...",
        "log": [],
        "scenes": [],
        "video_name": None,
        "cancel": False,
    }
    JOBS[jid] = job
    threading.Thread(target=run_job, args=(job, gemini_key, genrelay_key, fal_key), daemon=True).start()
    return jsonify(public(job))


@app.get("/videos/<path:name>")
def videos(name):
    return send_from_directory(OUT, name)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"\n  Veo Studio running at http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, threaded=True)
