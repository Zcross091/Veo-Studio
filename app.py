"""Veo Studio - local web UI that turns one prompt into a 1-3 minute animated video.

Google retired the Veo 3.1 preview models, so this uses gemini-omni-1.1-flash
(Interactions API). Omni makes ~10 second clips and caps its own extension at
40 seconds, so to go longer we chain clips: the last frame of each clip is fed
to the next one as its first frame, then all clips are joined with ffmpeg.
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
import uuid
from pathlib import Path

import imageio_ffmpeg
from flask import Flask, abort, jsonify, request, send_from_directory
from google import genai
from google.genai import types

BASE = Path(__file__).parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

DEFAULT_VIDEO_MODEL = "gemini-omni-1.1-flash"
DEFAULT_PLANNER_MODEL = "gemini-3.8-flash"
PLANNER_FALLBACKS = ["gemini-3.5-flash-lite", "gemini-2.5-flash"]
CLIP_SECONDS = 9  # planning estimate; real clips are 3-10s and are measured
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
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()

app = Flask(__name__, static_folder="static", static_url_path="/static")
JOBS = {}


class Cancelled(Exception):
    pass


def clips_for(seconds):
    return max(1, min(MAX_CLIPS, math.ceil(seconds / CLIP_SECONDS)))


# ---------- ffmpeg helpers ----------
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


# ---------- Gemini helpers ----------
def with_retry(fn, log, tries=3):
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            zero_quota = "limit: 0" in msg or "free_tier" in msg.lower()
            fatal = zero_quota or (
                any(t in msg for t in ("400", "401", "403", "404", "NOT_FOUND", "PERMISSION",
                                       "API key", "INVALID_ARGUMENT"))
                and "RESOURCE_EXHAUSTED" not in msg
            )
            if attempt == tries or fatal:
                raise
            wait = 30 * attempt if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) else 10 * attempt
            log(f"API error: {msg[:120]} - retrying in {wait}s")
            time.sleep(wait)


def friendly(msg):
    """Add a plain-language hint in front of the raw Google error (the raw error is always kept)."""
    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
        return ("Google refused the request: quota is zero or used up. Video generation does not work on "
                "free-tier API keys. Open Google AI Studio, link a billing account to this key's project, "
                "then try again. Raw error: " + msg)
    if "404" in msg or "NOT_FOUND" in msg:
        return "Model not found (it may have been retired). Change the model name in Settings. Raw error: " + msg
    return msg


def plan_scenes(client, planner_model, idea, n, log):
    """Ask Gemini to split the idea into n consecutive shots (tries a few model names)."""
    ask = (
        f"You are an animation director. Split this video idea into exactly {n} "
        f"consecutive shots of about 10 seconds each. Each shot must continue "
        f"directly from the previous one with the same characters, setting and "
        f"art style, and together they must tell a complete story with a "
        f"beginning, middle and end. Return ONLY a JSON array of {n} strings. "
        f"Each string is a self-contained prompt for a video model describing "
        f"the action, camera movement and mood, briefly re-describing the main "
        f"character so they stay consistent.\n\nIdea: {idea}"
    )
    for model in [planner_model] + [m for m in PLANNER_FALLBACKS if m != planner_model]:
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
                        arr.append(f"Continue the story smoothly toward its ending. {idea}")
                    log(f"Scenes planned with {model}")
                    return arr[:n]
        except Exception as e:
            log(f"Planner {model} failed: {str(e)[:90]}")
    log("Using simple continuation prompts instead of a planner")
    return [idea] + [f"Continue the same story smoothly, moving toward the ending. {idea}"] * (n - 1)


def save_video(client, out, path, check):
    """Save the video from an Interactions response (URI delivery, or inline base64)."""
    uri = getattr(out, "uri", None)
    if uri:
        name = uri.split("/")[-1]
        for _ in range(180):
            check()
            info = client.files.get(name=f"files/{name}")
            state = getattr(info.state, "name", str(info.state))
            if state == "ACTIVE":
                break
            if state == "FAILED":
                raise RuntimeError("Google reported the generated video file as FAILED")
            time.sleep(5)
        try:
            client.files.download(file=uri, destination=str(path))
        except TypeError:
            data = client.files.download(file=uri)
            path.write_bytes(data)
    elif getattr(out, "data", None):
        path.write_bytes(base64.b64decode(out.data))
    else:
        raise RuntimeError("The response had no video data")
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError("Video download produced an empty file")


def run_job(job, api_key):
    def log(m):
        job["log"].append(time.strftime("%H:%M:%S  ") + m)

    def check():
        if job["cancel"]:
            raise Cancelled()

    try:
        client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=900_000))
        if not hasattr(client, "interactions"):
            raise RuntimeError("Your google-genai package is too old. Run: pip install -U google-genai")
        n = job["total"]
        target = job["seconds"]
        job["message"] = "Planning the scenes..."
        log(f"Target ~{target}s -> up to {n} clips at {job['resolution']}")
        scenes = plan_scenes(client, job["planner"], job["prompt"], n, log)
        job["scenes"] = scenes
        check()

        clips, total_secs, frame = [], 0.0, None
        checkpoint = OUT / f"{job['id']}_so_far.mp4"
        for i, scene in enumerate(scenes):
            check()
            style = job["style"]
            text = f"{style}. " if style else ""
            text += f"{scene} The shot is about 10 seconds long."
            job["message"] = f"Generating clip {i + 1} of {n}..."
            log(f"Clip {i + 1}/{n}: {scene[:90]}")

            if frame is None:
                parts = text
            else:
                b64 = base64.b64encode(frame.read_bytes()).decode()
                parts = [
                    {"type": "image", "data": b64, "mime_type": "image/jpeg"},
                    {"type": "text", "text": "Use the attached image as the exact first frame of the video and "
                                             "continue the action from it, keeping the same characters and art "
                                             "style. " + text},
                ]

            def call(parts=parts):
                return client.interactions.create(
                    model=job["model"],
                    input=parts,
                    response_format={"type": "video", "aspect_ratio": job["aspect"],
                                     "resolution": job["resolution"], "delivery": "uri"},
                )

            interaction = with_retry(call, log)
            out = getattr(interaction, "output_video", None)
            if out is None:
                raise RuntimeError("No video returned (blocked by the safety filter, or the prompt was refused)")
            path = OUT / f"{job['id']}_clip{i + 1:02d}.mp4"
            save_video(client, out, path, check)

            d = duration(path)
            clips.append(path)
            total_secs += d
            job["done_steps"] = i + 1
            join(clips, checkpoint)
            job["video_name"] = checkpoint.name  # always the full video so far
            log(f"Clip {i + 1} done ({d:.1f}s). Total so far {total_secs:.0f}s")

            if total_secs >= target or i == len(scenes) - 1:
                break
            frame = OUT / f"{job['id']}_frame.jpg"
            last_frame(path, frame)

        final = OUT / f"video_{job['id']}.mp4"
        shutil.copy(checkpoint, final)
        job["video_name"] = final.name
        job["done_steps"] = job["total"]
        job["status"] = "done"
        job["message"] = f"Done - {total_secs:.0f}s video ready"
        log("Finished")
    except Cancelled:
        job["status"] = "cancelled"
        job["message"] = "Cancelled" + (" - partial video saved" if job.get("video_name") else "")
        log("Cancelled by user")
    except Exception as e:
        msg = friendly(str(e))
        job["status"] = "error"
        job["message"] = f"Failed: {msg[:1200]}" + (" (partial video saved)" if job.get("video_name") else "")
        log("ERROR: " + msg[:2000])


def public(job):
    d = {k: v for k, v in job.items() if k not in ("cancel",)}
    d["video_url"] = f"/videos/{job['video_name']}" if job.get("video_name") else None
    return d


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/config")
def config():
    return jsonify(
        has_key=bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
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
    JOBS[jid]["cancel"] = True
    return jsonify(ok=True)


@app.post("/api/jobs")
def create_job():
    data = request.get_json(force=True, silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify(error="Write a prompt first"), 400
    api_key = (data.get("api_key") or "").strip() or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return jsonify(error="No API key. Put GEMINI_API_KEY in the .env file or paste it in Settings."), 400
    if any(j["status"] == "running" for j in JOBS.values()):
        return jsonify(error="A video is already being generated. Wait for it or cancel it."), 409
    try:
        seconds = int(data.get("seconds", 60))
    except (TypeError, ValueError):
        seconds = 60
    seconds = max(10, min(seconds, MAX_SECONDS))
    jid = uuid.uuid4().hex[:8]
    job = {
        "id": jid,
        "created": time.time(),
        "prompt": prompt,
        "seconds": seconds,
        "aspect": data.get("aspect") if data.get("aspect") in ("16:9", "9:16") else "16:9",
        "resolution": data.get("resolution") if data.get("resolution") in ("360p", "720p", "1080p") else "720p",
        "model": (data.get("model") or DEFAULT_VIDEO_MODEL).strip(),
        "planner": (data.get("planner") or DEFAULT_PLANNER_MODEL).strip(),
        "style": (data.get("style") or "").strip(),
        "total": clips_for(seconds),
        "done_steps": 0,
        "status": "running",
        "message": "Starting...",
        "log": [],
        "scenes": [],
        "video_name": None,
        "cancel": False,
    }
    JOBS[jid] = job
    threading.Thread(target=run_job, args=(job, api_key), daemon=True).start()
    return jsonify(public(job))


@app.get("/videos/<path:name>")
def videos(name):
    return send_from_directory(OUT, name)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"\n  Veo Studio running at http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, threaded=True)
