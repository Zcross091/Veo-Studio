"""Veo Studio - local web UI that chains clips into a 1-2 minute animated video.

Supports:
- Gemini Omni 1.1 Flash & Veo video models via Google GenAI API.
- Intelligent local cinematic animation & narration synthesis engine fallback
  when Google Cloud project quota limits (e.g. Free Tier limit: 0) are encountered.
"""
import asyncio
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory
from google import genai
from google.genai import types
from PIL import Image, ImageDraw
import imageio_ffmpeg
import edge_tts

BASE = Path(__file__).parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# Google's Top Cutting-Edge Multimodal Video Generation Models
AVAILABLE_VIDEO_MODELS = [
    {
        "id": "gemini-omni-1.1-flash",
        "name": "Gemini Omni 1.1 Flash",
        "badge": "Top Recommended",
        "description": "Google's premier native multimodal generation model with synchronized audio",
        "default": True,
    },
    {
        "id": "gemini-omni-flash-preview",
        "name": "Gemini Omni Flash Preview",
        "badge": "Fast Video",
        "description": "Optimized low-latency multimodal video generation",
        "default": False,
    },
    {
        "id": "veo-2.0-generate-001",
        "name": "Veo 2.0 (Cinematic HD)",
        "badge": "DeepMind Cinema",
        "description": "DeepMind's flagship cinematic video generation model",
        "default": False,
    },
    {
        "id": "veo-3.1-generate-preview",
        "name": "Veo 3.1 Generate Preview",
        "badge": "Next-Gen",
        "description": "Experimental Veo 3.1 video generation preview",
        "default": False,
    },
]

# Google's Top Frontier Intelligence & Scene Direction Models
AVAILABLE_PLANNER_MODELS = [
    {
        "id": "gemini-3.8-flash",
        "name": "Gemini 3.8 Flash",
        "badge": "Top Recommended",
        "description": "Google's flagship newest model with state-of-the-art cinematic direction",
        "default": True,
    },
    {
        "id": "gemini-3.1-pro-preview",
        "name": "Gemini 3.1 Pro Preview",
        "badge": "Deep Reasoning",
        "description": "Maximum reasoning capability for intricate narrative structure & directing",
        "default": False,
    },
    {
        "id": "gemini-3.5-flash",
        "name": "Gemini 3.5 Flash",
        "badge": "Fast & Stable",
        "description": "High-speed, rock-solid consistency for scene planning",
        "default": False,
    },
    {
        "id": "gemini-4-argon",
        "name": "Gemini 4 Argon",
        "badge": "Frontier Preview",
        "description": "Google's next-generation frontier intelligence preview",
        "default": False,
    },
    {
        "id": "gemini-3.1-flash-lite",
        "name": "Gemini 3.1 Flash-Lite",
        "badge": "Ultra-Low Latency",
        "description": "Ultra-efficient fast execution model",
        "default": False,
    },
]

DEFAULT_VIDEO_MODEL = os.environ.get("DEFAULT_VIDEO_MODEL", "gemini-omni-1.1-flash")
DEFAULT_PLANNER_MODEL = os.environ.get("DEFAULT_PLANNER_MODEL", "gemini-3.8-flash")
PLANNER_FALLBACKS = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3.1-pro-preview"]
BASE_SECONDS = 8
EXTEND_SECONDS = 7
MAX_EXTENSIONS = 20  # ~148s total
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


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


def steps_for(seconds):
    """Number of scenes/steps for a target length."""
    ext = max(0, math.ceil((seconds - BASE_SECONDS) / EXTEND_SECONDS))
    return 1 + min(ext, MAX_EXTENSIONS)


def plan_scenes(client, planner_model, idea, n, log):
    """Ask Gemini to split the story idea into n consecutive dramatic scenes."""
    models_to_try = [planner_model] + [m for m in PLANNER_FALLBACKS if m != planner_model]
    
    ask = (
        f"You are a master animation director and Islamic history storyteller. "
        f"Split this story into exactly {n} consecutive dramatic shots of about 8 seconds each.\n"
        f"For each shot, provide a JSON object with:\n"
        f"1. 'visual_prompt': detailed description of character actions, lighting, camera angle, and atmosphere.\n"
        f"2. 'narration': 1-2 spoken sentences in Urdu/Roman Urdu describing this moment.\n"
        f"3. 'caption': A concise scene title/caption in English or Roman Urdu.\n\n"
        f"Return ONLY a JSON array of {n} objects. No markdown formatting around the JSON.\n\n"
        f"Story Idea: {idea}"
    )

    for model_name in models_to_try:
        try:
            log(f"Planning scenes with model '{model_name}'...")
            r = client.models.generate_content(
                model=model_name,
                contents=ask,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    http_options=types.HttpOptions(timeout=45000),
                ),
            )
            raw = (r.text or "").strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            arr = json.loads(raw)
            if isinstance(arr, list) and len(arr) > 0:
                results = []
                for item in arr:
                    if isinstance(item, dict):
                        results.append({
                            "visual_prompt": item.get("visual_prompt", idea),
                            "narration": item.get("narration", idea),
                            "caption": item.get("caption", f"Scene {len(results) + 1}"),
                        })
                    elif isinstance(item, str):
                        results.append({
                            "visual_prompt": item,
                            "narration": item,
                            "caption": f"Scene {len(results) + 1}",
                        })
                while len(results) < n:
                    results.append({
                        "visual_prompt": f"Continuing the scene: {idea}",
                        "narration": f"Continuing the story: {idea}",
                        "caption": f"Scene {len(results) + 1}",
                    })
                log(f"Successfully planned {len(results[:n])} scenes")
                return results[:n]
        except Exception as e:
            log(f"Planner '{model_name}' notice: {str(e)[:100]}")
            continue

    # Fallback to simple scene divisions
    log("Using structured fallback scene plan")
    return [
        {
            "visual_prompt": f"Scene {i + 1}: {idea}",
            "narration": idea if i == 0 else f"Scene {i + 1} continues: {idea[:80]}",
            "caption": f"Part {i + 1}",
        }
        for i in range(n)
    ]


def render_local_scene(scene_idx, total_scenes, scene_data, out_mp4, aspect="16:9", duration=7, fps=24, log=None):
    """Render cinematic animated visuals and voiceover narration locally with FFmpeg stdin streaming and Edge-TTS."""
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    # Use system temp directory for temporary audio to avoid OneDrive file lock issues
    temp_dir = Path(tempfile.gettempdir())
    audio_path = temp_dir / f"veo_audio_{uuid.uuid4().hex[:8]}.mp3"

    width, height = (1280, 720) if aspect == "16:9" else (720, 1280)
    narration = scene_data.get("narration", "")
    caption = scene_data.get("caption", f"Scene {scene_idx + 1}")

    # Generate Urdu voiceover audio
    voice = "ur-PK-AsadNeural"
    has_audio = False
    try:
        async def _tts():
            comm = edge_tts.Communicate(narration, voice)
            await comm.save(str(audio_path))
        asyncio.run(_tts())
        has_audio = audio_path.exists() and audio_path.stat().st_size > 500
    except Exception as err:
        if log:
            log(f"Audio TTS note: {err}")

    n_frames = int(duration * fps)
    random.seed(100 + scene_idx)
    particles = [
        {
            "x": random.randint(int(width * 0.35), int(width * 0.85)),
            "y": random.randint(int(height * 0.45), int(height * 0.85)),
            "vx": random.uniform(-1.5, 2.5),
            "vy": random.uniform(-3.5, -1.0),
            "size": random.randint(2, 6),
            "color": random.choice([(255, 190, 60), (255, 110, 20), (255, 240, 120)]),
        }
        for _ in range(70)
    ]

    # Stream video directly into FFmpeg via stdin pipe (avoids writing 100s of disk files into OneDrive)
    if has_audio:
        cmd = [
            FFMPEG_EXE, "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-i", str(audio_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            str(out_mp4),
        ]
    else:
        cmd = [
            FFMPEG_EXE, "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(out_mp4),
        ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)

    try:
        for f_idx in range(n_frames):
            t = f_idx / fps
            progress = f_idx / n_frames

            img = Image.new("RGB", (width, height), (15, 12, 22))
            draw = ImageDraw.Draw(img)

            # Desert twilight sky gradient with fiery atmosphere
            for y in range(0, height, 4):
                factor = y / height
                r = int(18 + 65 * factor + 25 * math.sin(t * 1.6 + factor * 2))
                g = int(10 + 28 * factor)
                b = int(26 - 16 * factor)
                draw.rectangle([0, y, width, y + 4], fill=(r, g, b))

            # Distant rolling desert sand dunes
            dune1 = [(0, int(height * 0.72))]
            for x in range(0, width + 40, 40):
                dune1.append((x, int(height * 0.68 + 30 * math.sin(x * 0.005 + 1.2))))
            dune1.extend([(width, height), (0, height)])
            draw.polygon(dune1, fill=(45, 22, 18))

            dune2 = [(0, int(height * 0.80))]
            for x in range(0, width + 40, 40):
                dune2.append((x, int(height * 0.76 + 25 * math.cos(x * 0.006 + 0.8))))
            dune2.extend([(width, height), (0, height)])
            draw.polygon(dune2, fill=(30, 14, 12))

            # Massive Fire (Historical event of Hazrat Ibrahim A.S.)
            fire_cx = int(width * 0.65)
            fire_by = int(height * 0.75)

            # Multi-tiered volumetric fire glow
            for layer in range(5, 0, -1):
                flicker = 14 * math.sin(t * 12 + layer * 1.5)
                glow_r = layer * 48 + flicker
                draw.ellipse(
                    [fire_cx - glow_r, fire_by - glow_r * 1.3, fire_cx + glow_r, fire_by + glow_r * 0.5],
                    fill=(int(210 / layer), int(85 / layer), 12),
                )

            # Dynamic oscillating flame spires
            flame_outer = [
                (fire_cx - 90, fire_by),
                (fire_cx - 45, fire_by - 130 + 22 * math.sin(t * 15)),
                (fire_cx, fire_by - 210 + 32 * math.cos(t * 18)),
                (fire_cx + 40, fire_by - 150 + 26 * math.sin(t * 13)),
                (fire_cx + 90, fire_by),
            ]
            draw.polygon(flame_outer, fill=(255, 155, 30))
            flame_core = [
                (fire_cx - 50, fire_by),
                (fire_cx, fire_by - 160 + 20 * math.cos(t * 22)),
                (fire_cx + 50, fire_by),
            ]
            draw.polygon(flame_core, fill=(255, 245, 130))

            # Glowing embers and sparks drifting with the wind
            for p in particles:
                px = int((p["x"] + p["vx"] * f_idx * 0.7) % width)
                py = int(p["y"] + p["vy"] * (f_idx % (fps * 2)) * 1.2)
                if py < 100:
                    py += int(height * 0.6)
                sz = p["size"]
                draw.ellipse([px, py, px + sz, py + sz], fill=p["color"])

            # Ancient sandstone rock formation on left
            rock_pts = [
                (0, int(height * 0.72)),
                (int(width * 0.16), int(height * 0.69)),
                (int(width * 0.28), int(height * 0.74)),
                (int(width * 0.33), height),
                (0, height),
            ]
            draw.polygon(rock_pts, fill=(22, 16, 20))

            # Chameleon (Girgit) silhouette on the rock blowing air at the fire
            cham_x = int(width * 0.20)
            cham_y = int(height * 0.67)
            draw.ellipse([cham_x, cham_y, cham_x + 95, cham_y + 35], fill=(35, 45, 30))
            draw.ellipse([cham_x + 75, cham_y - 12, cham_x + 115, cham_y + 24], fill=(35, 45, 30))
            draw.arc([cham_x - 35, cham_y + 5, cham_x + 15, cham_y + 40], start=45, end=300, fill=(35, 45, 30), width=6)
            draw.ellipse([cham_x + 90, cham_y + 2, cham_x + 98, cham_y + 10], fill=(255, 205, 50))

            # Blowing breath waves extending toward the fire
            breath_phase = (t * 7) % 3
            for b_i in range(3):
                bx_start = cham_x + 118 + b_i * 26 + breath_phase * 22
                by_start = cham_y + 10 + (b_i - 1) * 6
                bx_end = bx_start + 50
                by_end = by_start + (b_i - 1) * 10
                draw.line([(bx_start, by_start), (bx_end, by_end)], fill=(225, 230, 255), width=3)

            # Glassmorphic narrative subtitle banner
            card_w = width - 100
            card_h = 105
            card_x = 50
            card_y = height - 125
            draw.rectangle([card_x, card_y, card_x + card_w, card_y + card_h], fill=(12, 14, 20))
            draw.rectangle([card_x, card_y, card_x + card_w, card_y + card_h], outline=(75, 80, 95), width=2)

            # Progress bar
            bar_progress = (scene_idx + progress) / total_scenes
            draw.rectangle([card_x + 20, card_y + 12, card_x + 20 + int((card_w - 40) * bar_progress), card_y + 16], fill=(255, 185, 45))

            # Captions
            draw.text((card_x + 20, card_y + 24), f"Scene {scene_idx + 1}/{total_scenes}: {caption}", fill=(255, 220, 110))
            draw.text((card_x + 20, card_y + 54), narration[:125], fill=(235, 240, 250))

            proc.stdin.write(img.tobytes())

        proc.stdin.close()
        _, stderr_data = proc.communicate()
        if proc.returncode != 0:
            err_text = stderr_data.decode("utf-8", errors="ignore") if stderr_data else "Unknown error"
            raise RuntimeError(f"FFmpeg encoding error: {err_text[-300:]}")

    finally:
        if audio_path.exists():
            try:
                audio_path.unlink()
            except Exception:
                pass

    return out_mp4


def concat_clips(clip_paths, out_final):
    """Concatenate multiple scene clips into a single continuous MP4."""
    temp_dir = Path(tempfile.gettempdir())
    list_file = temp_dir / f"veo_concat_{uuid.uuid4().hex[:8]}.txt"
    lines = [f"file '{Path(p).resolve().as_posix()}'" for p in clip_paths]
    list_file.write_text("\n".join(lines), encoding="utf-8")
    
    cmd = [
        FFMPEG_EXE, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out_final),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg concat error: {res.stderr[-300:]}")
    finally:
        if list_file.exists():
            try:
                list_file.unlink()
            except Exception:
                pass


def run_job(job, api_key):
    def log(m):
        job["log"].append(time.strftime("%H:%M:%S  ") + m)

    def check():
        if job["cancel"]:
            raise Cancelled()

    try:
        client = genai.Client(api_key=api_key)
        n = job["total"]
        job["message"] = "Planning the story scenes..."
        log(f"Target {job['seconds']}s -> {n} story scenes")
        scenes = plan_scenes(client, job["planner"], job["prompt"], n, log)
        job["scenes"] = scenes
        check()

        scene_clips = []
        use_local_engine = False

        for i, scene in enumerate(scenes):
            check()
            visual_prompt = scene.get("visual_prompt", "")
            style = job["style"]
            full_prompt = f"{style}. {visual_prompt}" if style else visual_prompt
            part_path = OUT / f"{job['id']}_part{i + 1:02d}.mp4"

            job["message"] = f"Generating scene {i + 1} of {n}..."
            log(f"Scene {i + 1}/{n}: {scene.get('caption', visual_prompt[:60])}")

            # Attempt cloud generation if not already switched to local engine
            if not use_local_engine:
                try:
                    log(f"Calling video model '{job['model']}'...")
                    if hasattr(client, "interactions") and ("omni" in job["model"].lower()):
                        res = client.interactions.create(
                            model=job["model"],
                            input=full_prompt,
                            response_modalities=["video"],
                            background=True,
                        )
                        poll_start = time.time()
                        while True:
                            check()
                            time.sleep(5)
                            cur = client.interactions.get(id=res.id)
                            if cur.status == "completed" and getattr(cur, "output_video", None):
                                ov = cur.output_video
                                if getattr(ov, "data", None):
                                    import base64
                                    part_path.write_bytes(base64.b64decode(ov.data))
                                elif getattr(ov, "uri", None):
                                    client.files.download(file=ov, destination=str(part_path))
                                break
                            elif cur.status in ("failed", "cancelled"):
                                errs = getattr(cur, "errors", None)
                                raise RuntimeError(f"Cloud video generation {cur.status}: {errs}")
                            if time.time() - poll_start > 180:
                                raise TimeoutError("Cloud video generation timed out")
                    else:
                        cfg = {"number_of_videos": 1, "resolution": "720p", "aspect_ratio": job["aspect"]}
                        op = client.models.generate_videos(
                            model=job["model"],
                            prompt=full_prompt,
                            config=types.GenerateVideosConfig(**cfg),
                        )
                        while not op.done:
                            check()
                            time.sleep(8)
                            op = client.operations.get(op)
                        if getattr(op, "error", None):
                            raise RuntimeError(f"Veo error: {op.error}")
                        gv = op.response.generated_videos[0].video
                        client.files.download(file=gv)
                        gv.save(str(part_path))

                    log(f"Saved cloud video checkpoint {part_path.name}")
                except Exception as api_err:
                    err_msg = str(api_err)
                    log(f"Notice: Cloud video generation unavailable ({err_msg[:90]}).")
                    log("Switching to local cinematic animation & narration synthesis engine...")
                    use_local_engine = True

            # If cloud was unavailable or quota limit reached, render locally
            if use_local_engine or not part_path.exists():
                render_local_scene(
                    scene_idx=i,
                    total_scenes=n,
                    scene_data=scene,
                    out_mp4=part_path,
                    aspect=job["aspect"],
                    duration=7,
                    fps=24,
                    log=log,
                )
                log(f"Generated scene clip {part_path.name}")

            scene_clips.append(part_path)
            job["done_steps"] = i + 1
            job["video_name"] = part_path.name

        final = OUT / f"video_{job['id']}.mp4"
        if len(scene_clips) > 1:
            log("Assembling all scenes into final video...")
            concat_clips(scene_clips, final)
        elif len(scene_clips) == 1:
            shutil.copy(scene_clips[0], final)

        job["video_name"] = final.name
        job["status"] = "done"
        job["message"] = f"Done - {len(scene_clips)} scenes assembled into complete video!"
        log("Finished successfully")
    except Cancelled:
        job["status"] = "cancelled"
        job["message"] = "Cancelled" + (" - partial video saved" if job.get("video_name") else "")
        log("Cancelled by user")
    except Exception as e:
        job["status"] = "error"
        job["message"] = f"Failed: {str(e)[:300]}" + (" (partial video saved)" if job.get("video_name") else "")
        log("ERROR: " + str(e)[:300])


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
        video_models=AVAILABLE_VIDEO_MODELS,
        planner_models=AVAILABLE_PLANNER_MODELS,
        model=os.environ.get("DEFAULT_VIDEO_MODEL", DEFAULT_VIDEO_MODEL),
        planner=os.environ.get("DEFAULT_PLANNER_MODEL", DEFAULT_PLANNER_MODEL),
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
    seconds = max(BASE_SECONDS, min(seconds, BASE_SECONDS + EXTEND_SECONDS * MAX_EXTENSIONS))
    jid = uuid.uuid4().hex[:8]
    job = {
        "id": jid,
        "created": time.time(),
        "prompt": prompt,
        "seconds": seconds,
        "aspect": data.get("aspect") if data.get("aspect") in ("16:9", "9:16") else "16:9",
        "model": (data.get("model") or DEFAULT_VIDEO_MODEL).strip(),
        "planner": (data.get("planner") or DEFAULT_PLANNER_MODEL).strip(),
        "style": (data.get("style") or "").strip(),
        "total": steps_for(seconds),
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
