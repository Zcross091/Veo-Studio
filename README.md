# Veo Studio (`free-web-providers`)

A ChatGPT-style local web UI and Universal REST API that turns a single prompt into a 1–3 minute animated video using frontier AI video models and free-tier compute:
- **Universal Free-Tier Video Gateway (`bridge.py`)**: A single REST API aggregating 7 free and freemium video generation providers with automatic failover!
- **Google Veo 3.1 & Veo 3.1 FL** (via GenRelay)
- **MiniMax Video-01, Kling 1.0 & Luma Dream Machine** (via Fal.ai or Free Bridge)
- **Gemini Omni 1.1 Flash** (via Google Direct Interactions API)

---

## ⚡ Universal Free-Tier Hub Providers

This branch exposes a **Single REST API endpoint** (`POST /api/v1/video/generate` on port `5050` or within `app.py`) with intelligent **Auto-Cascading** (if one provider runs out of free credits, it automatically cascades to the next available free service):

| Provider | Free Allowance | Auth Setup |
| :--- | :--- | :--- |
| **OpenArt AI** | 40 tokens / month | Built-in CLI (`openart login` or `OPENART_TOKEN`) |
| **Kaggle GPU** | 30 hours / week NVIDIA T4/P100 GPU | `KAGGLE_API_TOKEN` & `KAGGLE_USERNAME` in `.env` |
| **Hugging Face ZeroGPU** | 100% Free Community GPU (LTX-Video / CogVideoX) | Free `HF_TOKEN` from huggingface.co |
| **Kling AI** | ~66 free daily credits | `KLING_ACCESS_KEY` & `KLING_SECRET_KEY` in `.env` |
| **Luma Dream Machine** | ~30 free renders / month | `LUMA_API_KEY` in `.env` |
| **Hailuo / MiniMax** | Video-01 Free Allowance | `MINIMAX_API_KEY` in `.env` |
| **Pollinations AI** | Zero-cost starter tier | `POLLINATIONS_KEY` in `.env` |

---

## 🚀 1-Click Quick Start

On Windows, simply double-click:
```cmd
run.bat
```
*(Or run `python run.py` from your terminal).*

It automatically:
1. Creates and activates the Python virtual environment (`.venv`).
2. Installs and verifies all dependencies (Flask, Kaggle SDK, Gradio Client, FFmpeg).
3. Auto-configures Kaggle API authentication (`~/.kaggle/access_token`).
4. Launches the **Universal REST API Hub** (`http://127.0.0.1:5050`).
5. Starts the **Studio Web Interface** (`http://127.0.0.1:5000` & mobile Wi-Fi link).
6. Opens your browser automatically!

---

## REST API Reference

Generate a video through the unified gateway:
```bash
curl -X POST http://127.0.0.1:5050/api/v1/video/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Cyberpunk city at midnight with neon rain",
    "provider": "auto",
    "aspect": "16:9",
    "duration": 5
  }'
```

Check status and retrieve video:
```bash
curl http://127.0.0.1:5050/api/v1/video/status/<task_id>
```

---

## How it works

1. **Scene Planning:** Gemini (`gemini-3.5-flash-lite`) splits your prompt into narrative shots of ~8-10 seconds each, maintaining character and environmental consistency.
2. **Shot Chaining:** The opening shot is generated from text. The last frame of each shot is automatically extracted and passed to the next shot, keeping character appearances identical across cuts.
3. **Stitching:** All generated clips are seamlessly combined using FFmpeg into a final high-definition MP4.
