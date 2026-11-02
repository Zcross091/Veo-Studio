# Veo Studio

A ChatGPT-style local web UI that turns a single prompt into a 1–3 minute animated video using frontier AI video models:
- **Google Veo 3.1 & Veo 3.1 FL** (via GenRelay)
- **MiniMax Video-01, Kling 1.0 & Luma Dream Machine** (via Fal.ai)
- **Gemini Omni 1.1 Flash** (via Google Direct Interactions API)

## Features

- **Multi-Engine Support:** Choose between Google Veo 3.1, MiniMax Video-01, Kling, and Luma with seamless first/last-frame shot chaining.
- **AI Scene Planning:** Direct film director scene breakdown using `gemini-3.5-flash-lite` and `gemini-3.8-flash`.
- **Instant Cancellation:** Interrupt generation at any second without freezing the queue.
- **Partial Video Saved:** Every completed clip is joined immediately into an MP4 and previewable in real-time.

## 1-Click Quick Start

On Windows, simply double-click:
```cmd
run.bat
```
*(Or run `python run.py` from your terminal).*

It automatically:
1. Creates and activates the Python virtual environment (`.venv`).
2. Installs and verifies all dependencies.
3. Prepares the OpenArt CLI tool.
4. Starts the web server (accessible from both your PC and your phone).
5. Opens your web browser automatically to the studio UI!

## How it works

1. **Scene Planning:** Gemini splits your idea into narrative shots of ~8-10 seconds each, maintaining character and environmental consistency.
2. **Shot Chaining:** The opening shot is generated from text. The last frame of each shot is automatically extracted and passed to the next shot, keeping character appearances identical across cuts.
3. **Stitching:** All generated clips are seamlessly combined using FFmpeg into a final high-definition MP4.
