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

## Quick Start

1. Install Python 3.10 or newer (on Windows, ensure "Add Python to PATH" is checked).
2. Install dependencies:
   ```bash
   pip install -U -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your keys:
   ```env
   GENRELAY_API="sk-..."
   FAL.AI_API="fal_sk_..."
   GEMINI_API_KEY="..."
   ```
   *(Or enter them directly into the Settings panel in the web UI).*
4. Run the server:
   ```bash
   python app.py
   ```
5. Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## How it works

1. **Scene Planning:** Gemini splits your idea into narrative shots of ~8-10 seconds each, maintaining character and environmental consistency.
2. **Shot Chaining:** The opening shot is generated from text. The last frame of each shot is automatically extracted and passed to the next shot, keeping character appearances identical across cuts.
3. **Stitching:** All generated clips are seamlessly combined using FFmpeg into a final high-definition MP4.
