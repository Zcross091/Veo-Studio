# Veo Studio

A ChatGPT-style local web UI that turns a prompt into a 1–3 minute animated video using **Google Veo 3.1** (via GenRelay) and **Google Gemini Direct** (`gemini-omni-1.1-flash`).

## Features

- **Google Veo 3.1 & Veo 3.1 FL Chaining:** True shot-to-shot character & visual consistency using first/last-frame image referencing.
- **AI Scene Planning:** Direct film director scene planning with `gemini-3.8-flash`.
- **Instant Cancellation:** Interrupt running generations at any second without freezing the server.
- **Partial Video Preserved:** Even if you cancel or stop early, all completed clips are stitched together and playable in your browser.
- **Dual Engine:** Use GenRelay for Veo 3.1 (no Google Cloud billing setup needed) or Google Direct for Gemini Omni.

## Quick Start

1. Install Python 3.10 or newer (on Windows, ensure "Add Python to PATH" is checked).
2. Install dependencies:
   ```bash
   pip install -U -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your keys:
   ```env
   GENRELAY_API="sk-..."
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
2. **Shot Chaining:** Veo 3.1 creates the opening shot. The last frame of each shot is automatically extracted and passed to the next shot with `veo_3_1-fl`, keeping character appearances identical across cuts.
3. **Stitching:** All generated clips are seamlessly combined using FFmpeg into a final high-definition MP4.
