# Veo Studio 🎬

A local web application that turns any story prompt into a continuous, multi-scene animated video with voiceover narration using Google's newest multimodal AI models (`gemini-omni-1.1-flash`, `veo-2.0`, `gemini-3.8-flash`).

Built with an intelligent hybrid pipeline:
- **Cloud Video Generation:** Uses Google's newest `gemini-omni-1.1-flash` and `veo-2.0` models via the official Google GenAI Interactions & Video APIs.
- **Resilient Fallback Synthesis:** If cloud video API quotas are restricted (e.g. Free Tier limit 0), it automatically switches to an in-memory cinematic 2D animation and neural voiceover narration engine (`edge-tts` + `imageio-ffmpeg`), guaranteeing that you always receive a completed, high-definition MP4.

---

## Features

- **Top Google Video Models:**
  - `gemini-omni-1.1-flash` *(Top Recommended)* — Native multimodal video generation with synchronized audio.
  - `gemini-omni-flash-preview` — High-speed, low-latency video generation.
  - `veo-2.0-generate-001` — DeepMind's cinematic HD video model.
  - `veo-3.1-generate-preview` — Experimental Veo 3.1 preview.
- **Top Frontier Story Directors / Planners:**
  - `gemini-3.8-flash` *(Top Recommended)* — Advanced cinematic scene planning and dramatic dialogue direction.
  - `gemini-3.1-pro-preview` — Deep reasoning for complex storylines.
  - `gemini-3.5-flash` — High throughput, stable scene planning.
  - `gemini-4-argon` — Next-gen frontier preview.
  - `gemini-3.1-flash-lite` — Low-latency execution.
- **Zero-Lock FFmpeg Streaming:** Frames are piped directly in memory to FFmpeg, preventing cloud storage (e.g. OneDrive) file-locking issues.
- **Live Checkpointing:** Partial checkpoint MP4s are saved in `outputs/` after each scene, and concatenated seamlessly into the final video.
- **Web UI:** ChatGPT-style interface with real-time progress bar, logs, embedded video player, and one-click MP4 download.

---

## Quick Start

### 1. Requirements
* Python 3.10 or newer (on Windows, tick *"Add Python to PATH"* during install).

### 2. Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/Zcross091/Veo-Studio.git
cd Veo-Studio
pip install -r requirements.txt
```

### 3. Configure API Key
Copy `.env.example` to `.env` and add your Gemini API key:
```env
GEMINI_API_KEY=your_api_key_here
```
*(You can also paste the API key directly into the Settings modal in the web UI).*

### 4. Run
Start the application:
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

---

## How It Works

1. **Scene Planning:** The selected Gemini Director model breaks your idea into structured, consecutive scenes with character actions, visual prompts, and localized voiceover narration.
2. **Generation:** Each scene is rendered with video, subtitles, and synchronized audio.
3. **Assembly:** All scene clips are stitched into a continuous video (`outputs/video_<id>.mp4`) ready to watch and download.

---

## License
MIT License
