# Veo Studio

A ChatGPT-style page on your own PC that turns one prompt into a 1-3 minute animated video using Google's `gemini-omni-1.1-flash` (the Veo 3.1 preview models were shut down on Oct 22, 2026).

## Run it

1. Install Python 3.10 or newer (python.org). On Windows tick "Add Python to PATH".
2. In a terminal in this folder:
   ```
   pip install -U -r requirements.txt
   ```
   (`-U` matters: the new API needs google-genai 2.19 or newer. A copy of ffmpeg is installed automatically.)
3. Copy `.env.example` to `.env` and paste your Gemini API key after `GEMINI_API_KEY=`.
   (Or paste it into Settings in the page.)
4. `python app.py`, then open http://127.0.0.1:5000

## How it works

- Gemini splits your idea into one prompt per ~10 second shot.
- Omni makes the first clip. For every next clip, the last frame of the previous clip is sent as its first frame, so the picture stays continuous.
- All clips are joined into one MP4. A "so far" video is saved in `outputs/` after every clip, so a failure still leaves you a partial video.

## Things to know

- Cost: each clip is billed to your Google account. Test with 30 seconds at 360p first.
- Time: a few minutes per clip, so 1 minute of video can take 20-30 minutes. Keep the terminal open.
- Clips join with a cut every ~10s, and music or voice can change at the cuts. You can add one soundtrack over the final video in any editor.
- Model names change. If you get "model not found", edit the name in Settings.
- The server only listens on 127.0.0.1, so only your PC can open it.
