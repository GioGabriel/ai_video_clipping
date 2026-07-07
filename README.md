# AI Clipping System

Modular local-first backend for converting long-form videos into short vertical clips with Whisper, Ollama, FFmpeg, and FastAPI.

## Architecture

```text
User
  |
  v
FastAPI API
  |
  v
Processing Job Queue
  |
  v
Video Processing Pipeline
  |- Video Downloader
  |- Audio Extractor
  |- Speech Transcriber
  |- Viral Moment Detector
  |- Clip Generator
  |- Subtitle Generator
  `- Export Manager
```

## Project Structure

```text
ai-clipping-system/
├── data/
│   ├── audio/
│   ├── clips/
│   ├── transcripts/
│   └── videos/
├── src/
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── core/
│   │   ├── command_runner.py
│   │   ├── config.py
│   │   ├── container.py
│   │   ├── database.py
│   │   ├── logging.py
│   │   ├── models.py
│   │   ├── queue.py
│   │   ├── repositories.py
│   │   └── timecode.py
│   ├── modules/
│   │   ├── audio_extractor/
│   │   ├── clip_generator/
│   │   ├── downloader/
│   │   ├── export_manager/
│   │   ├── subtitle_generator/
│   │   ├── transcription/
│   │   └── viral_detector/
│   ├── pipeline/
│   │   └── process_video.py
│   └── main.py
├── tests/
├── .env.example
├── pyproject.toml
└── README.md
```

## Features

- FastAPI endpoints for submitting jobs, checking status, and listing clips
- SQLite-backed job and clip tracking
- Background worker queue for offline processing
- Local video download with `yt-dlp`
- Local audio extraction and clip rendering with `ffmpeg`
- Local transcription with Whisper
- Local viral moment detection with Ollama + Llama 3
- Themeable kinetic captions with `tiktok`, `cinematic`, and `motivational` vibes
- Word-timed sidecar `.srt` and styled `.ass` generation per clip
- Export manifest for downstream tooling

## Setup

1. Create a Python 3.11+ virtual environment.
2. Install Python dependencies:

```bash
pip install -e .
```

3. Install system tools locally:

```bash
brew install ffmpeg yt-dlp ollama
```

4. Pull the local Llama model and start Ollama:

```bash
ollama pull llama3
```

5. Copy the environment file if you need overrides:

```bash
cp .env.example .env
```

If `yt-dlp` fails with a TLS certificate verification error on a trusted network that intercepts SSL, you can opt in to:

```text
YT_DLP_SKIP_CERTIFICATE_CHECK=true
```

This makes the downloader pass `--no-check-certificates`. Leave it `false` unless you understand the trust tradeoff.

6. Use the helper scripts to start the local stack:

```bash
./scripts/dev-start.sh
```
cd /Users/giogabrielsanchez/Ai_Video_Clipping
source .venv/bin/activate
./scripts/dev-start.sh

tail -f data/logs/api.log data/logs/ollama.log

Clean stop:

./scripts/dev-stop.sh

Force-stop anything still holding the ports:

./scripts/dev-stop.sh --all



This starts:

- `ollama serve`
- `uvicorn src.main:app --host APP_HOST --port APP_PORT`

If you want to run the API manually instead, you can still use:

```bash
ollama serve
uvicorn src.main:app --reload
```

## API

### Local dashboard

Open the browser UI at:

```text
http://127.0.0.1:8000/
```

The dashboard lets you:

- submit a new video URL
- track the current job
- browse recent jobs
- preview generated clips in-browser

## Runbook

### Start everything

```bash
./scripts/dev-start.sh
```

This script:

- loads values from `.env` if present
- starts Ollama if port `11434` is not already in use
- starts FastAPI if port `8000` is not already in use
- writes pid files to `data/runtime/`
- writes logs to `data/logs/`

### Check what is running

```bash
./scripts/dev-status.sh
```

This shows whether Ollama and FastAPI are running, and whether they were started by these helper scripts or by some external process.

### Stop the app cleanly

```bash
./scripts/dev-stop.sh
```

This stops the FastAPI and Ollama processes recorded in `data/runtime/api.pid` and `data/runtime/ollama.pid`.

### Force-stop listeners on the configured ports

```bash
./scripts/dev-stop.sh --all
```

Use this if:

- `uvicorn` was started manually
- the pid files are stale
- the dashboard says something is still listening on port `8000` or `11434`

`--all` is the stronger cleanup option. It stops any listener on `APP_PORT` and the Ollama port from `.env`.

### Watch logs

```bash
tail -f data/logs/api.log data/logs/ollama.log
```

### Run tests

```bash
./scripts/test.sh
```

This uses the project virtualenv directly so the suite runs against the same Python environment as the app.

### Important queue behavior

The current worker queue is in-process. If you kill FastAPI while a job is running:

- the active job will stop immediately
- on the next startup, stale `queued` or `running` jobs are reconciled to `failed`
- completed and failed jobs can be deleted from the dashboard to reclaim disk space

There is not yet a true per-job cancel/resume flow for active work. `dev-stop.sh --all` stops the processes, not just a single job.

### Submit a video

```http
POST /process-video
Content-Type: application/json

{
  "url": "https://youtube.com/watch?v=...",
  "caption_theme": "tiktok"
}
```

### Check job status

```http
GET /status/{job_id}
```

### List generated clips

```http
GET /clips/{video_id}
```

### List recent jobs

```http
GET /jobs
```

## Pipeline Outputs

- Downloaded source video: `data/videos/{video_id}.mp4`
- Extracted audio: `data/audio/{video_id}.wav`
- Transcript JSON: `data/transcripts/{video_id}.json`
- Generated clips: `data/clips/{video_id}/clip_001.mp4`
- Generated subtitles: `data/clips/{video_id}/subtitles/clip_001.srt`
- Export manifest: `data/clips/{video_id}/manifest.json`

## FFmpeg Commands Used

### Audio extraction

```bash
ffmpeg -y -i input.mp4 -vn -ac 1 -ar 16000 -c:a pcm_s16le output.wav
```

### Vertical clip rendering

```bash
ffmpeg -y -ss START -i input.mp4 -t DURATION \
  -vf "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920" \
  -c:v libx264 -preset medium -crf 20 \
  -c:a aac -b:a 192k -movflags +faststart clip_001.mp4
```

The current framing strategy uses center crop. Face tracking and automatic reframing can replace this module later without changing the rest of the pipeline.

## SQLite Schema

The schema lives in [`src/core/database.py`](/Users/giogabrielsanchez/Ai_Video_Clipping/src/core/database.py). It includes:

- `jobs` for queue state, pipeline step, file paths, timestamps, and failure details
- `clips` for exported assets, source timestamps, scores, and subtitle sidecars

## Extension Points

- Replace the in-process queue with Redis or a distributed worker later
- Add face tracking before `ClipGenerator`
- Burn styled captions into clips after `SubtitleGenerator`
- Add thumbnail generation or social publishing after `ExportManager`
- Scale to batch jobs by adding collection endpoints and queue fan-out
