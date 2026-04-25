# GPU Deployment Strategy (for Full Match Upload via UI)

## What you want
- Open Streamlit UI on your local machine.
- Upload full match video.
- Optionally upload/select YOLO model (`.pt`).
- Run analysis on strong remote GPU.

## Correct architecture
- **Local**: Streamlit UI (`Home.py`).
- **Remote GPU**: FastAPI server (`server/main.py`) deployed on GPU VM/Pod.
- Large-match flow: put the video in local `input_videos/`, select it in the UI, then upload it with resumable multipart upload.
- Small test flow: the Streamlit file uploader remains available for short clips only.

## New upload mode (recommended)
- The system now supports **Object Storage multipart upload** with:
  - Parallel parts
  - Resume support
  - Retry/backoff
- Flow:
  1. UI uploads video/model to object storage via signed URLs.
  2. GPU server downloads from storage and starts analysis.
  3. GPU server processes sampled batches and writes review artifacts.
  4. Output video + JSON/CSV reports are uploaded back to object storage when configured.
  5. This avoids common proxy upload failures/timeouts for large files.

## v1 processing defaults
```bash
analysis_fps=3
resize_width=1280
max_frames=0   # 0 / empty means full match
PROCESSING_BATCH_SIZE=16
GPU=RTX 4090 or L40S
```

The v1 report is conservative by design:
- Passes, shots, formations, distance, and max speed are marked unavailable unless confidence is sufficient.
- Field calibration is automatic-first, but low confidence disables spatial metrics instead of showing fake numbers.
- The annotated output is a lightweight sampled review video, not a broadcast-quality export.

## H100 note (important)
- You are not wrong: H100 is powerful.
- But for this pipeline, H100 can be **overkill/costly** unless you need very high throughput.
- Good cost-performance starting point: `RTX 4090` or `L40S`.
- Use H100 when you run many matches in parallel or strict SLA.

## Platform recommendation
1. **RunPod Serverless (recommended for automatic GPU on demand)**
   - Queue-based `/run` jobs start workers automatically and can scale down to zero.
   - Use `workersMin=0`, `workersMax=1` for the first production version.
2. RunPod Pods
   - Keep as a fallback for manual debugging or long interactive experiments.

## Deploy commands (on GPU host)
For the older Pod/FastAPI path:
```bash
cd /app/Football-Tactical-AI-System
pip install -r server/requirements.txt
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

For the automatic Serverless path:
```bash
docker build --platform linux/amd64 -f runpod/Dockerfile -t YOUR_DOCKERHUB_USER/football-tactical-ai:latest .
docker push YOUR_DOCKERHUB_USER/football-tactical-ai:latest
```

If local Docker storage is constrained, use GitHub Actions:
```bash
ghcr.io/mo7ammad09/football-tactical-ai-runpod:latest
```

## Use from local UI
- In sidebar:
  - `Processing Mode = RunPod Serverless`
  - `RUNPOD_ENDPOINT_ID`
  - `RUNPOD_API_KEY`
- Choose a video from `input_videos/` for large matches.
- Choose model path on server, selected local model, or upload a `.pt` model.

## Object storage environment variables
Set these in the UI host and the RunPod Serverless endpoint:

```bash
OBJECT_STORAGE_BUCKET=your-bucket
OBJECT_STORAGE_REGION=us-east-1
OBJECT_STORAGE_ENDPOINT_URL=https://<s3-compatible-endpoint>
OBJECT_STORAGE_ACCESS_KEY_ID=<key>
OBJECT_STORAGE_SECRET_ACCESS_KEY=<secret>
OBJECT_STORAGE_PREFIX=football-ai
OBJECT_STORAGE_PRESIGN_EXPIRES=86400
```

Optional:
```bash
OBJECT_STORAGE_PUBLIC_BASE_URL=https://public-cdn.example.com
```

## Optional API key
Set the same value on the server and local UI environment:

```bash
GPU_API_KEY=your-secret
REMOTE_GPU_API_KEY=your-secret
```

When configured, every API request must include `Authorization: Bearer <key>`.
