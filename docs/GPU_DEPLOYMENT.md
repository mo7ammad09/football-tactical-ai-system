# GPU Deployment Strategy (for Full Match Upload via UI)

## What you want
- Open Streamlit UI on your local machine.
- Upload full match video.
- Optionally upload/select YOLO model (`.pt`).
- Run analysis on strong remote GPU.

## Correct architecture
- **Local**: Streamlit UI (`Home.py`).
- **Remote GPU**: FastAPI server (`server/main.py`) deployed on GPU VM/Pod.
- UI sends video + optional model to remote server `/upload`.

## New upload mode (recommended)
- The system now supports **Object Storage multipart upload** with:
  - Parallel parts
  - Resume support
  - Retry/backoff
- Flow:
  1. UI uploads video/model to object storage via signed URLs.
  2. GPU server downloads from storage and starts analysis.
  3. This avoids common proxy upload failures/timeouts for large files.

## H100 note (important)
- You are not wrong: H100 is powerful.
- But for this pipeline, H100 can be **overkill/costly** unless you need very high throughput.
- Good cost-performance starting point: `RTX 4090` or `L40S`.
- Use H100 when you run many matches in parallel or strict SLA.

## Platform recommendation
1. **RunPod Pods (recommended for your use-case)**
   - Persistent server, easier large uploads, longer jobs.
2. RunPod Serverless
   - Better for short stateless jobs, less ideal for very large uploads.

## Deploy commands (on GPU host)
```bash
cd /app/Football-Tactical-AI-System
pip install -r server/requirements.txt
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

## Use from local UI
- In sidebar:
  - `Processing Mode = Remote GPU`
  - `Server URL = http://<GPU_IP>:8000`
- Upload video + choose model path or upload `.pt` model.

## Object storage environment variables (on GPU server)
Set these on the pod/endpoint before starting `server.main`:

```bash
OBJECT_STORAGE_BUCKET=your-bucket
OBJECT_STORAGE_REGION=us-east-1
OBJECT_STORAGE_ENDPOINT_URL=https://<s3-compatible-endpoint>
OBJECT_STORAGE_ACCESS_KEY_ID=<key>
OBJECT_STORAGE_SECRET_ACCESS_KEY=<secret>
OBJECT_STORAGE_PREFIX=football-ai
OBJECT_STORAGE_URL_EXPIRES=3600
```

Optional:
```bash
OBJECT_STORAGE_PUBLIC_BASE_URL=https://public-cdn.example.com
```
