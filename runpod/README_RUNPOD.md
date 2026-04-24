# RunPod GPU Deployment (Production Path)

هذا المسار مخصص لتحليل فيديوهات طويلة (حتى 90 دقيقة) من جهازك الضعيف عبر GPU خارجي.

## 1) Build image

من داخل مجلد المشروع:

```bash
cd runpod
docker build -t YOUR_DOCKERHUB_USER/football-tactical-ai:latest .
```

## 2) Push image

```bash
docker login
docker push YOUR_DOCKERHUB_USER/football-tactical-ai:latest
```

## 3) Create RunPod Serverless Endpoint

- Endpoint Type: `RunPod Worker`
- Container Image: `YOUR_DOCKERHUB_USER/football-tactical-ai:latest`
- GPU: `RTX 4090` أو أعلى
- Execution Timeout: `7200` (ساعتين)
- FlashBoot: `Enabled`

## 4) Run inference

```python
import requests

endpoint_id = "YOUR_ENDPOINT_ID"
api_key = "YOUR_RUNPOD_API_KEY"

payload = {
    "input": {
        "video_url": "https://.../match_90min.mp4",
        "analysis_fps": 1.0,
        "max_frames": 5400,
        "resize_width": 960,
        "detect_passes": True,
        "detect_shots": True,
        "output_upload_url": "https://..."  # optional pre-signed PUT URL
    }
}

resp = requests.post(
    f"https://api.runpod.ai/v2/{endpoint_id}/run",
    headers={"Authorization": f"Bearer {api_key}"},
    json=payload,
    timeout=60,
)
print(resp.json())
```

## Notes

- `output_upload_url` اختياري، لكنه عملياً ضروري إذا تبغى تحتفظ بالفيديو الناتج.
- بدون `output_upload_url` بيرجع لك JSON stats والتحليل فقط، والفيديو يبقى داخل worker بشكل مؤقت.
- إعدادات `analysis_fps + resize_width` هي المفتاح لتشغيل فيديو طويل بدون انفجار RAM.
