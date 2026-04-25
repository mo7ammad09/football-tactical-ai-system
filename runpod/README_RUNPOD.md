# RunPod GPU Deployment (Production Path)

هذا المسار مخصص لتحليل فيديوهات طويلة (حتى 90 دقيقة) من جهازك الضعيف عبر GPU خارجي.

## 1) Build image

### Recommended: GitHub Actions

لأن صورة CUDA/PyTorch ثقيلة، الأفضل بناؤها خارج جهازك عبر GitHub Actions:

1. ادفع الكود إلى GitHub.
2. افتح `Actions`.
3. شغل workflow باسم `Build RunPod Worker`.
4. استخدم الصورة الناتجة:

```bash
ghcr.io/mo7ammad09/football-tactical-ai-runpod:latest
```

### Local fallback

من داخل مجلد المشروع:

```bash
docker build --platform linux/amd64 -f runpod/Dockerfile -t YOUR_DOCKERHUB_USER/football-tactical-ai:latest .
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
- Workers Min: `0`
- Workers Max: `1` كبداية
- Idle Timeout: قصير حسب ميزانيتك
- Execution Timeout: `7200` (ساعتين)
- FlashBoot: `Enabled`
- Environment variables:
  - `OBJECT_STORAGE_BUCKET`
  - `OBJECT_STORAGE_REGION`
  - `OBJECT_STORAGE_ENDPOINT_URL`
  - `OBJECT_STORAGE_ACCESS_KEY_ID`
  - `OBJECT_STORAGE_SECRET_ACCESS_KEY`
  - `OBJECT_STORAGE_PREFIX`
  - `OBJECT_STORAGE_PUBLIC_BASE_URL` إذا عندك CDN/public bucket
  - `OBJECT_STORAGE_PRESIGN_EXPIRES=86400` إذا التخزين private وتحتاج روابط مؤقتة
  - `MODEL_PATH=models/abdullah_yolov5.pt` فقط إذا بنيت الموديل داخل الصورة

ملاحظة: الصورة الرسمية لا تضع ملفات `.pt` داخل Docker image. الواجهة سترفع الموديل إلى object storage مع job، والـworker سيحمله عبر `model_object_key`.

## 4) Run inference

```python
import requests

endpoint_id = "YOUR_ENDPOINT_ID"
api_key = "YOUR_RUNPOD_API_KEY"

payload = {
    "input": {
        "video_object_key": "football-ai/video/match_90min.mp4",
        "analysis_fps": 3.0,
        "resize_width": 1280,
        "max_frames": None,
        "model_path": "models/abdullah_yolov5.pt"
    }
}

resp = requests.post(
    f"https://api.runpod.ai/v2/{endpoint_id}/run",
    headers={"Authorization": api_key},
    json=payload,
    timeout=60,
)
print(resp.json())
```

## Notes

- الواجهة ترسل job إلى `/run` وتتابع `/status/{job_id}`.
- الـworker يحمّل الفيديو من object storage ويرفع فيديو المراجعة + JSON/CSV reports لنفس التخزين.
- إعدادات `analysis_fps + resize_width` هي المفتاح لتشغيل فيديو طويل بدون انفجار RAM.
- النتائج المتقدمة غير الموثوقة ترجع `Unavailable` بدلاً من أرقام وهمية.
