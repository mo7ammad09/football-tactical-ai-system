# ابدأ من هنا (بدون تعقيد)

إذا Web Terminal ما يتفعل، استخدم SSH مباشرة من جهازك. هذا أسهل.

## 1) جهّز المتغيرات (مرة واحدة)

- `RUNPOD_HOST`: آي بي أو هوست الـPod
- `RUNPOD_PORT`: بورت SSH الظاهر في RunPod

## 2) ارفع مشروعك الحالي وشغّل السيرفر تلقائيًا (بدون Git)

```bash
cd /Users/momac/Desktop/Football-Tactical-AI-System
RUNPOD_HOST=YOUR_HOST RUNPOD_PORT=YOUR_PORT bash scripts/ship_project_to_runpod.sh
```

## 3) اختبر صحة السيرفر

```bash
RUNPOD_HOST=YOUR_HOST bash scripts/check_runpod_health.sh
```

إذا رجع `healthy`، كمل.

## 4) شغّل الواجهة محليًا واربطها بالـGPU

```bash
cd /Users/momac/Desktop/Football-Tactical-AI-System
REMOTE_GPU_URL=http://YOUR_HOST:8000 bash scripts/start_ui.sh
```

من الواجهة:
- `Processing Mode = Remote GPU`
- `Test Connection`
- ارفع الفيديو
- اختر موديل (مسار أو رفع ملف `.pt`)
- `Start Analysis`

---

بديل (إذا تفضل Git): استخدم
```bash
RUNPOD_HOST=YOUR_HOST RUNPOD_PORT=YOUR_PORT REPO_URL=YOUR_GIT_URL bash scripts/deploy_to_runpod_via_ssh.sh
```
