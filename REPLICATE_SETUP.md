# 🚀 دليل إعداد Replicate الكامل

## الخطوة 1: إنشاء حساب Replicate

1. ادخل على [replicate.com](https://replicate.com)
2. سجل حساب جديد (مجاني)
3. اذهب لـ [Account Settings](https://replicate.com/account)
4. انسخ الـ API Token

## الخطوة 2: تثبيت Cog

```bash
# على جهازك (Mac)
pip install cog

# تأكد من التثبيت
cog --version
```

## الخطوة 3: تسجيل الدخول

```bash
cog login
# الصق الـ API Token لما يطلبه
```

## الخطوة 4: تدريب النموذج (قبل الرفع)

### 4.1 جهز البيانات

```bash
# هيكل المجلدات المطلوب
dataset/
├── images/
│   ├── train/          ← صور التدريب
│   ├── val/            ← صور التحقق
│   └── test/           ← صور الاختبار
└── labels/
    ├── train/          ← ملفات YOLO labels
    ├── val/
    └── test/
```

### 4.2 سوي labeling للصور

استخدم أحد هذه الأدوات:
- [Roboflow](https://roboflow.com) (أسهل)
- [CVAT](https://cvat.org)
- [LabelImg](https://github.com/tzutalin/labelImg)

**الأنواع المطلوبة:**
- `0`: player (لاعب)
- `1`: goalkeeper (حارس)
- `2`: ball (كرة)
- `3`: referee (حكم)

### 4.3 درب النموذج

```bash
cd replicate_model

# إنشاء data.yaml تلقائياً
python train.py --data /path/to/dataset --model m --epochs 100

# أو لو عندك data.yaml جاهز
python train.py --data data.yaml --model m --epochs 100 --batch 16
```

**خيارات النموذج:**
| الحجم | السرعة | الدقة | الاستخدام |
|-------|--------|-------|----------|
| `n` | سريع | أقل | تجريبي |
| `s` | سريع | متوسطة | سريع |
| `m` | متوسط | جيدة | **موصى به** |
| `l` | بطيء | عالية | دقيق |
| `x` | بطيء | الأعلى | أقصى دقة |

### 4.4 اختبر النموذج

```bash
# النموذج يتحفظ في runs/train/football_analysis/weights/best.pt
# جربه على فيديو
python -c "
from ultralytics import YOLO
model = YOLO('runs/train/football_analysis/weights/best.pt')
results = model('test_video.mp4', save=True)
"
```

## الخطوة 5: انسخ النموذج للمجلد

```bash
# انسخ النموذج المدرب
mkdir -p replicate_model/models
cp runs/train/football_analysis/weights/best.pt replicate_model/models/
```

## الخطوة 6: ارفع على Replicate

```bash
cd replicate_model

# ارفع النموذج
cog push r8.im/YOUR_USERNAME/football-tactical-ai

# مثال:
# cog push r8.im/ahmedkhalid/football-tactical-ai
```

**النتيجة:**
```
Building Docker image...
Pushing to Replicate...
✅ Successfully pushed to r8.im/YOUR_USERNAME/football-tactical-ai
```

## الخطوة 7: جرب النموذج على Replicate

### من المتصفح:
1. ادخل على `https://replicate.com/YOUR_USERNAME/football-tactical-ai`
2. ارفع فيديو
3. اضغط Run

### من الكود:
```python
import replicate

output = replicate.run(
    "YOUR_USERNAME/football-tactical-ai",
    input={"video": open("match.mp4", "rb")}
)

print(output)
```

## الخطوة 8: ربطه بالواجهة

### 8.1 حدّث `secrets.toml`

```toml
# .streamlit/secrets.toml
REPLICATE_API_TOKEN = "r8_xxxxxxxxxxxxxxxx"
REPLICATE_MODEL_VERSION = "YOUR_USERNAME/football-tactical-ai:latest"
```

### 8.2 حدّث `replicate_client.py`

```python
# في ملف src/api/replicate_client.py
# غير هذا السطر:
self.model_version = model_version or "YOUR_USERNAME/football-tactical-ai:latest"
```

### 8.3 شغل الواجهة

```bash
streamlit run Home.py
```

## 💰 التكلفة

### Replicate Pricing

| الاستخدام | التكلفة |
|-----------|---------|
| التسجيل | مجاني |
| التدريب | مجاني (على جهازك) |
| الرفع | مجاني |
| كل prediction | ~$0.01-0.20 |

### حساب تقديري

| عدد المباريات/الشهر | التكلفة الشهرية |
|--------------------|----------------|
| 10 مباريات | ~$1-3 |
| 50 مباراة | ~$5-15 |
| 100 مباراة | ~$10-30 |

## 🔧 استكشاف الأخطاء

### مشكلة: "Model not found"
```bash
# الحل: تأكد من اسم المستخدم والنموذج
cog push r8.im/YOUR_USERNAME/MODEL_NAME
```

### مشكلة: "Out of memory"
```bash
# الحل: استخدم نموذج أصغر
python train.py --model s  # بدل m
```

### مشكلة: "Video too long"
```bash
# الحل: قسّم الفيديو أو استخدم دقة أقل
# في predict.py غير image_size
```

## 📋 قائمة مراجعة

قبل الرفع، تأكد من:

- [ ] النموذج مدرب ومجرب
- [ ] `best.pt` موجود في `replicate_model/models/`
- [ ] `cog.yaml` محدث
- [ ] `predict.py` يشتغل محلياً
- [ ] حساب Replicate جاهز
- [ ] API Token نسخته
- [ ] Cog مثبت

## 🎯 الخطوات الجاية (بعد ما يشتغل Replicate)

1. ✅ جرب على فيديو حقيقي
2. ✅ اضبط الإعدادات (confidence, إلخ)
3. ✅ أضف Tactic Patterns جديدة
4. ✅ ربط AI (Gemini/OpenAI)
5. ✅ تخصيص الواجهة

---

## 📞 دعم

- [Replicate Docs](https://replicate.com/docs)
- [Cog Docs](https://github.com/replicate/cog)
- [Ultralytics Docs](https://docs.ultralytics.com)
