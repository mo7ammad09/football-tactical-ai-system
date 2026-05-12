# 🚀 دليل البدء السريع

## ⚡ في 5 دقائق

### 1. شغل الواجهة (بدون Replicate)

```bash
cd /Users/momac/Downloads/Football-Tactical-AI-System

# نصب المتطلبات
pip install streamlit requests matplotlib pandas

# شغل الواجهة
streamlit run Home.py
```

**يفتح على:** http://localhost:8501

### 2. جرب الوضع التجريبي (Mock)

في الواجهة:
1. اختر **"Mock (No API)"** من إعدادات AI
2. ارفع أي فيديو (حتى فيديو قصير)
3. اضغط **"بدء التحليل"**
4. استمتع بالنتائج التجريبية!

---

## 🔧 في 30 دقيقة (ربط Replicate)

### 1. سجل في Replicate
- ادخل [replicate.com](https://replicate.com)
- سجل حساب
- انسخ API Token

### 2. حدّث الإعدادات

```bash
# افتح الملف
nano .streamlit/secrets.toml
```

غير هذي الأسطر:
```toml
REPLICATE_API_TOKEN = "r8_xxxxxxxxxxxxxxxx"  ← حط توكنك
REPLICATE_MODEL_VERSION = "your-username/football-tactical-ai:latest"  ← لما تسوي deploy
```

### 3. شغل الواجهة مرة ثانية

```bash
streamlit run Home.py
```

---

## 🎯 في يوم واحد (تدريب نموذجك)

### 1. جهز البيانات

```bash
# سوي هيكل المجلدات
mkdir -p dataset/images/train dataset/images/val
mkdir -p dataset/labels/train dataset/labels/val

# حط صورك في images/train
# حط labels في labels/train
```

### 2. سوي Labeling

استخدم [Roboflow](https://roboflow.com):
1. سجل حساب (مجاني)
2. ارفع صورك
3. اعمل labeling للاعبين والكرة
4. صدّر بصيغة YOLO

### 3. درب النموذج

```bash
cd replicate_model

# إنشاء data.yaml
python train.py --data /path/to/dataset --model m --epochs 50

# النتيجة: runs/train/football_analysis/weights/best.pt
```

### 4. ارفع على Replicate

```bash
# انسخ النموذج
cp runs/train/football_analysis/weights/best.pt models/

# ثبت cog
pip install cog
cog login

# ارفع
cog push r8.im/YOUR_USERNAME/football-tactical-ai
```

### 5. جربه!

ارجع للواجهة، رفع فيديو، وشوف النتائج الحقيقية!

---

## 📁 الملفات المهمة

| الملف | الغرض |
|-------|-------|
| `Home.py` | الواجهة الرئيسية |
| `REPLICATE_SETUP.md` | دليل كامل لـ Replicate |
| `CHECKLIST.md` | قائمة مراجعة |
| `replicate_model/train.py` | تدريب النموذج |
| `replicate_model/predict.py` | Replicate endpoint |
| `.streamlit/secrets.toml` | الإعدادات السرية |

---

## 🆘 المساعدة السريعة

### مشكلة: "Module not found"
```bash
pip install -r requirements.txt
```

### مشكلة: "Replicate API token required"
```bash
# تأكد من secrets.toml
# أو حط التوكن مؤقتاً:
export REPLICATE_API_TOKEN="r8_xxxxxxxx"
```

### مشكلة: الفيديو ما يشتغل
```bash
# نصب ffmpeg
brew install ffmpeg  # Mac
# أو
sudo apt install ffmpeg  # Linux
```

---

## 🎉 مبروك!

لما توصل هنا، عندك:
- ✅ واجهة جاهزة
- ✅ كود مجرب
- ✅ دليل كامل
- ✅ خطة واضحة

**الخطوة الجاية:** ابدأ بالوضع التجريبي، ثم طور تدريجياً!
