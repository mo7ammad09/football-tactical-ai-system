# ✅ قائمة مراجعة إعداد Replicate

## المرحلة 1: التحضير (قبل التدريب)

### البيانات
- [ ] جمعت فيديوهات مباريات (5-10 على الأقل)
- [ ] سويت labeling للاعبين والكرة
- [ ] هيكل البيانات صحيح (images + labels)
- [ ] أنواع الكائنات: player, goalkeeper, ball, referee

### البيئة
- [ ] Python 3.10 مثبت
- [ ] pip محدث
- [ ] GPU متاح (للتدريب)

---

## المرحلة 2: التدريب

### التثبيت
```bash
pip install ultralytics
```

### التدريب
- [ ] سويت data.yaml
- [ ] دربت النموذج:
  ```bash
  cd replicate_model
  python train.py --data /path/to/dataset --model m --epochs 100
  ```
- [ ] النموذج اجتاز الاختبار
- [ ] النتائج مقبولة (mAP > 0.7)

### التصدير
- [ ] انسخت best.pt لـ replicate_model/models/

---

## المرحلة 3: إعداد Replicate

### الحساب
- [ ] سجلت في [replicate.com](https://replicate.com)
- [ ] نسخت API Token

### Cog
- [ ] ثبتت cog:
  ```bash
  pip install cog
  ```
- [ ] سجلت الدخول:
  ```bash
  cog login
  # الصق الـ API Token
  ```

---

## المرحلة 4: الرفع

### التحقق
- [ ] best.pt موجود في replicate_model/models/
- [ ] cog.yaml محدث
- [ ] predict.py يشتغل محلياً:
  ```bash
  cd replicate_model
  cog predict -i video=@test.mp4
  ```

### الرفع
- [ ] رفعت النموذج:
  ```bash
  cog push r8.im/YOUR_USERNAME/football-tactical-ai
  ```
- [ ] ظهرت رسالة النجاح

---

## المرحلة 5: الربط بالواجهة

### الإعدادات
- [ ] حدثت `.streamlit/secrets.toml`:
  ```toml
  REPLICATE_API_TOKEN = "r8_xxxxxxxx"
  REPLICATE_MODEL_VERSION = "your-username/football-tactical-ai:latest"
  ```
- [ ] حدثت `src/api/replicate_client.py`:
  ```python
  self.model_version = "your-username/football-tactical-ai:latest"
  ```

### الاختبار
- [ ] شغلت الواجهة:
  ```bash
  streamlit run Home.py
  ```
- [ ] رفعت فيديو تجريبي
- [ ] التحليل اشتغل
- [ ] النتائج ظهرت صح

---

## المرحلة 6: التشغيل

### أول مباراة حقيقية
- [ ] رفعت فيديو مباراة كاملة
- [ ] استنيت النتيجة
- [ ] فحصت الإحصائيات
- [ ] فحصت اللوحة التكتيكية
- [ ] فحصت تحليل AI

### التحسين
- [ ] ضبطت confidence threshold
- [ ] اختبرت خيارات مختلفة
- [ ] سجلت ملاحظات للتحسين

---

## 🎯 بعد ما يشتغل Replicate

### تطوير الميزات
- [ ] أضفت Tactic Patterns جديدة
- [ ] ربطت AI (Gemini/OpenAI)
- [ ] طوّرت الواجهة
- [ ] أضفت تقارير PDF

### التوسع
- [ ] تحويل لسيرفر GPU خاص (اختياري)
- [ ] إضافة دعم متعدد المستخدمين
- [ ] تطبيق موبايل

---

## 📞 روابط مهمة

| الغرض | الرابط |
|-------|--------|
| Replicate Account | https://replicate.com/account |
| Replicate Docs | https://replicate.com/docs |
| Cog GitHub | https://github.com/replicate/cog |
| Ultralytics Docs | https://docs.ultralytics.com |
| Roboflow (Labeling) | https://roboflow.com |

---

## 💰 تتبع التكلفة

| البند | التكلفة |
|-------|---------|
| Replicate Account | مجاني |
| التدريب | مجاني (على جهازك) |
| الرفع | مجاني |
| Prediction الأولى | ~$0.01-0.02 |
| Prediction 90 دقيقة | ~$0.10-0.20 |

**التكلفة الشهرية التقديرية:**
- 10 مباريات: ~$1-3
- 50 مباراة: ~$5-15

---

## ✅ جاهز للانطلاق!

لما تكمل كل الخطوات فوق، النظام جاهز للاستخدام الفعلي.

**الخطوة الجاية:** ابدأ بجمع البيانات وتدريب النموذج!
