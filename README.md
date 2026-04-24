# ⚽ Football Tactical AI System

نظام تحليل تكتيكي ذكي لمباريات كرة القدم باستخدام الرؤية الحاسوبية والذكاء الاصطناعي.

## 🎯 المميزات

### القسم الأول: تحليل ما بعد المباراة
- 🔍 كشف وتتبع اللاعبين والكرة والحكام
- 👥 تصنيف الفرق تلقائياً (A vs B)
- ⚽ تحديد استحواذ الكرة
- 📹 تقدير حركة الكاميرا
- 🔄 تحويل الإحداثيات للملعب 2D
- 🏃 حساب السرعة والمسافة
- 📊 التحليل التكتيكي (التشكيلات، السيطرة، الضغط)
- 🤖 تقارير ذكاء اصطناعي

### القسم الثاني: تحليل مباشر
- 📡 استقبال فيديو مباشر من الكاميرا
- ⚡ تحليل فوري
- 📢 نصائح مباشرة للمدرب
- 🎯 اكتشاف الأنماط التكتيكية لحظياً

## 🛠️ التقنيات المستخدمة

- **YOLOv8** - كشف الكائنات
- **ByteTrack** - تتبع متعدد الأجسام
- **OpenCV** - معالجة الفيديو
- **Streamlit** - واجهة المستخدم
- **scikit-learn** - التعلم الآلي
- **Gemini/OpenAI API** - الذكاء الاصطناعي

## 📁 هيكل المشروع

```
Football-Tactical-AI-System/
├── src/
│   ├── trackers/              # كشف وتتبع
│   ├── team_assigner/         # تصنيف الفرق
│   ├── ball_assigner/         # استحواذ الكرة
│   ├── camera_movement/       # حركة الكاميرا
│   ├── speed_distance/        # السرعة والمسافة
│   ├── view_transformer/      # تحويل الإحداثيات
│   ├── tactical_analysis/     # التحليل التكتيكي
│   ├── ai_analysis/           # الذكاء الاصطناعي
│   ├── utils/                 # أدوات مساعدة
│   └── visualizations/        # الرسوم البيانية
├── pages/                     # صفحات Streamlit
├── config/                    # الإعدادات
├── models/                    # النماذج
├── tests/                     # الاختبارات
└── docs/                      # التوثيق
```

## 🚀 التثبيت

```bash
# Clone repository
git clone <repo-url>
cd Football-Tactical-AI-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download YOLO model
# Place your trained model in models/yolov8m.pt
```

## 💻 الاستخدام

### تحليل فيديو
```bash
python main.py --input input_videos/match.mp4 --output output_videos/analyzed.mp4 --target-fps 10 --max-frames 3000
```

### الواجهة التفاعلية
```bash
streamlit run Home.py
```

### واجهة + GPU خارجي
- من نفس الواجهة اختر `Processing Mode = Remote GPU`
- أدخل رابط سيرفر الـGPU (FastAPI)
- ارفع فيديو المباراة
- اختر إما:
  - `Model path on server` (مثل: `models/abdullah_yolov5.pt`)
  - أو `Upload model file (.pt)` لرفع موديلك مع الطلب

### تشغيل جاهز بأوامر بسيطة
```bash
# على سيرفر GPU
bash scripts/start_gpu_server.sh

# على جهازك المحلي
REMOTE_GPU_URL=http://YOUR_GPU_IP:8000 bash scripts/start_ui.sh
```

### ابدأ مباشرة
اقرأ الملف هذا أولاً:
```bash
READY_START_HERE.md
```

### إذا Web Terminal في RunPod ما يشتغل
```bash
RUNPOD_HOST=YOUR_HOST RUNPOD_PORT=YOUR_PORT REPO_URL=YOUR_GIT_URL bash scripts/deploy_to_runpod_via_ssh.sh
RUNPOD_HOST=YOUR_HOST bash scripts/check_runpod_health.sh
```

### استخدام الـ Stubs (للتطوير السريع)
```bash
python main.py --input input_videos/match.mp4 --output output_videos/analyzed.mp4 --stubs
```

### تحليل على GPU خارجي (RunPod)
راجع:
```bash
runpod/README_RUNPOD.md
```

### مزامنة نسخ `src` (RunPod/Replicate)
```bash
bash scripts/sync_variants.sh
```

## 📋 متطلبات الفيديو

- دقة 1080p أو أعلى
- زاوية eagle eye (من فوق)
- 24-30 FPS
- يفضل كاميرا ثابتة

## 🤝 المساهمة

هذا المشروع مفتوح المصدر. نرحب بالمساهمات في:
- إضافة أنماط تكتيكية جديدة
- تحسين دقة الكشف
- دعم لغات جديدة
- تحسين واجهة المستخدم

## 📄 الترخيص

MIT License

## 🙏 شكراً لـ

- [football_analysis_yolo](https://github.com/TrishamBP/football_analysis_yolo)
- [Football Match Intelligence](https://github.com/DataKnight1/football-match-intelligence)
- [Tactic Zone](https://github.com/AbdelrahmanAtef01/Tactic_Zone)
- [Football Simple AI](https://github.com/farshidrayhancv/Football_Simple_AI)
