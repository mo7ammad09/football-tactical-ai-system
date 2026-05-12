# Football Tactical AI System - Agent Guidelines

## Project Overview
نظام تحليل تكتيكي ذكي لمباريات كرة القدم باستخدام الرؤية الحاسوبية والذكاء الاصطناعي.

## Architecture
```
Football-Tactical-AI-System/
├── src/
│   ├── trackers/              # كشف وتتبع اللاعبين والكرة
│   ├── team_assigner/         # تصنيف الفرق (A vs B)
│   ├── ball_assigner/         # تحديد من يملك الكرة
│   ├── camera_movement/       # تقدير حركة الكاميرا
│   ├── speed_distance/        # حساب السرعة والمسافة
│   ├── view_transformer/      # تحويل الإحداثيات للملعب 2D
│   ├── tactical_analysis/     # التحليل التكتيكي الأساسي
│   ├── ai_analysis/           # التحليل بالذكاء الاصطناعي
│   ├── utils/                 # دوال مساعدة
│   ├── visualizations/        # الرسوم البيانية والملعب التكتيكي
│   └── data/                  # معالجة البيانات
├── pages/                     # صفحات Streamlit
├── config/                    # الإعدادات
├── models/                    # نماذج YOLO المدربة
├── tests/                     # اختبارات
└── docs/                      # التوثيق
```

## Tech Stack
- **Python 3.10+**
- **YOLOv8** (Ultralytics) - كشف الكائنات
- **ByteTrack** - تتبع متعدد الأجسام
- **OpenCV** - معالجة الفيديو
- **NumPy/Pandas** - تحليل البيانات
- **Streamlit** - واجهة المستخدم
- **mplsoccer** - رسم الملعب التكتيكي
- **scikit-learn** - التعلم الآلي
- **OpenAI/Gemini API** - التحليل بالذكاء الاصطناعي

## Coding Standards

### Python Style
- PEP 8 compliant
- Type hints في جميع الدوال
- Docstrings بصيغة Google style
- تسمية بالإنجليزية، التعليقات بالعربية أو الإنجليزية

### Module Structure
كل وحدة يجب أن تحتوي على:
```python
"""
وصف الوحدة والغرض منها.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np

class ModuleName:
    """وصف الكلاس."""
    
    def __init__(self, config: Dict):
        """Initialize module.
        
        Args:
            config: إعدادات الوحدة.
        """
        pass
    
    def process(self, input_data: np.ndarray) -> Dict:
        """Process input data.
        
        Args:
            input_data: البيانات المدخلة.
            
        Returns:
            Dict containing results.
        """
        pass
```

## Key Principles
1. **Modular Design** - كل وظيفة في وحدة منفصلة
2. **Caching** - استخدام pickle stubs للنتائج الوسيطة
3. **Error Handling** - معالجة الأخطاء بسلاسة
4. **Logging** - تسجيل جميع العمليات
5. **Config-Driven** - جميع الإعدادات في config/

## Video Input Requirements
- دقة 1080p أو أعلى
- زاوية eagle eye (من فوق)
- 24-30 FPS
- يفضل كاميرا ثابتة أو حركة بطيئة

## Team Assignment Logic
1. استخراج لون القميص من bounding box
2. KMeans clustering على الألوان
3. تخصيص فريق لكل لاعب
4. تتبع الهوية عبر الفريمات (ID consistency)

## Ball Tracking Challenges
- الكرة صغيرة وسريعة
- استخدام interpolation للفريمات المفقودة
- تقدير المسافة الأقرب للاعب لتحديد الاستحواذ

## AI Integration Points
1. **Post-Match Analysis** - تحليل المباراة بعد انتهائها
2. **Real-Time Coaching** - تعليمات مباشرة للمدرب
3. **Tactic Pattern Recognition** - اكتشاف الأنماط التكتيكية
4. **Opponent Weakness Analysis** - تحليل نقاط ضعف الخصم

## NEVER Do
- لا تعدل على منطق التتبع الأساسي بدون اختبار
- لا تحذف ملفات stubs/cache
- لا ترفع ملفات الفيديو للـ git
- لا تستخدم نماذج غير موثوقة

## Testing
- اختبار كل وحدة بشكل منفصل
- استخدام فيديو قصير (30 ثانية) للتطوير
- التحقق من الإحداثيات على الملعب 2D
