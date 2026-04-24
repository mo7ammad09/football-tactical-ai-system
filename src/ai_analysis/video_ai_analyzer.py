"""
AI Analyzer that ACTUALLY looks at video frames.
Uses Gemini Pro Vision or GPT-4V to analyze real frames.
"""

import base64
import os
from io import BytesIO
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


class VideoAIAnalyzer:
    """AI analyzer that analyzes real video frames."""

    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None):
        """Initialize analyzer.

        Args:
            provider: "gemini" or "openai"
            api_key: API key
        """
        self.provider = provider
        self.api_key = api_key or self._get_api_key(provider)
        self.client = None
        
        if provider == "gemini" and self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel('gemini-1.5-flash')  # يدعم الفيديو!
            except ImportError:
                print("⚠️ google-generativeai not installed")
                print("   pip install google-generativeai")

    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key from environment."""
        if provider == "gemini":
            return os.environ.get("GEMINI_API_KEY")
        elif provider == "openai":
            return os.environ.get("OPENAI_API_KEY")
        return None

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert frame to base64 string.

        Args:
            frame: Video frame (numpy array)

        Returns:
            Base64 encoded image string
        """
        # Convert BGR to RGB
        frame_rgb = frame[:, :, ::-1]
        pil_image = Image.fromarray(frame_rgb)
        
        # Save to buffer
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        
        # Encode to base64
        return base64.b64encode(buffer.read()).decode('utf-8')

    def _select_key_frames(self, frames: List[np.ndarray], num_frames: int = 5) -> List[np.ndarray]:
        """Select key frames from video for analysis.

        Args:
            frames: All video frames
            num_frames: Number of frames to select

        Returns:
            Selected key frames
        """
        if len(frames) <= num_frames:
            return frames
        
        # Select evenly spaced frames
        indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        return [frames[i] for i in indices]

    def analyze_video_real(
        self,
        frames: List[np.ndarray],
        stats: Dict,
        language: str = "ar"
    ) -> str:
        """Analyze video by looking at actual frames.

        Args:
            frames: Video frames
            stats: Match statistics
            language: "ar" or "en"

        Returns:
            Real analysis based on what AI sees
        """
        if self.client is None:
            return self._fallback_analysis(stats, language)
        
        # Select key frames
        key_frames = self._select_key_frames(frames, num_frames=5)
        
        # Convert frames to base64
        frame_data = []
        for i, frame in enumerate(key_frames):
            base64_image = self._frame_to_base64(frame)
            frame_data.append({
                "frame_num": i,
                "base64": base64_image
            })
        
        # Build prompt
        if language == "ar":
            prompt = f"""
أنت محلل تكتيكي محترف لكرة القدم. قمت بتحليل فيديو مباراة ورصدت هذه البيانات:

البيانات المستخرجة:
- عدد اللاعبين: {stats.get('player_count', 0)}
- استحواذ الفريق 1: {stats.get('possession_team1', 0):.1f}%
- استحواذ الفريق 2: {stats.get('possession_team2', 0):.1f}%

أمامك 5 صور أساسية من المباراة (فريمات رئيسية).

المطلوب:
1. صف ما تراه في الصور من تشكيلات وحركة لاعبين
2. حلل الوضع التكتيكي بناءً على مواقع اللاعبين
3. اذكر نقاط القوة والضعف المرئية
4. لا تختلق معلومات - قل الحقائق اللي تشوفها فقط
5. إذا ما تقدر تحدد شيء، قل "غير واضح في الصور"

قدم تحليلاً قصيراً ومفيداً (5-7 نقاط فقط).
"""
        else:
            prompt = f"""
You are an expert football tactical analyst. You analyzed a match video and extracted this data:

Extracted Data:
- Player count: {stats.get('player_count', 0)}
- Team 1 possession: {stats.get('possession_team1', 0):.1f}%
- Team 2 possession: {stats.get('possession_team2', 0):.1f}%

You have 5 key frames from the match.

Task:
1. Describe what you see in the images - formations, player positions
2. Analyze tactical situation based on player positions
3. Mention visible strengths and weaknesses
4. DO NOT make up information - only state what you see
5. If you can't determine something, say "not clear from images"

Provide a short, useful analysis (5-7 points only).
"""
        
        try:
            # For Gemini - prepare content with images
            if self.provider == "gemini":
                content = [prompt]
                
                # Add images
                for frame_info in frame_data:
                    image_bytes = base64.b64decode(frame_info["base64"])
                    content.append(Image.open(BytesIO(image_bytes)))
                
                # Generate response
                response = self.client.generate_content(content)
                return response.text
            
            else:
                # Fallback for other providers
                return self._fallback_analysis(stats, language)
                
        except Exception as e:
            print(f"AI Analysis Error: {e}")
            return self._fallback_analysis(stats, language)

    def _fallback_analysis(self, stats: Dict, language: str) -> str:
        """Fallback analysis when AI is not available.

        Args:
            stats: Match statistics
            language: Language

        Returns:
            Basic analysis based on stats only
        """
        if language == "ar":
            return f"""
## تحليل المباراة (بناءً على البيانات المستخرجة)

### البيانات الأساسية:
- عدد اللاعبين المكتشفين: {stats.get('player_count', 0)}
- استحواذ الفريق 1: {stats.get('possession_team1', 0):.1f}%
- استحواذ الفريق 2: {stats.get('possession_team2', 0):.1f}%

### ملاحظات:
⚠️ **للحصول على تحليل ذكي حقيقي:**
1. احصل على مفتاح Gemini API من: https://makersuite.google.com/app/apikey
2. أضفه في إعدادات الواجهة
3. سيتمكن AI من رؤية الصور وتحليلها فعلياً

### بدون AI:
التحليل الحالي يعتمد على البيانات الرقمية فقط (مواقع اللاعبين، الاستحواذ).
للحصول على تحليل تكتيكي متقدم، يرجى توصيل AI.
"""
        else:
            return f"""
## Match Analysis (Based on Extracted Data)

### Basic Data:
- Players detected: {stats.get('player_count', 0)}
- Team 1 possession: {stats.get('possession_team1', 0):.1f}%
- Team 2 possession: {stats.get('possession_team2', 0):.1f}%

### Notes:
⚠️ **For real AI analysis:**
1. Get Gemini API key from: https://makersuite.google.com/app/apikey
2. Add it in the UI settings
3. AI will be able to see images and analyze them

### Without AI:
Current analysis is based on numerical data only (player positions, possession).
For advanced tactical analysis, please connect AI.
"""

    def analyze_opponent_real(
        self,
        frames: List[np.ndarray],
        language: str = "ar"
    ) -> Dict:
        """Analyze opponent from real frames.

        Args:
            frames: Video frames
            language: Language

        Returns:
            Analysis dict
        """
        if self.client is None:
            return {"error": "AI not configured"}
        
        key_frames = self._select_key_frames(frames, num_frames=3)
        
        prompt = """
Analyze the opponent team from these match frames. Focus on:
1. Formation used
2. Defensive line height
3. Pressing intensity
4. Weaknesses visible
5. Key players

Return JSON format:
{
    "formation": "4-4-2",
    "defensive_line": "medium",
    "pressing": "low",
    "weaknesses": ["slow fullbacks", "gap between midfield and defense"],
    "key_players": ["striker", "playmaker"]
}
"""
        
        try:
            content = [prompt]
            for frame in key_frames:
                base64_image = self._frame_to_base64(frame)
                image_bytes = base64.b64decode(base64_image)
                content.append(Image.open(BytesIO(image_bytes)))
            
            response = self.client.generate_content(content)
            
            # Try to parse JSON
            import json
            try:
                return json.loads(response.text)
            except:
                return {"analysis": response.text}
                
        except Exception as e:
            return {"error": str(e)}


class MockVideoAnalyzer:
    """Mock analyzer for testing without API."""

    def analyze_video_real(self, frames: List[np.ndarray], stats: Dict, language: str = "ar") -> str:
        """Return honest mock analysis."""
        if language == "ar":
            return f"""
## ⚠️ تحليل تجريبي (بدون AI حقيقي)

### البيانات المستخرجة فعلياً من الفيديو:
- ✅ عدد اللاعبين المكتشفين: {stats.get('player_count', 0)}
- ✅ استحواذ الفريق 1: {stats.get('possession_team1', 0):.1f}%
- ✅ استحواذ الفريق 2: {stats.get('possession_team2', 0):.1f}%
- ✅ عدد الفريمات المحللة: {len(frames)}

### ⚠️ للحصول على تحليل ذكي حقيقي:

**الخطوة 1:** احصل على مفتاح Gemini API
- ادخل: https://makersuite.google.com/app/apikey
- سجل حساب Google (مجاني)
- أنشئ مفتاح API

**الخطوة 2:** أضف المفتاح في الواجهة
- افتح الإعدادات (الشريط الجانبي)
- اختر "Gemini" كمزود AI
- أدخل المفتاح

**الخطوة 3:** جرب التحليل مرة أخرى
- AI سيشاهد الفريمات فعلياً
- سيعطي تحليلاً حقيقياً بناءً على اللي شافه
- لا هراء - فقط حقائق من الفيديو

### 💡 لماذا التحليل الحالي "فلسفة"؟
لأنه يستخدم بيانات وهمية (Mock) ما لها علاقة بفيديوك.
لما توصل AI حقيقي، راح يحلل الفريمات الفعلية ويعطيك تحليل دقيق.
"""
        else:
            return f"""
## ⚠️ Demo Analysis (No Real AI)

### Actually Extracted Data:
- ✅ Players detected: {stats.get('player_count', 0)}
- ✅ Team 1 possession: {stats.get('possession_team1', 0):.1f}%
- ✅ Team 2 possession: {stats.get('possession_team2', 0):.1f}%
- ✅ Frames analyzed: {len(frames)}

### ⚠️ To Get Real AI Analysis:

**Step 1:** Get Gemini API Key
- Go to: https://makersuite.google.com/app/apikey
- Sign up for Google account (free)
- Create API key

**Step 2:** Add key in UI
- Open settings (sidebar)
- Select "Gemini" as AI provider
- Enter your key

**Step 3:** Try analysis again
- AI will see actual frames
- Will give real analysis based on what it sees
- No BS - only facts from video

### 💡 Why Current Analysis is "Philosophy"?
Because it uses fake mock data unrelated to your video.
When you connect real AI, it will analyze actual frames and give accurate analysis.
"""
