"""
AI-powered analysis and coaching assistant.
Based on: Tactic_Zone (Gemini integration)
"""

import json
import os
from typing import Dict, List, Optional


class AIAnalyzer:
    """AI analyzer for tactical insights and coaching advice."""

    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None):
        """Initialize AI analyzer.

        Args:
            provider: "gemini", "openai", or "mock".
            api_key: API key. If None, reads from env.
        """
        self.provider = provider
        self.api_key = api_key or self._get_api_key(provider)
        self.client = None

        if provider == "gemini" and self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel('gemini-pro')
            except ImportError:
                print("Warning: google-generativeai not installed")

        elif provider == "openai" and self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                print("Warning: openai not installed")

    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key from environment."""
        if provider == "gemini":
            return os.environ.get("GEMINI_API_KEY")
        elif provider == "openai":
            return os.environ.get("OPENAI_API_KEY")
        return None

    # ==================== MATCH REPORTS ====================

    def generate_match_report(
        self,
        match_data: Dict,
        language: str = "ar"
    ) -> str:
        """Generate comprehensive tactical match report.

        Args:
            match_data: Match statistics and tactical data.
            language: "ar" or "en".

        Returns:
            Formatted report text.
        """
        prompt = self._build_match_report_prompt(match_data, language)
        return self._generate_text(prompt)

    def _build_match_report_prompt(self, data: Dict, language: str) -> str:
        """Build prompt for match report."""
        if language == "ar":
            return f"""
أنت محلل تكتيكي محترف لكرة القدم. قدم تحليلاً شاملاً للمباراة:

البيانات:
التشكيلة: {data.get('formation', 'غير معروف')}
استحواذ الفريق 1: {data.get('possession_team1', 0):.1f}%
استحواذ الفريق 2: {data.get('possession_team2', 0):.1f}%
التسديدات: {data.get('shots', 0)}
التمريرات الناجحة: {data.get('passes', 0)}
أخطاء: {data.get('errors', 0)}
شدة الضغط: {data.get('pressing_intensity', 'متوسطة')}

قدم تحليلاً تكتيكياً يشمل:
1. نقاط القوة والضعف لكل فريق
2. الأنماط التكتيكية الملاحظة
3. اللاعبين الأكثر تأثيراً
4. توصيات للمباريات القادمة
5. تغييرات مقترحة في التشكيلة
"""
        else:
            return f"""
You are an expert football tactical analyst. Analyze this match:

Data:
Formation: {data.get('formation', 'unknown')}
Team 1 Possession: {data.get('possession_team1', 0):.1f}%
Team 2 Possession: {data.get('possession_team2', 0):.1f}%
Shots: {data.get('shots', 0)}
Successful Passes: {data.get('passes', 0)}
Errors: {data.get('errors', 0)}
Pressing Intensity: {data.get('pressing_intensity', 'medium')}

Provide comprehensive tactical analysis including:
1. Strengths and weaknesses of each team
2. Tactical patterns observed
3. Key players and their impact
4. Recommendations for next matches
5. Suggested formation changes
"""

    # ==================== OPPONENT ANALYSIS ====================

    def analyze_opponent(
        self,
        opponent_data: Dict,
        language: str = "ar"
    ) -> Dict:
        """Analyze opponent weaknesses and suggest counter strategies.

        Args:
            opponent_data: Opponent match data.
            language: "ar" or "en".

        Returns:
            Structured analysis with weaknesses and strategies.
        """
        prompt = self._build_opponent_prompt(opponent_data, language)
        response = self._generate_text(prompt)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"analysis": response}

    def _build_opponent_prompt(self, data: Dict, language: str) -> str:
        """Build prompt for opponent analysis."""
        if language == "ar":
            return f"""
حلل نقاط ضعف الخصم التالي واقترح استراتيجيات مضادة:

بيانات الخصم:
{json.dumps(data, ensure_ascii=False, indent=2)}

قدم النتيجة بتنسيق JSON:
{{
    "weaknesses": ["نقطة ضعف 1", "نقطة ضعف 2"],
    "counter_strategies": ["استراتيجية 1", "استراتيجية 2"],
    "key_players_to_watch": ["لاعب 1", "لاعب 2"],
    "recommended_formation": "التشكيلة المقترحة",
    "training_focus": ["تركيز تدريبي 1", "تركيز تدريبي 2"]
}}
"""
        else:
            return f"""
Analyze opponent weaknesses and suggest counter strategies:

Opponent Data:
{json.dumps(data, indent=2)}

Return JSON format:
{{
    "weaknesses": ["weakness 1", "weakness 2"],
    "counter_strategies": ["strategy 1", "strategy 2"],
    "key_players_to_watch": ["player 1", "player 2"],
    "recommended_formation": "suggested formation",
    "training_focus": ["focus 1", "focus 2"]
}}
"""

    # ==================== TRAINING PLANS ====================

    def generate_training_plan(
        self,
        team_weaknesses: List[str],
        upcoming_opponent: Optional[Dict] = None,
        language: str = "ar"
    ) -> str:
        """Generate training plan based on weaknesses.

        Args:
            team_weaknesses: List of identified weaknesses.
            upcoming_opponent: Optional opponent data.
            language: "ar" or "en".

        Returns:
            Training plan text.
        """
        prompt = self._build_training_prompt(team_weaknesses, upcoming_opponent, language)
        return self._generate_text(prompt)

    def _build_training_prompt(
        self,
        weaknesses: List[str],
        opponent: Optional[Dict],
        language: str
    ) -> str:
        """Build prompt for training plan."""
        opponent_str = ""
        if opponent:
            opponent_str = f"\nالخصم القادم:\n{json.dumps(opponent, ensure_ascii=False, indent=2)}"

        if language == "ar":
            return f"""
قم بإنشاء خطة تدريبية احترافية:

نقاط الضعف المراد معالجتها:
{chr(10).join(f'- {w}' for w in weaknesses)}
{opponent_str}

الخطة يجب أن تشمل:
1. تمارين تقنية (30 دقيقة)
2. تمارين تكتيكية (30 دقيقة)
3. تمارين بدنية (20 دقيقة)
4. تمرين تطبيقي (20 دقيقة)
5. جدول زمني لأسبوع كامل
6. تمارين خاصة لكل نقطة ضعف
"""
        else:
            return f"""
Create a professional training plan:

Weaknesses to address:
{chr(10).join(f'- {w}' for w in weaknesses)}
{opponent_str}

Include:
1. Technical drills (30 min)
2. Tactical exercises (30 min)
3. Physical conditioning (20 min)
4. Practical application (20 min)
5. Weekly schedule
6. Specific drills for each weakness
"""

    # ==================== REAL-TIME COACHING ====================

    def get_realtime_advice(
        self,
        match_state: Dict,
        language: str = "ar"
    ) -> str:
        """Get real-time coaching advice.

        Args:
            match_state: Current match state.
            language: "ar" or "en".

        Returns:
            Coaching advice (1-2 sentences).
        """
        prompt = self._build_realtime_prompt(match_state, language)
        return self._generate_text(prompt)

    def _build_realtime_prompt(self, state: Dict, language: str) -> str:
        """Build prompt for real-time advice."""
        if language == "ar":
            return f"""
أنت مدرب مساعد ذكي. قدم نصيحة فورية (جملة أو جملتين فقط):

الحالة الحالية:
- النتيجة: {state.get('score', '0-0')}
- الوقت: {state.get('time', '0')}
- استحواذ: {state.get('possession', '50-50')}
- آخر حدث: {state.get('last_event', 'لا شيء')}
- موقع الكرة: {state.get('ball_position', 'وسط الملعب')}
- شدة الضغط: {state.get('pressing', 'متوسطة')}
- حالة الفريق: {state.get('team_status', 'متوازن')}

النصيحة يجب أن تكون:
- قصيرة ومباشرة
- قابلة للتنفيذ فوراً
- باللغة العربية الفصحى
"""
        else:
            return f"""
You are an AI assistant coach. Give immediate advice (1-2 sentences only):

Current State:
- Score: {state.get('score', '0-0')}
- Time: {state.get('time', '0')}
- Possession: {state.get('possession', '50-50')}
- Last Event: {state.get('last_event', 'none')}
- Ball Position: {state.get('ball_position', 'center')}
- Pressing: {state.get('pressing', 'medium')}
- Team Status: {state.get('team_status', 'balanced')}

Advice should be:
- Short and direct
- Immediately actionable
- In English
"""

    # ==================== PLAYER ANALYSIS ====================

    def analyze_player_performance(
        self,
        player_data: Dict,
        language: str = "ar"
    ) -> str:
        """Analyze individual player performance.

        Args:
            player_data: Player statistics.
            language: "ar" or "en".

        Returns:
            Player analysis text.
        """
        prompt = self._build_player_prompt(player_data, language)
        return self._generate_text(prompt)

    def _build_player_prompt(self, data: Dict, language: str) -> str:
        """Build prompt for player analysis."""
        if language == "ar":
            return f"""
حلل أداء اللاعب التالي:

بيانات اللاعب:
{json.dumps(data, ensure_ascii=False, indent=2)}

قدم تحليلاً يشمل:
1. نقاط القوة
2. نقاط الضعف
3. مقارنة مع المتوسط
4. توصيات للتحسين
5. أفضل دور تكتيكي له
"""
        else:
            return f"""
Analyze this player's performance:

Player Data:
{json.dumps(data, indent=2)}

Provide analysis including:
1. Strengths
2. Weaknesses
3. Comparison to average
4. Improvement recommendations
5. Best tactical role
"""

    # ==================== CORE GENERATION ====================

    def _generate_text(self, prompt: str) -> str:
        """Generate text using configured AI provider.

        Args:
            prompt: Input prompt.

        Returns:
            Generated text.
        """
        if self.provider == "mock" or self.client is None:
            return self._mock_response(prompt)

        try:
            if self.provider == "gemini":
                response = self.client.generate_content(prompt)
                return response.text

            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2048
                )
                return response.choices[0].message.content

        except Exception as e:
            return f"Error generating response: {str(e)}"

        return "Unknown error"

    def _mock_response(self, prompt: str) -> str:
        """Generate mock response for testing without API."""
        if "match_report" in prompt.lower() or "تحليل" in prompt:
            return """
# التحليل التكتيكي للمباراة

## نقاط القوة:
- استحواذ ممتاز في وسط الملعب (55.3%)
- ضغط عالٍ يُجبر الخصم على الأخطاء
- انتقال سريع من الدفاع للهجوم

## نقاط الضعف:
- ضعف في التسديد من خارج المنطقة
- بطء في الانتقال الدفاعي عند الهجمات المرتدة
- قلة التمريرات الحاسمة

## التوصيات:
1. زيادة التسديد من خارج المنطقة في التدريب
2. تسريع الانتقال الدفاعي
3. العمل على الكرات الثابتة
"""
        elif "opponent" in prompt.lower() or "خصم" in prompt:
            return json.dumps({
                "weaknesses": [
                    "ضعف في الجانب الأيسر للدفاع",
                    "بطء في بناء الهجمة",
                    "صعوبة في مواجهة الضغط العالي"
                ],
                "counter_strategies": [
                    "الضغط العالي على مدافعيهم",
                    "الهجوم من الجانب الأيمن",
                    "التسديد من خارج المنطقة"
                ],
                "key_players_to_watch": ["المهاجم رقم 9", "الوسط المدافع رقم 6"],
                "recommended_formation": "4-3-3",
                "training_focus": ["الضغط العالي", "الهجمات المرتدة"]
            }, ensure_ascii=False)
        elif "training" in prompt.lower() or "تدريب" in prompt:
            return """
# خطة تدريبية أسبوعية

## الاثنين:
- تمارين تقنية: التمرير الدقيق (30 دقيقة)
- تمارين تكتيكية: الضغط العالي (30 دقيقة)

## الثلاثاء:
- تمارين بدنية: سرعة وقوة (40 دقيقة)
- تمرين تطبيقي: مباراة مصغرة (40 دقيقة)

## الأربعاء:
- راحة استشفاء

## الخميس:
- تمارين تقنية: التسديد (30 دقيقة)
- تمارين تكتيكية: الكرات الثابتة (30 دقيقة)

## الجمعة:
- تمرين تطبيقي: محاكاة الخصم (60 دقيقة)

## السبت:
- مباراة ودية
"""
        elif "realtime" in prompt.lower() or "فوري" in prompt:
            return "💡 نصيحة: الفريق يضغط بقوة - اقترح التمرير السريع للأطراف لتفكيك الضغط"
        else:
            return "تم إنشاء التحليل بنجاح. يرجى مراجعة البيانات المدخلة للتفاصيل."
