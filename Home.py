"""
Main Streamlit application - Football Tactical AI System.
Now with REAL model support!
"""

import os
import time
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

# Page config
st.set_page_config(
    page_title="Football Tactical AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our modules
from src.trackers.tracker import Tracker
from src.team_assigner.team_assigner import TeamAssigner
from src.ball_assigner.ball_assigner import BallAssigner
from src.camera_movement.camera_movement_estimator import CameraMovementEstimator
from src.view_transformer.view_transformer import ViewTransformer
from src.speed_distance.speed_distance_estimator import SpeedDistanceEstimator
from src.ai_analysis.ai_analyzer import AIAnalyzer
from src.ai_analysis.video_ai_analyzer import VideoAIAnalyzer, MockVideoAnalyzer
from src.visualizations.tactical_board import TacticalBoard
from src.utils.video_utils import read_video, read_video_sampled, save_video, get_video_properties
from src.api.runpod_serverless_client import RunPodServerlessClient
from src.api.server_client import ServerClient


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
SMALL_UPLOAD_LIMIT_MB = 512
CONFIG_ENV_PATH = Path(__file__).resolve().parent / ".env"
LOCAL_CONFIG_KEYS = {
    "RUNPOD_ENDPOINT_ID",
    "RUNPOD_API_KEY",
    "OBJECT_STORAGE_BUCKET",
    "OBJECT_STORAGE_REGION",
    "OBJECT_STORAGE_ENDPOINT_URL",
    "OBJECT_STORAGE_ACCESS_KEY_ID",
    "OBJECT_STORAGE_SECRET_ACCESS_KEY",
    "OBJECT_STORAGE_PREFIX",
    "OBJECT_STORAGE_PRESIGN_EXPIRES",
}


def _parse_env_value(value: str) -> str:
    """Parse a simple .env value."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value.replace("\\n", "\n")


def _format_env_value(value: str) -> str:
    """Format a value for a local .env file."""
    escaped = value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
    return f'"{escaped}"'


def load_local_config(path: Path = CONFIG_ENV_PATH) -> None:
    """Load local app settings from .env without overriding real environment variables."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.replace("export ", "", 1).strip()
        if key in LOCAL_CONFIG_KEYS and key not in os.environ:
            os.environ[key] = _parse_env_value(value)


def save_local_config(values: dict[str, str], path: Path = CONFIG_ENV_PATH) -> None:
    """Persist local app settings to .env while preserving unrelated lines."""
    existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    pending = {key: str(value) for key, value in values.items() if key in LOCAL_CONFIG_KEYS}
    written = set()
    output_lines: list[str] = []

    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            output_lines.append(line)
            continue
        raw_key = stripped.split("=", 1)[0].strip()
        key = raw_key.replace("export ", "", 1).strip() if raw_key.startswith("export ") else raw_key
        if key in pending:
            output_lines.append(f"{key}={_format_env_value(pending[key])}")
            written.add(key)
        else:
            output_lines.append(line)

    if not output_lines:
        output_lines.append("# Local settings for Football Tactical AI. Do not commit this file.")
    for key, value in pending.items():
        if key not in written:
            output_lines.append(f"{key}={_format_env_value(value)}")

    path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    os.environ.update(pending)


load_local_config()


def list_input_videos(input_dir: str = "input_videos") -> list[Path]:
    """Return local videos intended for large-match processing."""
    root = Path(input_dir)
    if not root.exists():
        return []
    return sorted(
        [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def human_file_size(path: Path) -> str:
    """Format a file size for UI labels."""
    return format_bytes(path.stat().st_size)


def format_bytes(size: int | float) -> str:
    """Format byte counts for progress messages."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


def save_uploaded_file_stream(uploaded_file, dest_path: str) -> None:
    """Save a Streamlit upload in chunks instead of materializing getbuffer()."""
    with open(dest_path, "wb") as f:
        while True:
            chunk = uploaded_file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    uploaded_file.seek(0)


# ==================== SESSION STATE ====================
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "remote_server_url" not in st.session_state:
    st.session_state.remote_server_url = os.environ.get("REMOTE_GPU_URL", "http://localhost:8000")
if "video_properties" not in st.session_state:
    st.session_state.video_properties = None


# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("⚙️ الإعدادات")
    st.markdown("---")
    
    # Language
    language = st.radio("🌐 اللغة:", ["العربية", "English"], index=0)
    lang = "ar" if language == "العربية" else "en"
    
    st.markdown("---")

    processing_mode = st.radio(
        "⚙️ وضع المعالجة:" if lang == "ar" else "⚙️ Processing Mode:",
        ["RunPod Serverless", "Remote GPU", "Local"],
        index=0
    )
    use_remote_gpu = processing_mode == "Remote GPU"
    use_runpod_serverless = processing_mode == "RunPod Serverless"

    st.markdown("---")
    
    # Model Selection
    st.subheader("🧠 نموذج الكشف" if lang == "ar" else "🧠 Detection Model")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check available models
    old_data_path = os.path.join(base_dir, "models", "old_data.pt")
    abdullah_path = os.path.join(base_dir, "models", "abdullah_yolov5.pt")
    
    old_data_exists = os.path.exists(old_data_path)
    abdullah_exists = os.path.exists(abdullah_path)
    
    # Build model options based on availability
    available_models = []
    
    if old_data_exists:
        available_models.append("Tactic Zone (old_data.pt)")
    
    if abdullah_exists:
        available_models.append("Abdullah Tarek (yolov5)")
    
    available_models.append("Mock (تجريبي)" if lang == "ar" else "Mock (Demo)")
    
    model_choice = st.radio(
        "اختر النموذج:" if lang == "ar" else "Select Model:",
        available_models,
        index=0
    )
    
    # Determine which model to use
    use_real_model = model_choice != "Mock (تجريبي)" and model_choice != "Mock (Demo)"
    
    if "Tactic Zone" in model_choice:
        model_path = old_data_path
        model_name = "Tactic Zone"
    elif "Abdullah" in model_choice:
        model_path = abdullah_path
        model_name = "Abdullah Tarek"
    else:
        model_path = None
        model_name = "Mock"
    
    # Show model info
    if use_real_model:
        st.success(f"✅ {model_name} جاهز!" if lang == "ar" else f"✅ {model_name} ready!")
        st.caption(f"المسار: {model_path}" if lang == "ar" else f"Path: {model_path}")
    else:
        st.info("ℹ️ وضع تجريبي - بيانات وهمية" if lang == "ar" else "ℹ️ Demo mode - fake data")
    
    # Show download instructions for missing models
    if not old_data_exists and not abdullah_exists:
        st.warning("⚠️ لا يوجد نموذج حقيقي!" if lang == "ar" else "⚠️ No real model found!")
        st.info("""
        **لتحميل النماذج:**
        
        **Tactic Zone:**
        ```bash
        curl -L -o models/old_data.pt "https://www.dropbox.com/scl/fi/5wh4yy2ego497sw7ut01y/old_data.pt?rlkey=pkktrpl7kudux5xbaxu2is550&st=ftxxrz0d&dl=1"
        ```
        
        **Abdullah Tarek:**
        1. ادخل: https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view
        2. حمل الملف يدوياً
        3. حطه في: models/abdullah_yolov5.pt
        """)
    
    st.markdown("---")

    runpod_api_key = ""
    runpod_endpoint_id = ""
    storage_bucket = ""
    storage_region = ""
    storage_endpoint_url = ""
    storage_access_key = ""
    storage_secret_key = ""
    storage_prefix = "football-ai"
    storage_presign_expires = "86400"

    if use_remote_gpu:
        st.subheader("🌐 " + ("إعدادات السيرفر GPU" if lang == "ar" else "GPU Server Settings"))
        remote_server_url = st.text_input(
            "رابط السيرفر" if lang == "ar" else "Server URL",
            value=st.session_state.remote_server_url,
            help="مثال: http://YOUR_GPU_SERVER_IP:8000" if lang == "ar" else "Example: http://YOUR_GPU_SERVER_IP:8000"
        )
        st.session_state.remote_server_url = remote_server_url

        remote_api_key = st.text_input(
            "مفتاح API (اختياري)" if lang == "ar" else "API Key (Optional)",
            type="password",
            value=os.environ.get("REMOTE_GPU_API_KEY", "")
        )

        if st.button("🔎 اختبار الاتصال" if lang == "ar" else "🔎 Test Connection", use_container_width=True):
            try:
                health_headers = {"Authorization": f"Bearer {remote_api_key}"} if remote_api_key else None
                health_resp = requests.get(f"{remote_server_url.rstrip('/')}/health", headers=health_headers, timeout=10)
                if health_resp.ok:
                    st.success("✅ السيرفر متصل وجاهز" if lang == "ar" else "✅ Server is reachable")
                else:
                    st.error(f"❌ Health check failed: {health_resp.status_code}")
            except Exception as conn_err:
                st.error(f"❌ {conn_err}")

    elif use_runpod_serverless:
        st.subheader("☁️ " + ("RunPod Serverless" if lang == "ar" else "RunPod Serverless"))
        runpod_endpoint_id = st.text_input(
            "Endpoint ID" if lang == "ar" else "Endpoint ID",
            value=os.environ.get("RUNPOD_ENDPOINT_ID", ""),
        )
        runpod_api_key = st.text_input(
            "RunPod API Key" if lang == "ar" else "RunPod API Key",
            type="password",
            value=os.environ.get("RUNPOD_API_KEY", ""),
        )
        st.caption(
            "هذا الوضع يشغّل GPU تلقائياً عند إرسال job ويتوقف بعد الخمول حسب إعدادات RunPod."
            if lang == "ar"
            else "This mode starts GPU workers on demand and idles them down according to RunPod settings."
        )

        with st.expander("🗄️ " + ("إعدادات التخزين" if lang == "ar" else "Storage Settings"), expanded=True):
            storage_bucket = st.text_input(
                "Bucket",
                value=os.environ.get("OBJECT_STORAGE_BUCKET", ""),
                help="مثال: football-ai-mohammad-01" if lang == "ar" else "Example: football-ai-mohammad-01",
            )
            storage_region = st.text_input(
                "Region",
                value=os.environ.get("OBJECT_STORAGE_REGION", "eu-north-1"),
                help="مثال: eu-north-1" if lang == "ar" else "Example: eu-north-1",
            )
            storage_endpoint_url = st.text_input(
                "Endpoint URL",
                value=os.environ.get("OBJECT_STORAGE_ENDPOINT_URL", "https://s3.eu-north-1.amazonaws.com"),
            )
            storage_access_key = st.text_input(
                "AWS Access Key ID",
                type="password",
                value=os.environ.get("OBJECT_STORAGE_ACCESS_KEY_ID", ""),
            )
            storage_secret_key = st.text_input(
                "AWS Secret Access Key",
                type="password",
                value=os.environ.get("OBJECT_STORAGE_SECRET_ACCESS_KEY", ""),
            )
            storage_prefix = st.text_input(
                "Prefix",
                value=os.environ.get("OBJECT_STORAGE_PREFIX", "football-ai"),
            )
            storage_presign_expires = st.text_input(
                "Presigned URL expiry seconds",
                value=os.environ.get("OBJECT_STORAGE_PRESIGN_EXPIRES", "86400"),
            )

        local_config_values = {
            "RUNPOD_ENDPOINT_ID": runpod_endpoint_id.strip(),
            "RUNPOD_API_KEY": runpod_api_key.strip(),
            "OBJECT_STORAGE_BUCKET": storage_bucket.strip(),
            "OBJECT_STORAGE_REGION": storage_region.strip(),
            "OBJECT_STORAGE_ENDPOINT_URL": storage_endpoint_url.strip(),
            "OBJECT_STORAGE_ACCESS_KEY_ID": storage_access_key.strip(),
            "OBJECT_STORAGE_SECRET_ACCESS_KEY": storage_secret_key.strip(),
            "OBJECT_STORAGE_PREFIX": storage_prefix.strip(),
            "OBJECT_STORAGE_PRESIGN_EXPIRES": storage_presign_expires.strip(),
        }
        st.caption(
            "اضغط حفظ بعد تعبئة البيانات مرة واحدة. سيتم حفظها محلياً في ملف .env على جهازك."
            if lang == "ar"
            else "Save once after filling the fields. Settings are stored locally in your .env file."
        )
        if st.button("💾 حفظ الإعدادات على هذا الجهاز" if lang == "ar" else "💾 Save settings on this device", use_container_width=True):
            missing_config = [key for key, value in local_config_values.items() if not value]
            if missing_config:
                st.error(
                    "قبل الحفظ، أكمل: " + ", ".join(missing_config)
                    if lang == "ar"
                    else "Before saving, complete: " + ", ".join(missing_config)
                )
            else:
                save_local_config(local_config_values)
                st.success(
                    "تم الحفظ. المرة القادمة ستظهر البيانات تلقائياً."
                    if lang == "ar"
                    else "Saved. These values will load automatically next time."
                )

        if st.button("🔎 اختبار RunPod" if lang == "ar" else "🔎 Test RunPod", use_container_width=True):
            try:
                if not runpod_endpoint_id or not runpod_api_key:
                    raise ValueError("RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY are required")
                health_resp = requests.get(
                    f"https://api.runpod.ai/v2/{runpod_endpoint_id}/health",
                    headers={"Authorization": runpod_api_key},
                    timeout=20,
                )
                if health_resp.ok:
                    st.success("✅ RunPod endpoint جاهز" if lang == "ar" else "✅ RunPod endpoint is reachable")
                    st.json(health_resp.json())
                else:
                    st.error(f"❌ RunPod health failed: {health_resp.status_code} {health_resp.text[:200]}")
            except Exception as conn_err:
                st.error(f"❌ {conn_err}")

    if use_remote_gpu or use_runpod_serverless:
        analysis_fps_remote = st.number_input(
            "FPS للتحليل (أقل = أسرع)" if lang == "ar" else "Analysis FPS (lower = faster)",
            min_value=0.5, max_value=30.0, value=3.0, step=0.5,
            help=(
                "2-4 FPS مناسب للمباراة الكاملة. ارفعها فقط للمقاطع القصيرة."
                if lang == "ar"
                else "2-4 FPS is suitable for a full match. Increase only for short clips."
            )
        )
        analyze_full_match = st.checkbox(
            "تحليل المباراة كاملة" if lang == "ar" else "Analyze full match",
            value=True,
            help=(
                "إذا ألغيت الخيار سيتم تحليل أول عدد محدد من الفريمات فقط."
                if lang == "ar"
                else "Disable to cap processing to the first N sampled frames."
            ),
        )
        if analyze_full_match:
            max_frames_remote = None
        else:
            max_frames_remote = st.number_input(
                "أقصى عدد فريمات" if lang == "ar" else "Max Frames",
                min_value=300, max_value=50000, value=5400, step=300
            )
        resize_width_remote = st.number_input(
            "عرض الفريم قبل التحليل" if lang == "ar" else "Resize Width",
            min_value=480, max_value=1920, value=1280, step=80
        )
        st.caption(
            "للمقاطع القصيرة: FPS=8 و Resize=1280. للمباريات الطويلة: FPS=2-4."
            if lang == "ar"
            else "For short clips: FPS=8 and Resize=1280. For long matches: FPS=2-4."
        )
    else:
        remote_server_url = ""
        remote_api_key = ""
        analysis_fps_remote = 3.0
        max_frames_remote = None
        resize_width_remote = 1280

    st.markdown("---")
    
    # AI Provider
    ai_provider = st.selectbox(
        "🤖 مزود التحليل الذكي:" if lang == "ar" else "🤖 AI Analysis Provider:",
        ["Mock (تجريبي)", "Gemini", "OpenAI"]
    )
    
    st.markdown("---")
    st.caption("⚽ Football Tactical AI v1.0")


# ==================== HEADER ====================
st.markdown("<h1 style='text-align: center; color: #1a5f2a;'>⚽ Football Tactical AI</h1>", unsafe_allow_html=True)
if lang == "ar":
    st.markdown("<p style='text-align: center; color: #888;'>نظام تحليل تكتيكي ذكي لمباريات كرة القدم</p>", unsafe_allow_html=True)
else:
    st.markdown("<p style='text-align: center; color: #888;'>AI-Powered Football Match Analysis</p>", unsafe_allow_html=True)

st.markdown("---")


# ==================== VIDEO UPLOAD ====================
st.header("📹 " + ("مصدر الفيديو" if lang == "ar" else "Video Source"))
local_videos = list_input_videos()
video_path = None
uploaded_file = None

source_mode = st.radio(
    "اختر طريقة إدخال الفيديو" if lang == "ar" else "Choose video input method",
    [
        "اختيار ملف من input_videos" if lang == "ar" else "Select from input_videos",
        "رفع اختبار صغير" if lang == "ar" else "Upload small test file",
    ],
    index=0,
    horizontal=True,
)

if source_mode.startswith("اختيار") or source_mode.startswith("Select"):
    if not local_videos:
        st.warning(
            "ضع ملفات المباريات الكبيرة داخل مجلد input_videos ثم أعد تحميل الصفحة."
            if lang == "ar"
            else "Place large match files in input_videos, then refresh the page."
        )
    else:
        selected_video = st.selectbox(
            "الفيديو المحلي" if lang == "ar" else "Local video",
            local_videos,
            format_func=lambda path: f"{path.name} ({human_file_size(path)})",
        )
        video_path = str(selected_video)
        st.session_state.video_path = video_path
        st.session_state.video_properties = get_video_properties(video_path)
else:
    uploaded_file = st.file_uploader(
        "اسحب فيديو قصير هنا أو انقر للاختيار" if lang == "ar" else "Drag a short test video here or click to browse",
        type=["mp4", "avi", "mov", "mkv"],
    )
    if uploaded_file:
        file_size_mb = (uploaded_file.size or 0) / (1024 * 1024)
        if file_size_mb > SMALL_UPLOAD_LIMIT_MB:
            st.warning(
                f"هذا المسار للاختبارات الصغيرة فقط. لملف {file_size_mb:.1f} MB استخدم input_videos حتى لا يدخل الملف ذاكرة Streamlit."
                if lang == "ar"
                else f"This path is for small tests only. For a {file_size_mb:.1f} MB file, use input_videos to avoid Streamlit memory pressure."
            )
        os.makedirs("temp", exist_ok=True)
        safe_upload_name = Path(uploaded_file.name).name
        video_path = f"temp/{safe_upload_name}"
        save_uploaded_file_stream(uploaded_file, video_path)
        st.session_state.video_path = video_path
        st.session_state.video_properties = get_video_properties(video_path)

if video_path:
    col1, col2 = st.columns([2, 1])
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    with col1:
        if file_size_mb <= SMALL_UPLOAD_LIMIT_MB:
            st.video(video_path)
        else:
            st.info(
                "تم اختيار ملف كبير. لن يتم عرضه داخل المتصفح لتجنب استهلاك الذاكرة."
                if lang == "ar"
                else "Large file selected. Browser preview is skipped to avoid memory use."
            )
    with col2:
        st.markdown("### 📋 معلومات الفيديو" if lang == "ar" else "### 📋 Video Info")
        st.info(f"{'المسار' if lang == 'ar' else 'Path'}: {video_path}")
        st.info(f"{'الحجم' if lang == 'ar' else 'Size'}: {file_size_mb:.1f} MB")
        video_props = st.session_state.get("video_properties") or {}
        duration_seconds = int(video_props.get("duration_seconds", 0) or 0)
        if duration_seconds > 0:
            mins = duration_seconds // 60
            secs = duration_seconds % 60
            st.info(f"{'المدة' if lang == 'ar' else 'Duration'}: {mins:02d}:{secs:02d}")
        if video_props.get("fps"):
            st.info(f"FPS: {float(video_props['fps']):.1f}")
        if use_real_model:
            st.success("🧠 " + ("سيتم استخدام النموذج الحقيقي" if lang == "ar" else "Real model will be used"))
        else:
            st.warning("⚠️ " + ("سيتم استخدام الوضع التجريبي" if lang == "ar" else "Demo mode will be used"))

    remote_model_file = None
    remote_model_path = None
    remote_model_mode = None
    if use_remote_gpu or use_runpod_serverless:
        st.markdown("---")
        st.subheader("🧠 " + ("الموديل على GPU" if lang == "ar" else "Model on GPU"))
        remote_model_mode = st.radio(
            "طريقة تحديد الموديل" if lang == "ar" else "Model Source",
            [
                "Use selected app model (recommended)",
                "Upload model file (.pt)",
                "Model path on server",
            ],
            index=0,
        )
        if remote_model_mode == "Use selected app model (recommended)":
            if use_real_model and model_path and os.path.exists(model_path):
                st.success(
                    f"✅ {'سيتم رفع الموديل المحدد تلقائياً' if lang == 'ar' else 'Selected model will be uploaded automatically'}: {os.path.basename(model_path)}"
                )
            else:
                st.warning(
                    "⚠️ "
                    + (
                        "الموديل المحدد غير متاح محلياً. اختر Upload model file أو Model path on server."
                        if lang == "ar"
                        else "Selected model is not available locally. Choose Upload model file or Model path on server."
                    )
                )
        elif remote_model_mode == "Model path on server":
            remote_model_path = st.text_input(
                "مسار الموديل على السيرفر" if lang == "ar" else "Model path on server",
                value="",
                placeholder="e.g. /workspace/models/abdullah_yolov5.pt",
            )
        else:
            remote_model_file = st.file_uploader(
                "ارفع ملف الموديل (.pt)" if lang == "ar" else "Upload model file (.pt)",
                type=["pt"],
            )
    
    # Analysis button
    st.markdown("---")
    
    analyze_btn = st.button(
        "🚀 بدء التحليل" if lang == "ar" else "🚀 Start Analysis",
        type="primary",
        use_container_width=True
    )
    
    if analyze_btn:
        # Progress container
        progress_container = st.container()
        
        with progress_container:
            st.markdown("---")
            st.subheader("⏳ جاري التحليل..." if lang == "ar" else "⏳ Analyzing...")

            progress_bar = st.progress(0)
            status_text = st.empty()
            upload_detail = st.empty()
            
            try:
                if use_remote_gpu:
                    if not remote_server_url.strip():
                        raise ValueError("يرجى إدخال رابط سيرفر GPU" if lang == "ar" else "Please provide GPU server URL")

                    props = st.session_state.get("video_properties") or {}
                    duration_seconds = int(props.get("duration_seconds", 0) or 0)
                    if duration_seconds > 0 and max_frames_remote:
                        needed_frames = int(duration_seconds * float(analysis_fps_remote))
                        if needed_frames > int(max_frames_remote):
                            covered_minutes = (int(max_frames_remote) / float(analysis_fps_remote)) / 60.0
                            st.warning(
                                (
                                    f"تنبيه: بالإعدادات الحالية سيُحلَّل فقط أول {covered_minutes:.1f} دقيقة من الفيديو. "
                                    "ارفع Max Frames أو خفّض FPS."
                                )
                                if lang == "ar"
                                else (
                                    f"Note: with current settings, only the first {covered_minutes:.1f} minutes will be analyzed. "
                                    "Increase Max Frames or lower FPS."
                                )
                            )

                    model_file_path = None
                    model_path_for_server = None
                    if remote_model_mode == "Use selected app model (recommended)":
                        if not use_real_model or not model_path or not os.path.exists(model_path):
                            raise ValueError(
                                "الموديل المحدد غير موجود محلياً. اختر Upload model file (.pt) أو أدخل مسار صحيح على السيرفر."
                                if lang == "ar"
                                else "Selected model is not available locally. Choose Upload model file (.pt) or provide a valid server model path."
                            )
                        model_file_path = model_path
                    elif remote_model_mode == "Upload model file (.pt)":
                        if remote_model_file is None:
                            raise ValueError(
                                "ارفع ملف موديل (.pt) أولاً."
                                if lang == "ar"
                                else "Please upload a model file (.pt) first."
                            )
                        os.makedirs("temp", exist_ok=True)
                        model_file_path = f"temp/{remote_model_file.name}"
                        save_uploaded_file_stream(remote_model_file, model_file_path)
                    else:
                        if remote_model_path is None or not remote_model_path.strip():
                            raise ValueError(
                                "أدخل مسار الموديل على السيرفر."
                                if lang == "ar"
                                else "Please provide model path on server."
                            )
                        model_path_for_server = remote_model_path.strip()

                    status_text.info("☁️ رفع الفيديو إلى سيرفر GPU..." if lang == "ar" else "☁️ Uploading video to GPU server...")
                    progress_bar.progress(5)

                    client = ServerClient(server_url=remote_server_url, api_key=remote_api_key or None)
                    upload_started_at = time.time()
                    job_id = client.upload_video(
                        video_path=video_path,
                        progress_callback=lambda p: progress_bar.progress(max(5, min(int(p), 95))),
                        model_path=model_path_for_server,
                        model_file_path=model_file_path,
                        analysis_fps=float(analysis_fps_remote),
                        max_frames=int(max_frames_remote) if max_frames_remote else None,
                        resize_width=int(resize_width_remote),
                    )
                    upload_elapsed_s = time.time() - upload_started_at

                    status_text.info(
                        (
                            f"🧠 Job ID: {job_id} - انتهى الرفع خلال {upload_elapsed_s:.1f} ثانية. جاري المعالجة..."
                            if lang == "ar"
                            else f"🧠 Job ID: {job_id} - Upload finished in {upload_elapsed_s:.1f}s. Processing..."
                        )
                    )
                    progress_bar.progress(15)

                    processing_started_at = time.time()
                    while True:
                        status = client.get_status(job_id)
                        progress_bar.progress(int(status.get("progress", 15)))
                        processing_elapsed_s = time.time() - processing_started_at
                        status_text.info(
                            (
                                f"⏳ {status.get('message', 'processing')} ({processing_elapsed_s:.1f}s)"
                                if lang == "ar"
                                else f"⏳ {status.get('message', 'processing')} ({processing_elapsed_s:.1f}s)"
                            )
                        )
                        if status.get("status") == "completed":
                            break
                        if status.get("status") == "failed":
                            raise ValueError(status.get("message", "Remote analysis failed"))
                        time.sleep(3)

                    results = client.get_results(job_id)
                    video_url = results.get("annotated_video_url")
                    if video_url and video_url.startswith("/"):
                        video_url = f"{remote_server_url.rstrip('/')}{video_url}"
                    results["output_video"] = video_url
                    for artifact_url_key in ("report_json_url", "report_csv_url"):
                        artifact_url = results.get(artifact_url_key)
                        if artifact_url and artifact_url.startswith("/"):
                            results[artifact_url_key] = f"{remote_server_url.rstrip('/')}{artifact_url}"

                    st.session_state.analysis_results = results
                    st.session_state.analysis_done = True

                    progress_bar.progress(100)
                    status_text.success("✅ اكتمل التحليل على GPU بنجاح!" if lang == "ar" else "✅ GPU analysis completed successfully!")

                    if video_url:
                        st.markdown("---")
                        st.subheader("🎬 " + ("الفيديو المعلّم (GPU)" if lang == "ar" else "Annotated Video (GPU)"))
                        st.video(video_url)
                        st.markdown(
                            f"[⬇️ {'تحميل الفيديو الناتج' if lang == 'ar' else 'Download output video'}]({video_url})"
                        )

                elif use_runpod_serverless:
                    if not runpod_endpoint_id.strip() or not runpod_api_key.strip():
                        raise ValueError(
                            "أدخل RUNPOD_ENDPOINT_ID و RUNPOD_API_KEY."
                            if lang == "ar"
                            else "Please provide RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY."
                        )
                    required_storage_fields = {
                        "OBJECT_STORAGE_BUCKET": storage_bucket.strip(),
                        "OBJECT_STORAGE_REGION": storage_region.strip(),
                        "OBJECT_STORAGE_ENDPOINT_URL": storage_endpoint_url.strip(),
                        "OBJECT_STORAGE_ACCESS_KEY_ID": storage_access_key.strip(),
                        "OBJECT_STORAGE_SECRET_ACCESS_KEY": storage_secret_key.strip(),
                        "OBJECT_STORAGE_PREFIX": storage_prefix.strip(),
                        "OBJECT_STORAGE_PRESIGN_EXPIRES": storage_presign_expires.strip(),
                    }
                    missing_storage = [key for key, value in required_storage_fields.items() if not value]
                    if missing_storage:
                        raise ValueError(
                            (
                                "أكمل إعدادات التخزين في الشريط الجانبي: "
                                + ", ".join(missing_storage)
                            )
                            if lang == "ar"
                            else "Complete storage settings in the sidebar: " + ", ".join(missing_storage)
                        )
                    os.environ.update(required_storage_fields)

                    model_file_path = None
                    model_path_for_worker = None
                    if remote_model_mode == "Use selected app model (recommended)":
                        if not use_real_model or not model_path or not os.path.exists(model_path):
                            raise ValueError(
                                "الموديل المحدد غير موجود محلياً. اختر Upload model file (.pt) أو أدخل مسار موديل داخل صورة RunPod."
                                if lang == "ar"
                                else "Selected model is not available locally. Choose Upload model file (.pt) or provide a model path inside the RunPod image."
                            )
                        model_file_path = model_path
                    elif remote_model_mode == "Upload model file (.pt)":
                        if remote_model_file is None:
                            raise ValueError(
                                "ارفع ملف موديل (.pt) أولاً."
                                if lang == "ar"
                                else "Please upload a model file (.pt) first."
                            )
                        os.makedirs("temp", exist_ok=True)
                        model_file_path = f"temp/{remote_model_file.name}"
                        save_uploaded_file_stream(remote_model_file, model_file_path)
                    else:
                        if remote_model_path is None or not remote_model_path.strip():
                            raise ValueError(
                                "أدخل مسار الموديل داخل صورة RunPod."
                                if lang == "ar"
                                else "Please provide a model path inside the RunPod image."
                            )
                        model_path_for_worker = remote_model_path.strip()

                    status_text.info(
                        "☁️ رفع الفيديو إلى التخزين ثم إرسال job إلى RunPod Serverless..."
                        if lang == "ar"
                        else "☁️ Uploading to object storage, then submitting RunPod Serverless job..."
                    )
                    progress_bar.progress(5)

                    client = RunPodServerlessClient(
                        api_key=runpod_api_key,
                        endpoint_id=runpod_endpoint_id,
                    )
                    last_upload_ui_update = {"at": 0.0}

                    def update_runpod_progress(
                        percent: float,
                        transferred_bytes: int | None = None,
                        total_bytes: int | None = None,
                        phase: str = "upload",
                    ) -> None:
                        progress_bar.progress(max(5, min(int(percent), 95)))
                        now = time.time()
                        should_update = now - last_upload_ui_update["at"] >= 0.5
                        if total_bytes and transferred_bytes is not None:
                            should_update = should_update or transferred_bytes >= total_bytes
                        if not should_update:
                            return
                        last_upload_ui_update["at"] = now

                        if phase == "video_upload" and total_bytes:
                            upload_percent = min(100.0, (transferred_bytes or 0) / total_bytes * 100)
                            status_text.info(
                                (
                                    f"⬆️ جاري رفع الفيديو: {upload_percent:.1f}% - "
                                    f"{format_bytes(transferred_bytes or 0)} من {format_bytes(total_bytes)}"
                                )
                                if lang == "ar"
                                else (
                                    f"⬆️ Uploading video: {upload_percent:.1f}% - "
                                    f"{format_bytes(transferred_bytes or 0)} of {format_bytes(total_bytes)}"
                                )
                            )
                            upload_detail.caption(
                                "بعد اكتمال الرفع سيتم إرسال job إلى RunPod تلقائياً."
                                if lang == "ar"
                                else "After upload finishes, the job will be submitted to RunPod automatically."
                            )
                        elif phase == "model_upload" and total_bytes:
                            upload_percent = min(100.0, (transferred_bytes or 0) / total_bytes * 100)
                            status_text.info(
                                (
                                    f"⬆️ جاري رفع الموديل: {upload_percent:.1f}% - "
                                    f"{format_bytes(transferred_bytes or 0)} من {format_bytes(total_bytes)}"
                                )
                                if lang == "ar"
                                else (
                                    f"⬆️ Uploading model: {upload_percent:.1f}% - "
                                    f"{format_bytes(transferred_bytes or 0)} of {format_bytes(total_bytes)}"
                                )
                            )
                        elif phase == "video_uploaded":
                            status_text.info(
                                "✅ انتهى رفع الفيديو. جاري تجهيز طلب RunPod..."
                                if lang == "ar"
                                else "✅ Video upload finished. Preparing RunPod request..."
                            )
                        elif phase == "submitting":
                            status_text.info(
                                "☁️ جاري إرسال job إلى RunPod Serverless..."
                                if lang == "ar"
                                else "☁️ Submitting job to RunPod Serverless..."
                            )

                    upload_started_at = time.time()
                    job_id = client.upload_video(
                        video_path=video_path,
                        progress_callback=update_runpod_progress,
                        model_path=model_path_for_worker,
                        model_file_path=model_file_path,
                        analysis_fps=float(analysis_fps_remote),
                        max_frames=int(max_frames_remote) if max_frames_remote else None,
                        resize_width=int(resize_width_remote),
                    )
                    st.session_state.runpod_job_id = job_id
                    upload_elapsed_s = time.time() - upload_started_at

                    status_text.info(
                        (
                            f"🧠 RunPod Job ID: {job_id} - أُرسل خلال {upload_elapsed_s:.1f} ثانية. جاري الانتظار..."
                            if lang == "ar"
                            else f"🧠 RunPod Job ID: {job_id} - Submitted in {upload_elapsed_s:.1f}s. Waiting..."
                        )
                    )
                    progress_bar.progress(15)

                    processing_started_at = time.time()
                    while True:
                        status = client.get_status(job_id)
                        progress_bar.progress(int(status.get("progress", 15)))
                        processing_elapsed_s = time.time() - processing_started_at
                        status_text.info(
                            (
                                f"⏳ {status.get('message', 'processing')} ({processing_elapsed_s:.1f}s)"
                                if lang == "ar"
                                else f"⏳ {status.get('message', 'processing')} ({processing_elapsed_s:.1f}s)"
                            )
                        )
                        if status.get("status") == "completed":
                            break
                        if status.get("status") == "failed":
                            failure_message = status.get("message", "RunPod analysis failed")
                            raise ValueError(f"RunPod job {job_id} failed: {failure_message}")
                        time.sleep(10)

                    results = client.get_results(job_id)
                    video_url = results.get("annotated_video_url")
                    results["output_video"] = video_url

                    st.session_state.analysis_results = results
                    st.session_state.analysis_done = True

                    progress_bar.progress(100)
                    status_text.success("✅ اكتمل التحليل عبر RunPod Serverless!" if lang == "ar" else "✅ RunPod Serverless analysis completed!")

                    if video_url:
                        st.markdown("---")
                        st.subheader("🎬 " + ("الفيديو المعلّم (RunPod)" if lang == "ar" else "Annotated Video (RunPod)"))
                        st.video(video_url)
                        st.markdown(
                            f"[⬇️ {'تحميل الفيديو الناتج' if lang == 'ar' else 'Download output video'}]({video_url})"
                        )

                elif use_real_model:
                    # ===== REAL MODEL ANALYSIS =====
                    local_size_mb = os.path.getsize(video_path) / (1024 * 1024)
                    if local_size_mb > SMALL_UPLOAD_LIMIT_MB:
                        raise ValueError(
                            "المعالجة المحلية مخصصة للمقاطع الصغيرة. استخدم RunPod Serverless أو Remote GPU للمباريات الكبيرة."
                            if lang == "ar"
                            else "Local processing is for small clips. Use RunPod Serverless or Remote GPU for large matches."
                        )
                    
                    # Step 1: Read video
                    status_text.info("📹 جاري قراءة الفيديو..." if lang == "ar" else "📹 Reading video...")
                    progress_bar.progress(5)
                    video_frames = read_video(video_path)
                    total_frames = len(video_frames)
                    
                    if total_frames == 0:
                        raise ValueError("Could not read video")
                    
                    status_text.info(f"✅ تم تحميل {total_frames} فريم" if lang == "ar" else f"✅ Loaded {total_frames} frames")
                    progress_bar.progress(10)
                    
                    # Step 2: Initialize tracker
                    status_text.info("🧠 جاري تحميل النموذج..." if lang == "ar" else "🧠 Loading model...")
                    tracker = Tracker(model_path)
                    progress_bar.progress(15)
                    
                    # Step 3: Detect and track
                    status_text.info("🔍 جاري كشف وتتبع اللاعبين..." if lang == "ar" else "🔍 Detecting and tracking players...")
                    tracks = tracker.get_object_tracks(video_frames)
                    tracker.add_position_to_tracks(tracks)
                    progress_bar.progress(40)
                    
                    # Step 4: Camera movement
                    status_text.info("📹 جاري تقدير حركة الكاميرا..." if lang == "ar" else "📹 Estimating camera movement...")
                    camera_estimator = CameraMovementEstimator(video_frames[0])
                    camera_movements = camera_estimator.get_camera_movement(video_frames)
                    camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movements)
                    progress_bar.progress(50)
                    
                    # Step 5: View transformer
                    status_text.info("🔄 جاري تحويل الإحداثيات..." if lang == "ar" else "🔄 Transforming coordinates...")
                    view_transformer = ViewTransformer()
                    view_transformer.add_transformed_position_to_tracks(tracks)
                    progress_bar.progress(55)
                    
                    # Step 6: Interpolate ball
                    status_text.info("⚽ جاري معالجة مواقع الكرة..." if lang == "ar" else "⚽ Interpolating ball positions...")
                    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
                    progress_bar.progress(60)
                    
                    # Step 7: Speed and distance
                    status_text.info("🏃 جاري حساب السرعة والمسافة..." if lang == "ar" else "🏃 Calculating speed and distance...")
                    speed_estimator = SpeedDistanceEstimator()
                    speed_estimator.add_speed_and_distance_to_tracks(tracks)
                    progress_bar.progress(70)
                    
                    # Step 8: Team assignment
                    status_text.info("👥 جاري تصنيف الفرق..." if lang == "ar" else "👥 Assigning teams...")
                    team_assigner = TeamAssigner()
                    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
                    
                    for frame_num, player_track in enumerate(tracks['players']):
                        for player_id, track in player_track.items():
                            team = team_assigner.get_player_team(
                                video_frames[frame_num],
                                track['bbox'],
                                player_id
                            )
                            tracks['players'][frame_num][player_id]['team'] = team
                            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
                    progress_bar.progress(80)
                    
                    # Step 9: Ball possession
                    status_text.info("⚽ جاري تحديد استحواذ الكرة..." if lang == "ar" else "⚽ Assigning ball possession...")
                    ball_assigner = BallAssigner()
                    team_ball_control = []
                    
                    for frame_num, player_track in enumerate(tracks['players']):
                        ball_bbox = tracks['ball'][frame_num][1]['bbox'] if tracks['ball'][frame_num] else None
                        
                        if ball_bbox:
                            assigned_player = ball_assigner.assign_ball_to_player(player_track, ball_bbox)
                            
                            if assigned_player != -1:
                                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                                team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
                            else:
                                team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
                        else:
                            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
                    
                    team_ball_control = np.array(team_ball_control)
                    progress_bar.progress(85)
                    
                    # Step 10: Draw annotations
                    status_text.info("🎨 جاري رسم التعليمات..." if lang == "ar" else "🎨 Drawing annotations...")
                    output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
                    output_frames = camera_estimator.draw_camera_movement(output_frames, camera_movements)
                    output_frames = speed_estimator.draw_speed_and_distance(output_frames, tracks)
                    progress_bar.progress(90)
                    
                    # Step 11: Save output
                    status_text.info("💾 جاري حفظ الفيديو المعلّم..." if lang == "ar" else "💾 Saving annotated video...")
                    os.makedirs("output_videos", exist_ok=True)
                    output_path = "output_videos/analyzed.mp4"
                    save_video(output_frames, output_path)
                    progress_bar.progress(95)
                    
                    # Calculate stats
                    total_possession = len(team_ball_control)
                    stats = {
                        "total_frames": total_frames,
                        "possession_team1": float((team_ball_control == 1).sum() / total_possession * 100) if total_possession > 0 else 0,
                        "possession_team2": float((team_ball_control == 2).sum() / total_possession * 100) if total_possession > 0 else 0,
                        "total_passes": 0,  # Would need pass detection
                        "total_shots": 0,   # Would need shot detection
                        "player_count": len(tracks['players'][0]) if tracks['players'] else 0,
                    }
                    
                    # Extract player stats
                    player_stats = []
                    if tracks['players'] and len(tracks['players']) > 0:
                        last_frame_idx = len(tracks['players']) - 1
                        for player_id, track in tracks['players'][last_frame_idx].items():
                            player_stats.append({
                                "id": int(player_id),
                                "name": f"Player {player_id}",
                                "team": int(track.get('team', 1)),
                                "distance_km": float(track.get('distance', 0)) / 1000,
                                "max_speed_kmh": float(track.get('speed', 0)),
                            })
                    
                    results = {
                        "stats": stats,
                        "tactical_analysis": {
                            "formation_team1": "4-3-3",
                            "formation_team2": "4-4-2",
                            "pressing_intensity": "medium",
                            "key_moments": []
                        },
                        "player_stats": player_stats,
                        "output_video": output_path,
                    }
                    
                    st.session_state.analysis_results = results
                    st.session_state.analysis_done = True
                    
                    progress_bar.progress(100)
                    status_text.success("✅ اكتمل التحليل بنجاح!" if lang == "ar" else "✅ Analysis complete!")
                    
                    # Show output video
                    st.markdown("---")
                    st.subheader("🎬 " + ("الفيديو المعلّم" if lang == "ar" else "Annotated Video"))
                    st.video(output_path)
                    
                else:
                    # ===== MOCK ANALYSIS (Demo) =====
                    
                    # Step 1: Initialize
                    status_text.info("🔌 جاري الاتصال بالنظام..." if lang == "ar" else "🔌 Connecting to system...")
                    progress_bar.progress(10)
                    time.sleep(0.5)
                    
                    # Step 2: Process video
                    status_text.info("📤 جاري معالجة الفيديو..." if lang == "ar" else "📤 Processing video...")
                    progress_bar.progress(30)
                    time.sleep(0.5)
                    
                    # Step 3: Detect objects
                    status_text.info("🔍 جاري كشف اللاعبين والكرة..." if lang == "ar" else "🔍 Detecting players and ball...")
                    progress_bar.progress(50)
                    time.sleep(0.5)
                    
                    # Step 4: Track
                    status_text.info("📍 جاري تتبع الحركة..." if lang == "ar" else "📍 Tracking movement...")
                    progress_bar.progress(70)
                    time.sleep(0.5)
                    
                    # Step 5: Analyze
                    status_text.info("🧠 جاري التحليل التكتيكي..." if lang == "ar" else "🧠 Analyzing tactics...")
                    progress_bar.progress(90)
                    time.sleep(0.5)
                    
                    # Generate mock results
                    results = {
                        "stats": {
                            "possession_team1": 55.3,
                            "possession_team2": 44.7,
                            "total_passes": 342,
                            "total_shots": 12,
                            "player_count": 22
                        },
                        "tactical_analysis": {
                            "formation_team1": "4-3-3",
                            "formation_team2": "4-4-2",
                            "pressing_intensity": "high",
                            "key_moments": [
                                {"time": "15:32", "event": "goal", "team": 1},
                                {"time": "42:10", "event": "yellow_card", "team": 2},
                                {"time": "67:15", "event": "substitution", "team": 1}
                            ]
                        },
                        "player_stats": [
                            {"id": 1, "name": "Player 1", "team": 1, "distance_km": 9.2, "max_speed_kmh": 28.5},
                            {"id": 2, "name": "Player 2", "team": 1, "distance_km": 10.1, "max_speed_kmh": 31.2},
                            {"id": 3, "name": "Player 3", "team": 2, "distance_km": 8.7, "max_speed_kmh": 26.8},
                            {"id": 4, "name": "Player 4", "team": 2, "distance_km": 9.5, "max_speed_kmh": 29.1},
                            {"id": 5, "name": "Player 5", "team": 1, "distance_km": 11.2, "max_speed_kmh": 33.5},
                            {"id": 6, "name": "Player 6", "team": 2, "distance_km": 8.9, "max_speed_kmh": 27.3}
                        ]
                    }
                    
                    st.session_state.analysis_results = results
                    st.session_state.analysis_done = True
                    
                    progress_bar.progress(100)
                    status_text.success("✅ اكتمل التحليل بنجاح! (وضع تجريبي)" if lang == "ar" else "✅ Analysis complete! (Demo mode)")
                
                time.sleep(1)
                
            except Exception as e:
                st.error(f"❌ خطأ: {str(e)}" if lang == "ar" else f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.session_state.analysis_done = False


# ==================== RESULTS ====================
if st.session_state.analysis_done and st.session_state.analysis_results:
    st.markdown("---")
    
    results = st.session_state.analysis_results
    output_video = results.get("output_video")
    warnings = results.get("warnings", [])
    confidence = results.get("confidence", {})

    if warnings:
        st.warning("\n".join(f"- {warning}" for warning in warnings))

    if output_video:
        st.subheader("🎬 " + ("الفيديو الناتج" if lang == "ar" else "Output Video"))
        st.video(output_video)
        if isinstance(output_video, str) and output_video.startswith(("http://", "https://")):
            st.markdown(
                f"[⬇️ {'تحميل الفيديو الناتج' if lang == 'ar' else 'Download output video'}]({output_video})"
            )
        elif os.path.exists(output_video):
            with open(output_video, "rb") as out_f:
                st.download_button(
                    "⬇️ " + ("تحميل الفيديو الناتج" if lang == "ar" else "Download output video"),
                    data=out_f.read(),
                    file_name=os.path.basename(output_video),
                    mime="video/mp4",
                    use_container_width=True,
                )

    report_json_url = results.get("report_json_url")
    report_csv_url = results.get("report_csv_url")
    if report_json_url or report_csv_url:
        st.markdown("### 📄 " + ("التقارير" if lang == "ar" else "Reports"))
        cols = st.columns(2)
        if report_json_url:
            cols[0].markdown(f"[JSON report]({report_json_url})")
        if report_csv_url:
            cols[1].markdown(f"[CSV player report]({report_csv_url})")
    
    # Tabs
    tab_labels = ["📊 إحصائيات", "🎯 لوحة تكتيكية", "👥 لاعبين", "🤖 تحليل AI"] if lang == "ar" else ["📊 Stats", "🎯 Tactical", "👥 Players", "🤖 AI"]
    tab1, tab2, tab3, tab4 = st.tabs(tab_labels)
    
    # ===== Tab 1: Stats =====
    with tab1:
        st.subheader("📊 " + ("إحصائيات المباراة" if lang == "ar" else "Match Statistics"))
        
        stats = results["stats"]
        tactical = results.get("tactical_analysis", {})

        def metric_value(value, suffix=""):
            if value is None:
                return "غير متاح" if lang == "ar" else "Unavailable"
            if isinstance(value, float):
                return f"{value:.1f}{suffix}"
            return f"{value}{suffix}"
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("استحواذ الفريق 1" if lang == "ar" else "Team 1 Possession", metric_value(stats.get("possession_team1"), "%"))
        with col2:
            st.metric("استحواذ الفريق 2" if lang == "ar" else "Team 2 Possession", metric_value(stats.get("possession_team2"), "%"))
        with col3:
            st.metric("التمريرات" if lang == "ar" else "Passes", metric_value(stats.get("total_passes")))
        with col4:
            st.metric("التسديدات" if lang == "ar" else "Shots", metric_value(stats.get("total_shots")))

        if confidence:
            st.markdown("### " + ("الثقة" if lang == "ar" else "Confidence"))
            st.dataframe(
                pd.DataFrame(
                    [{"metric": key, "confidence": value} for key, value in confidence.items()]
                ),
                use_container_width=True,
                hide_index=True,
            )
        
        st.markdown("---")
        
        # Formations
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ⚔️ " + ("التشكيلات" if lang == "ar" else "Formations"))
            formation_data = {
                "الفريق" if lang == "ar" else "Team": ["Team 1", "Team 2"],
                "التشكيلة" if lang == "ar" else "Formation": [
                    tactical.get("formation_team1") or ("غير متاح" if lang == "ar" else "Unavailable"),
                    tactical.get("formation_team2") or ("غير متاح" if lang == "ar" else "Unavailable")
                ]
            }
            st.dataframe(pd.DataFrame(formation_data), use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### 📈 " + ("الأحداث" if lang == "ar" else "Events"))
            for event in tactical.get("key_moments", []):
                emoji = "⚽" if event["event"] == "goal" else "🟨" if event["event"] == "yellow_card" else "🔄"
                st.write(f"{emoji} {event['time']} - {event['event']} (Team {event['team']})")
        
        # Possession chart
        st.markdown("---")
        st.subheader("📊 " + ("توزيع الاستحواذ" if lang == "ar" else "Possession"))
        
        fig, ax = plt.subplots(figsize=(10, 2))
        if stats.get("possession_team1") is not None and stats.get("possession_team2") is not None:
            possession = [stats["possession_team1"], stats["possession_team2"]]
            colors = ['#2196F3', '#f44336']
            ax.barh([''], [possession[0]], color=colors[0], height=0.5)
            ax.barh([''], [possession[1]], left=[possession[0]], color=colors[1], height=0.5)
            ax.set_xlim(0, 100)
            ax.text(possession[0]/2, 0, f"Team 1\n{possession[0]:.1f}%", ha='center', va='center', color='white', fontweight='bold')
            ax.text(possession[0] + possession[1]/2, 0, f"Team 2\n{possession[1]:.1f}%", ha='center', va='center', color='white', fontweight='bold')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("الاستحواذ غير متاح بثقة كافية." if lang == "ar" else "Possession is not available with enough confidence.")
    
    # ===== Tab 2: Tactical Board =====
    with tab2:
        st.subheader("🎯 " + ("اللوحة التكتيكية" if lang == "ar" else "Tactical Board"))
        if (confidence or {}).get("field_calibration", 0) >= 0.5:
            st.info(
                "معايرة الملعب نجحت، لكن رسم اللوحة من بيانات المباراة سيضاف في الخطوة التالية."
                if lang == "ar"
                else "Field calibration passed, but match-derived board rendering is scheduled for the next step."
            )
        else:
            st.info(
                "اللوحة التكتيكية والتشكيلات معطّلة لأن معايرة الملعب غير موثوقة. هذا أفضل من عرض مواقع وهمية."
                if lang == "ar"
                else "Tactical board and formations are disabled because field calibration is not reliable. This avoids fake positions."
            )
    
    # ===== Tab 3: Players =====
    with tab3:
        st.subheader("👥 " + ("إحصائيات اللاعبين" if lang == "ar" else "Player Statistics"))
        player_stats = results.get("player_stats", [])
        if not player_stats:
            st.info("لا توجد بيانات لاعبين كافية." if lang == "ar" else "No player data available.")
        else:
            df = pd.DataFrame(player_stats)
            rename_map = {
                "id": "ID" if lang == "en" else "الرقم",
                "name": "Name" if lang == "en" else "الاسم",
                "team": "Team" if lang == "en" else "الفريق",
                "frames_seen": "Frames Seen" if lang == "en" else "فريمات مرصودة",
                "distance_km": "Distance (km)" if lang == "en" else "المسافة (كم)",
                "max_speed_kmh": "Max Speed (km/h)" if lang == "en" else "أقصى سرعة (كم/س)",
                "distance_speed_confidence": "Distance/Speed Confidence" if lang == "en" else "ثقة السرعة/المسافة",
            }
            st.dataframe(df.rename(columns=rename_map), use_container_width=True, hide_index=True)

            if df["distance_km"].notna().any():
                st.markdown("---")
                st.subheader("📊 " + ("المسافة المقطوعة" if lang == "ar" else "Distance Covered"))
                fig, ax = plt.subplots(figsize=(10, 4))
                colors = ['#2196F3' if p.get("team") == 1 else '#f44336' for p in player_stats]
                ax.bar([p["name"] for p in player_stats], [p["distance_km"] or 0 for p in player_stats], color=colors)
                ax.set_ylabel("km")
                st.pyplot(fig)
            else:
                st.info(
                    "المسافة والسرعة غير معروضتين لأن معايرة الملعب غير موثوقة."
                    if lang == "ar"
                    else "Distance and speed are hidden because field calibration is not reliable."
                )
    
    # ===== Tab 4: AI Analysis =====
    with tab4:
        st.subheader("🤖 " + ("التحليل الذكي" if lang == "ar" else "AI Analysis"))
        
        # Check if Gemini API is configured
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        has_real_ai = bool(gemini_key)
        
        if not has_real_ai:
            st.warning("""
            ⚠️ **التحليل الذكي غير مفعل**
            
            للحصول على تحليل حقيقي يشاهد الفيديو:
            1. احصل على مفتاح Gemini API من: https://makersuite.google.com/app/apikey
            2. أضفه في ملف `.streamlit/secrets.toml`:
               ```
               GEMINI_API_KEY = "your-key-here"
               ```
            3. أعد تشغيل الواجهة
            """)
        
        # Initialize analyzer
        if has_real_ai:
            analyzer = VideoAIAnalyzer(provider="gemini", api_key=gemini_key)
            st.success("✅ AI جاهز - سيتم تحليل الفريمات الحقيقية!")
        else:
            analyzer = MockVideoAnalyzer()
            st.info("ℹ️ وضع تجريبي - البيانات حقيقية لكن التحليل نصي")
        
        analysis_type = st.selectbox(
            "نوع التحليل" if lang == "ar" else "Analysis Type",
            ["تقرير المباراة", "تحليل الخصم"] if lang == "ar" else ["Match Report", "Opponent Analysis"]
        )
        
        if st.button("توليد التحليل" if lang == "ar" else "Generate Analysis", type="primary"):
            with st.spinner("🧠 جاري التحليل..." if lang == "ar" else "🧠 Analyzing..."):
                
                # Get frames from video if available
                video_frames = []
                if st.session_state.get("video_path") and os.path.exists(st.session_state.video_path):
                    try:
                        video_frames = read_video_sampled(
                            st.session_state.video_path,
                            target_fps=0.2,
                            max_frames=10,
                            resize_width=960,
                        )
                    except:
                        pass
                
                stats = results["stats"]
                
                if "تقرير" in analysis_type or "Report" in analysis_type:
                    if has_real_ai and video_frames:
                        # REAL AI ANALYSIS - looks at actual frames!
                        response = analyzer.analyze_video_real(video_frames, stats, lang)
                        st.markdown(response)
                    else:
                        # Mock analysis - honest about being fake
                        response = analyzer.analyze_video_real([], stats, lang)
                        st.markdown(response)
                        
                elif "خصم" in analysis_type or "Opponent" in analysis_type:
                    if has_real_ai and video_frames:
                        response = analyzer.analyze_opponent_real(video_frames, lang)
                        st.json(response)
                    else:
                        st.info("تحليل الخصم يتطلب AI حقيقي" if lang == "ar" else "Opponent analysis requires real AI")


# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>⚽ Football Tactical AI System</p>", unsafe_allow_html=True)
