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
from src.utils.video_utils import read_video, save_video
from src.api.server_client import ServerClient


# ==================== SESSION STATE ====================
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "remote_server_url" not in st.session_state:
    st.session_state.remote_server_url = os.environ.get("REMOTE_GPU_URL", "http://localhost:8000")


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
        ["Remote GPU", "Local"],
        index=0
    )
    use_remote_gpu = processing_mode == "Remote GPU"

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
                health_resp = requests.get(f"{remote_server_url.rstrip('/')}/health", timeout=10)
                if health_resp.ok:
                    st.success("✅ السيرفر متصل وجاهز" if lang == "ar" else "✅ Server is reachable")
                else:
                    st.error(f"❌ Health check failed: {health_resp.status_code}")
            except Exception as conn_err:
                st.error(f"❌ {conn_err}")

        analysis_fps_remote = st.number_input(
            "FPS للتحليل (أقل = أسرع)" if lang == "ar" else "Analysis FPS (lower = faster)",
            min_value=0.2, max_value=30.0, value=1.0, step=0.2
        )
        max_frames_remote = st.number_input(
            "أقصى عدد فريمات" if lang == "ar" else "Max Frames",
            min_value=300, max_value=20000, value=5400, step=300
        )
        resize_width_remote = st.number_input(
            "عرض الفريم قبل التحليل" if lang == "ar" else "Resize Width",
            min_value=480, max_value=1920, value=960, step=80
        )
    else:
        remote_server_url = ""
        remote_api_key = ""
        analysis_fps_remote = 1.0
        max_frames_remote = 5400
        resize_width_remote = 960

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
if lang == "ar":
    st.header("📹 رفع الفيديو")
    uploaded_file = st.file_uploader("اسحب الفيديو هنا أو انقر للاختيار", type=["mp4", "avi", "mov"])
else:
    st.header("📹 Upload Video")
    uploaded_file = st.file_uploader("Drag video here or click to browse", type=["mp4", "avi", "mov"])


if uploaded_file:
    # Save file
    os.makedirs("temp", exist_ok=True)
    video_path = f"temp/{uploaded_file.name}"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.video_path = video_path
    
    # Show video
    col1, col2 = st.columns([2, 1])
    with col1:
        st.video(uploaded_file)
    with col2:
        st.markdown("### 📋 معلومات الفيديو" if lang == "ar" else "### 📋 Video Info")
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.info(f"{'الحجم' if lang == 'ar' else 'Size'}: {file_size:.1f} MB")
        st.info(f"{'النوع' if lang == 'ar' else 'Type'}: {uploaded_file.type}")
        
        # Show model info
        if use_real_model:
            st.success("🧠 " + ("سيتم استخدام النموذج الحقيقي" if lang == "ar" else "Real model will be used"))
        else:
            st.warning("⚠️ " + ("سيتم استخدام الوضع التجريبي" if lang == "ar" else "Demo mode will be used"))

    remote_model_file = None
    remote_model_path = None
    if use_remote_gpu:
        st.markdown("---")
        st.subheader("🧠 " + ("الموديل على السيرفر GPU" if lang == "ar" else "Model on GPU Server"))
        remote_model_mode = st.radio(
            "طريقة تحديد الموديل" if lang == "ar" else "Model Source",
            ["Model path on server", "Upload model file (.pt)"],
            index=0
        )
        if remote_model_mode == "Model path on server":
            remote_model_path = st.text_input(
                "مسار الموديل على السيرفر" if lang == "ar" else "Model path on server",
                value="models/abdullah_yolov5.pt"
            )
        else:
            remote_model_file = st.file_uploader(
                "ارفع ملف الموديل (.pt)" if lang == "ar" else "Upload model file (.pt)",
                type=["pt"]
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
            
            try:
                if use_remote_gpu:
                    if not remote_server_url.strip():
                        raise ValueError("يرجى إدخال رابط سيرفر GPU" if lang == "ar" else "Please provide GPU server URL")
                    if remote_model_file is None and (remote_model_path is None or not remote_model_path.strip()):
                        raise ValueError(
                            "حدد مسار موديل على السيرفر أو ارفع ملف موديل .pt"
                            if lang == "ar"
                            else "Provide model path on server or upload a .pt model file"
                        )

                    status_text.info("☁️ رفع الفيديو إلى سيرفر GPU..." if lang == "ar" else "☁️ Uploading video to GPU server...")
                    progress_bar.progress(5)

                    model_file_path = None
                    if remote_model_file is not None:
                        os.makedirs("temp", exist_ok=True)
                        model_file_path = f"temp/{remote_model_file.name}"
                        with open(model_file_path, "wb") as mf:
                            mf.write(remote_model_file.getbuffer())

                    client = ServerClient(server_url=remote_server_url, api_key=remote_api_key or None)
                    job_id = client.upload_video(
                        video_path=video_path,
                        progress_callback=lambda p: progress_bar.progress(max(5, min(int(p), 95))),
                        model_path=remote_model_path,
                        model_file_path=model_file_path,
                        analysis_fps=float(analysis_fps_remote),
                        max_frames=int(max_frames_remote),
                        resize_width=int(resize_width_remote),
                    )

                    status_text.info(f"🧠 Job ID: {job_id} - جاري المعالجة..." if lang == "ar" else f"🧠 Job ID: {job_id} - Processing...")
                    progress_bar.progress(15)

                    while True:
                        status = client.get_status(job_id)
                        progress_bar.progress(int(status.get("progress", 15)))
                        status_text.info(
                            f"⏳ {status.get('message', 'processing')}" if lang == "ar" else f"⏳ {status.get('message', 'processing')}"
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

                    st.session_state.analysis_results = results
                    st.session_state.analysis_done = True

                    progress_bar.progress(100)
                    status_text.success("✅ اكتمل التحليل على GPU بنجاح!" if lang == "ar" else "✅ GPU analysis completed successfully!")

                    if video_url:
                        st.markdown("---")
                        st.subheader("🎬 " + ("الفيديو المعلّم (GPU)" if lang == "ar" else "Annotated Video (GPU)"))
                        st.video(video_url)

                elif use_real_model:
                    # ===== REAL MODEL ANALYSIS =====
                    
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
    
    # Tabs
    tab_labels = ["📊 إحصائيات", "🎯 لوحة تكتيكية", "👥 لاعبين", "🤖 تحليل AI"] if lang == "ar" else ["📊 Stats", "🎯 Tactical", "👥 Players", "🤖 AI"]
    tab1, tab2, tab3, tab4 = st.tabs(tab_labels)
    
    # ===== Tab 1: Stats =====
    with tab1:
        st.subheader("📊 " + ("إحصائيات المباراة" if lang == "ar" else "Match Statistics"))
        
        stats = results["stats"]
        tactical = results.get("tactical_analysis", {})
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("استحواذ الفريق 1" if lang == "ar" else "Team 1 Possession", f"{stats['possession_team1']:.1f}%")
        with col2:
            st.metric("استحواذ الفريق 2" if lang == "ar" else "Team 2 Possession", f"{stats['possession_team2']:.1f}%")
        with col3:
            st.metric("التمريرات" if lang == "ar" else "Passes", stats.get("total_passes", 0))
        with col4:
            st.metric("التسديدات" if lang == "ar" else "Shots", stats.get("total_shots", 0))
        
        st.markdown("---")
        
        # Formations
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ⚔️ " + ("التشكيلات" if lang == "ar" else "Formations"))
            formation_data = {
                "الفريق" if lang == "ar" else "Team": ["Team 1", "Team 2"],
                "التشكيلة" if lang == "ar" else "Formation": [
                    tactical.get("formation_team1", "4-3-3"),
                    tactical.get("formation_team2", "4-4-2")
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
        possession = [stats["possession_team1"], stats["possession_team2"]]
        colors = ['#2196F3', '#f44336']
        ax.barh([''], [possession[0]], color=colors[0], height=0.5)
        ax.barh([''], [possession[1]], left=[possession[0]], color=colors[1], height=0.5)
        ax.set_xlim(0, 100)
        ax.text(possession[0]/2, 0, f"Team 1\n{possession[0]:.1f}%", ha='center', va='center', color='white', fontweight='bold')
        ax.text(possession[0] + possession[1]/2, 0, f"Team 2\n{possession[1]:.1f}%", ha='center', va='center', color='white', fontweight='bold')
        ax.axis('off')
        st.pyplot(fig)
    
    # ===== Tab 2: Tactical Board =====
    with tab2:
        st.subheader("🎯 " + ("اللوحة التكتيكية" if lang == "ar" else "Tactical Board"))
        
        board = TacticalBoard()
        fig, ax = board.draw_pitch()
        
        # Sample positions
        players = {
            1: {"position": (15, 34), "team": 1, "has_ball": False},
            2: {"position": (25, 20), "team": 1, "has_ball": True},
            3: {"position": (25, 48), "team": 1, "has_ball": False},
            4: {"position": (40, 34), "team": 1, "has_ball": False},
            5: {"position": (85, 34), "team": 2, "has_ball": False},
            6: {"position": (75, 20), "team": 2, "has_ball": False},
            7: {"position": (75, 48), "team": 2, "has_ball": False},
        }
        
        board.draw_players(ax, players, ball_position=(25, 34))
        st.pyplot(fig)
        
        # Formation diagrams
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("⚔️ Team 1 (4-3-3)")
            fig1, ax1 = board.create_formation_diagram("4-3-3", team=1)
            st.pyplot(fig1)
        with col2:
            st.subheader("⚔️ Team 2 (4-4-2)")
            fig2, ax2 = board.create_formation_diagram("4-4-2", team=2)
            st.pyplot(fig2)
    
    # ===== Tab 3: Players =====
    with tab3:
        st.subheader("👥 " + ("إحصائيات اللاعبين" if lang == "ar" else "Player Statistics"))
        
        df = pd.DataFrame(results["player_stats"])
        df.columns = ["ID", "Name", "Team", "Distance (km)", "Max Speed (km/h)"] if lang == "en" else ["الرقم", "الاسم", "الفريق", "المسافة (كم)", "أقصى سرعة (كم/س)"]
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Chart
        st.markdown("---")
        st.subheader("📊 " + ("المسافة المقطوعة" if lang == "ar" else "Distance Covered"))
        
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['#2196F3' if p["team"] == 1 else '#f44336' for p in results["player_stats"]]
        ax.bar([p["name"] for p in results["player_stats"]], [p["distance_km"] for p in results["player_stats"]], color=colors)
        ax.set_ylabel("km")
        st.pyplot(fig)
    
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
                        video_frames = read_video(st.session_state.video_path)
                        # Limit frames for AI (too many = slow/expensive)
                        if len(video_frames) > 50:
                            # Take 10 evenly spaced frames
                            indices = np.linspace(0, len(video_frames)-1, 10, dtype=int)
                            video_frames = [video_frames[i] for i in indices]
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
