import os
# Ensure legacy Keras, disable oneDNN conflicts, and suppress C++ warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io
import time
import tempfile
from datetime import datetime
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import streamlit as st

from anpr_engine import ANPREngine

# Page Configuration - auto sidebar for mobile, wide on desktop
st.set_page_config(
    page_title="Smart Number Plate Reader",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom Styling (Dark Glassmorphism, Cyber Automotive Theme, Full Mobile Responsive)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;700;800&family=Outfit:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main background */
    .stApp {
        background: radial-gradient(circle at top left, #0e1524 0%, #070a12 100%);
    }

    /* Mobile Container Padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 1200px;
    }

    /* Responsive Hero Header */
    .hero-header {
        background: linear-gradient(135deg, rgba(16, 26, 46, 0.85) 0%, rgba(9, 14, 26, 0.95) 100%);
        border: 1px solid rgba(0, 229, 255, 0.25);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 28px -10px rgba(0, 229, 255, 0.18);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
    }

    .hero-title {
        font-size: 1.9rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        color: #8fa0bc;
        font-size: 0.9rem;
        margin-top: 6px;
        font-weight: 400;
    }

    .hero-badge {
        background: rgba(0, 242, 254, 0.1);
        border: 1px solid #00f2fe;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        color: #00f2fe;
        font-weight: 600;
        white-space: nowrap;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* License Plate Badge - Fully Responsive */
    .plate-badge-container {
        display: inline-flex;
        align-items: stretch;
        background: #fbfbfd;
        border: 3px solid #111;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.8);
        margin: 10px 0;
        max-width: 100%;
    }

    .plate-flag {
        background: #003399;
        color: #ffcc00;
        padding: 4px 8px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 1px;
        min-width: 32px;
    }

    .plate-string {
        font-family: 'JetBrains Mono', monospace;
        font-size: clamp(1.1rem, 3.5vw, 1.55rem);
        font-weight: 800;
        color: #111;
        padding: 6px 14px;
        letter-spacing: 2px;
        display: flex;
        align-items: center;
        text-shadow: 0 1px 0 rgba(255, 255, 255, 0.6);
        word-break: break-word;
    }

    /* Result Card Container */
    .result-card {
        background: rgba(18, 27, 44, 0.7);
        border: 1px solid rgba(0, 229, 255, 0.2);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 16px;
        backdrop-filter: blur(10px);
    }

    /* Metric Stat Cards Grid */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 12px;
        margin-bottom: 16px;
    }

    .stat-card {
        background: rgba(18, 27, 44, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 16px;
        backdrop-filter: blur(10px);
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .stat-card:hover {
        border-color: rgba(0, 229, 255, 0.4);
        transform: translateY(-2px);
    }

    .stat-val {
        font-size: clamp(1.4rem, 3vw, 1.8rem);
        font-weight: 800;
        color: #00f2fe;
    }

    .stat-lbl {
        color: #7e8da4;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 4px;
    }

    /* Mobile Responsive Tab Bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        overflow-x: auto;
        flex-wrap: nowrap;
        -webkit-overflow-scrolling: touch;
        scrollbar-width: none;
        padding-bottom: 4px;
    }

    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 8px 14px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #8fa0bc;
        white-space: nowrap;
        min-height: 42px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.18) 0%, rgba(79, 172, 254, 0.18) 100%) !important;
        border-color: #00f2fe !important;
        color: #00f2fe !important;
    }

    /* Touch-friendly buttons & inputs */
    button, input, select, textarea {
        min-height: 44px;
        touch-action: manipulation;
    }

    /* Mobile Media Query Breakpoints */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem !important;
            padding-left: 0.6rem !important;
            padding-right: 0.6rem !important;
        }

        .hero-header {
            flex-direction: column;
            align-items: flex-start;
            padding: 16px;
            gap: 10px;
            border-radius: 12px;
        }

        .hero-title {
            font-size: 1.45rem;
        }

        .hero-subtitle {
            font-size: 0.82rem;
        }

        .hero-badge {
            font-size: 0.75rem;
            padding: 4px 10px;
        }

        .plate-string {
            font-size: 1.25rem;
            letter-spacing: 1.5px;
            padding: 5px 10px;
        }

        .stat-card {
            padding: 10px 12px;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 6px 12px;
            font-size: 0.82rem;
        }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing SSD MobileNet & EasyOCR Engine...")
def get_engine():
    """Initializes and caches the ANPR Engine in memory for high-speed inference."""
    return ANPREngine()


engine = get_engine()

# Initialize Session State
if "session_history" not in st.session_state:
    st.session_state.session_history = []


# Sidebar Settings & Model Parameters
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/license-plate.png", width=56)
    st.title("Plate Scanner")
    st.caption("AI Vehicle Detection & Text Reader")
    st.divider()

    st.subheader("⚙️ Detection Thresholds")
    det_thresh = st.slider(
        "Detector Confidence",
        min_value=0.10,
        max_value=1.00,
        value=0.45,
        step=0.05,
        help="Minimum confidence required for the SSD MobileNet to propose a license plate box."
    )

    region_thresh = st.slider(
        "ROI Region Size Ratio",
        min_value=0.05,
        max_value=0.60,
        value=0.18,
        step=0.02,
        help="Minimum area ratio a text box must occupy within the plate crop to filter out noise."
    )

    ocr_thresh = st.slider(
        "OCR Text Confidence",
        min_value=0.10,
        max_value=0.95,
        value=0.20,
        step=0.05,
        help="Minimum confidence threshold for the OCR reader to accept text tokens."
    )

    st.divider()
    auto_save = st.checkbox("Auto-save Detections to Disk", value=True)

    st.divider()
    st.subheader("🖥️ Status")
    st.success("🟢 Model: Ready (ckpt-11)")
    st.success("🟢 OCR: Ready (EasyOCR)")

    if st.button("🗑️ Clear Session History", use_container_width=True):
        st.session_state.session_history = []
        st.rerun()


# Responsive Hero Header
st.markdown("""
<div class="hero-header">
    <div>
        <h1 class="hero-title">🚗 Smart Number Plate Reader</h1>
        <div class="hero-subtitle">Instant Vehicle License Plate Detection & Scanner</div>
    </div>
    <div>
        <span class="hero-badge">
            ● Mobile & Cloud Ready
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# Main Tabs with Mobile-Friendly Labels & Touch Scrolling
tab_image, tab_camera, tab_video, tab_logs, tab_info = st.tabs([
    "📸 Photo Upload",
    "📱 Live Camera",
    "📹 Video Stream",
    "📊 History",
    "ℹ️ Info"
])


# -------------------------------------------------------------
# TAB 1: Image Recognition (Mobile & Desktop Responsive)
# -------------------------------------------------------------
with tab_image:
    upload_choice = st.radio(
        "Choose Image Source:",
        ["📁 Upload Your Photo", "🚗 Demo Vehicle Presets"],
        horizontal=True
    )

    image_to_process = None

    if upload_choice == "📁 Upload Your Photo":
        uploaded_file = st.file_uploader(
            "Select vehicle image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Take or select a photo from your phone or PC."
        )
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    else:
        sample_folder = "sample_images"
        sample_files = [f for f in os.listdir(sample_folder) if f.endswith((".jpg", ".png", ".jpeg"))] if os.path.exists(sample_folder) else []
        
        selected_sample = st.selectbox("Choose demo preset vehicle:", sample_files)
        if selected_sample:
            sample_path = os.path.join(sample_folder, selected_sample)
            image_to_process = cv2.imread(sample_path)

    if image_to_process is not None:
        st.markdown("<br>", unsafe_allow_html=True)

        with st.spinner("Analyzing image through neural detection and OCR..."):
            start_time = time.time()
            results = engine.detect(
                image_to_process,
                detection_threshold=det_thresh,
                region_threshold=region_thresh,
                ocr_confidence_threshold=ocr_thresh
            )
            elapsed_ms = (time.time() - start_time) * 1000

        # Mobile-adaptive display: Tabs for side-by-side or stacked view
        view_mode = st.radio("Display View:", ["🎯 Detection Result", "🚙 Original Photo", "↔️ Side-by-Side"], horizontal=True)

        if view_mode == "🎯 Detection Result":
            st.image(cv2.cvtColor(results["annotated_image"], cv2.COLOR_BGR2RGB), use_container_width=True)
        elif view_mode == "🚙 Original Photo":
            st.image(cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            col_orig, col_det = st.columns(2)
            with col_orig:
                st.caption("Original")
                st.image(cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col_det:
                st.caption("Detection")
                st.image(cv2.cvtColor(results["annotated_image"], cv2.COLOR_BGR2RGB), use_container_width=True)

        st.caption(f"⚡ Latency: {elapsed_ms:.1f} ms | Found {results['count']} license plate(s)")

        # Render Detected Plates Showcase
        if results["count"] > 0:
            st.markdown("### 🔍 Detected License Plates")

            for idx, plate in enumerate(results["plates"]):
                st.markdown(f"""
                <div class="result-card">
                    <div style="font-size: 0.8rem; color: #8fa0bc; text-transform: uppercase; font-weight: 600;">
                        License Plate #{idx + 1}
                    </div>
                    <div class="plate-badge-container">
                        <div class="plate-flag">IND</div>
                        <div class="plate-string">{plate['plate_text']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_crop, col_metrics = st.columns([1, 1])

                with col_crop:
                    st.image(
                        cv2.cvtColor(plate["plate_crop"], cv2.COLOR_BGR2RGB),
                        caption="Extracted Plate Crop",
                        use_container_width=True
                    )

                with col_metrics:
                    m1, m2 = st.columns(2)
                    m1.metric("Detection Confidence", f"{plate['detection_score'] * 100:.1f}%")
                    m2.metric("OCR Accuracy", f"{plate['ocr_confidence'] * 100:.1f}%")

                    ymin, xmin, ymax, xmax = plate["box"]
                    st.caption(f"📍 Box Coordinates: [{ymin}, {xmin}, {ymax}, {xmax}]")

                # Auto-save
                if auto_save:
                    saved_name = engine.save_result(plate["plate_crop"], plate["plate_text"])
                    st.session_state.session_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "plate_text": plate["plate_text"],
                        "det_score": f"{plate['detection_score'] * 100:.1f}%",
                        "ocr_conf": f"{plate['ocr_confidence'] * 100:.1f}%",
                        "image_name": saved_name,
                        "crop": plate["plate_crop"]
                    })
        else:
            st.warning("⚠️ No license plates detected above the selected confidence threshold. Try lowering the 'Detector Confidence' slider in the sidebar.")


# -------------------------------------------------------------
# TAB 2: Live Camera Input (Native Smartphone Camera Integration)
# -------------------------------------------------------------
with tab_camera:
    st.markdown("#### 📱 Mobile & Webcam License Plate Scanner")
    st.caption("Capture directly using your smartphone camera or PC webcam for instantaneous license plate recognition.")

    camera_photo = st.camera_input("Point camera at vehicle license plate")

    if camera_photo is not None:
        bytes_data = camera_photo.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner("Recognizing license plate from camera snapshot..."):
            results = engine.detect(
                cv_img,
                detection_threshold=det_thresh,
                region_threshold=region_thresh,
                ocr_confidence_threshold=ocr_thresh
            )

        st.image(cv2.cvtColor(results["annotated_image"], cv2.COLOR_BGR2RGB), use_container_width=True)

        if results["count"] > 0:
            st.success(f"Recognized {results['count']} Plate(s):")
            for idx, plate in enumerate(results["plates"]):
                st.markdown(f"""
                <div class="result-card">
                    <div class="plate-badge-container">
                        <div class="plate-flag">IND</div>
                        <div class="plate-string">{plate['plate_text']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                c1.metric("Detection Score", f"{plate['detection_score'] * 100:.1f}%")
                c2.metric("OCR Confidence", f"{plate['ocr_confidence'] * 100:.1f}%")
        else:
            st.warning("No license plates detected in camera snapshot. Adjust distance or lighting.")


# -------------------------------------------------------------
# TAB 3: Video Stream Analysis
# -------------------------------------------------------------
with tab_video:
    st.markdown("#### 📹 Video File License Plate Scanner")
    st.caption("Upload a traffic or dashcam video file (MP4, AVI, MOV) for frame-by-frame analysis.")

    video_file = st.file_uploader("Upload video file", type=["mp4", "avi", "mov", "mkv"], key="video_uploader")

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

        col_v1, col_v2 = st.columns([1, 1])
        with col_v1:
            frame_skip = st.slider("Frame Skip Rate", min_value=1, max_value=30, value=5, help="Process every N-th frame for speed.")
        with col_v2:
            st.info(f"Video: {total_frames} frames | ~{fps} FPS | Duration: {total_frames / fps:.1f}s")

        if st.button("🚀 Start Video Analysis", use_container_width=True):
            st_frame = st.empty()
            st_progress = st.progress(0)
            st_stats = st.empty()

            frame_idx = 0
            unique_plates = set()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    results = engine.detect(
                        frame,
                        detection_threshold=det_thresh,
                        region_threshold=region_thresh,
                        ocr_confidence_threshold=ocr_thresh
                    )

                    for p in results["plates"]:
                        clean_text = p["plate_text"]
                        if clean_text not in ["UNKNOWN", "NO TEXT DETECTED"]:
                            unique_plates.add(clean_text)

                    annotated_rgb = cv2.cvtColor(results["annotated_image"], cv2.COLOR_BGR2RGB)
                    st_frame.image(annotated_rgb, channels="RGB", use_container_width=True)

                    progress_pct = min(1.0, frame_idx / max(total_frames, 1))
                    st_progress.progress(progress_pct)
                    st_stats.markdown(f"**Frame:** {frame_idx}/{total_frames} | **Unique Plates:** `{len(unique_plates)}` {list(unique_plates)[-3:]}")

                frame_idx += 1

            cap.release()
            st.success(f"🎉 Complete! Found {len(unique_plates)} unique plate(s): {', '.join(unique_plates)}")


# -------------------------------------------------------------
# TAB 4: Detection History & Logs (Responsive Cards)
# -------------------------------------------------------------
with tab_logs:
    st.markdown("#### 📊 Detection Registry & Stats")

    persistent_records = []
    if os.path.exists("detection_results.csv"):
        try:
            with open("detection_results.csv", "r", encoding="utf-8") as f:
                import csv
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        img_name = row[0]
                        text = row[1]
                        ts = row[2] if len(row) > 2 else "Archived"
                        persistent_records.append({
                            "timestamp": ts,
                            "plate_text": text,
                            "image_name": img_name
                        })
        except Exception:
            pass

    # Responsive Stat Grid
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-val">{len(st.session_state.session_history)}</div>
            <div class="stat-lbl">Active Session</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{len(persistent_records)}</div>
            <div class="stat-lbl">Total Logged</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{len(os.listdir('Detection_Images')) if os.path.exists('Detection_Images') else 0}</div>
            <div class="stat-lbl">Plate Assets</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.session_history:
        st.markdown("##### 🕒 Session Log")
        df_session = pd.DataFrame([
            {
                "Time": item["timestamp"],
                "Plate Number": item["plate_text"],
                "Detector": item["det_score"],
                "OCR": item["ocr_conf"],
                "File": item["image_name"]
            }
            for item in st.session_state.session_history
        ])
        st.dataframe(df_session, use_container_width=True)

        csv_data = df_session.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Session CSV",
            data=csv_data,
            file_name=f"number_plate_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No plates recognized during this active session yet.")

    if persistent_records:
        with st.expander("📂 View Full Persistent Disk Log (`detection_results.csv`)", expanded=False):
            df_disk = pd.DataFrame(persistent_records)
            st.dataframe(df_disk, use_container_width=True)


# -------------------------------------------------------------
# TAB 5: Architecture & Model Specs
# -------------------------------------------------------------
with tab_info:
    st.markdown("#### 🧠 Two-Stage Deep Learning Architecture")
    
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("""
        ##### 1️⃣ Object Detection Stage
        - **Model**: `SSD MobileNet V2 FPNLite 320x320`
        - **Framework**: TensorFlow Object Detection API (TF2)
        - **Pretrained Checkpoint**: `ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8`
        - **Fine-Tuning Dataset**: Kaggle Car License Plate Dataset (433 images)
        - **Checkpoint Restored**: `ckpt-11` (10,000+ training steps)
        - **Feature Extractor**: MobileNetV2 with Feature Pyramid Network (FPN)
        - **Loss Functions**: Smooth L1 + Sigmoid Focal Loss
        """)

    with col_a2:
        st.markdown("""
        ##### 2️⃣ Text Recognition (OCR) Stage
        - **Model**: `EasyOCR (PyTorch Engine)`
        - **Architecture**: CRAFT + ResNet/CRNN + CTC Decoder
        - **Text Filtering**: Custom relative bounding box area filter `filter_text()`
        - **Character Set**: Alphanumeric Latin (English)
        - **Confidence Metric**: Multi-token averaged probability
        """)

    st.markdown("---")
    st.markdown("##### 📁 System Pipeline Schema")
    st.code("""
[Input Image / Camera Frame]
           │
           ▼
[SSD MobileNet V2 Detector] ────► Proposes Bounding Box (ymin, xmin, ymax, xmax)
           │
           ▼
[Crop Plate ROI]
           │
           ▼
[EasyOCR Neural Recognizer] ────► Character Sequence & Probabilities
           │
           ▼
[Area & Confidence Filtering] ──► Final Cleaned Plate String (e.g. "6526 JHD")
           │
           ├─► Overlay on UI with Realistic Automotive Plate Card
           ├─► Save Crop to Detection_Images/<UUID>.jpg
           └─► Record into detection_results.csv
    """, language="text")
