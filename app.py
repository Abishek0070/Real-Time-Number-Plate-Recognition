import os
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

# Page Configuration
st.set_page_config(
    page_title="ANPR Sentinel | AI License Plate Recognition",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism & Cyber Automotive Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Outfit:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main container */
    .main {
        background: radial-gradient(circle at top left, #0d131f 0%, #070a10 100%);
    }

    /* App Header Banner */
    .hero-header {
        background: linear-gradient(135deg, rgba(16, 26, 46, 0.8) 0%, rgba(9, 14, 26, 0.9) 100%);
        border: 1px solid rgba(0, 229, 255, 0.25);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 229, 255, 0.15);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        color: #8fa0bc;
        font-size: 0.95rem;
        margin-top: 6px;
        font-weight: 400;
    }

    /* License Plate Realistic Badge */
    .plate-badge-container {
        display: inline-flex;
        align-items: stretch;
        background: #fbfbfd;
        border: 3px solid #111;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.8);
        margin: 10px 0;
    }

    .plate-flag {
        background: #003399;
        color: #ffcc00;
        padding: 4px 10px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 1px;
    }

    .plate-string {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem;
        font-weight: 800;
        color: #111;
        padding: 6px 18px;
        letter-spacing: 3px;
        display: flex;
        align-items: center;
        text-shadow: 0 1px 0 rgba(255, 255, 255, 0.6);
    }

    /* Metric Card */
    .stat-card {
        background: rgba(18, 27, 44, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .stat-card:hover {
        border-color: rgba(0, 229, 255, 0.4);
        transform: translateY(-2px);
    }

    .stat-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00f2fe;
    }

    .stat-lbl {
        color: #7e8da4;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
        color: #8fa0bc;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.15) 0%, rgba(79, 172, 254, 0.15) 100%) !important;
        border-color: #00f2fe !important;
        color: #00f2fe !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing SSD MobileNet & EasyOCR Engine...")
def get_engine():
    """Initializes and caches the ANPR Engine in memory for high-speed inference."""
    return ANPREngine()


engine = get_engine()

# Initialize Session State for Historical Detections
if "session_history" not in st.session_state:
    st.session_state.session_history = []


# Sidebar Settings & Model Parameters
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/license-plate.png", width=64)
    st.title("ANPR Control Hub")
    st.caption("SSD MobileNet V2 + EasyOCR")
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
        help="Minimum area ratio a text box must occupy within the plate crop to be recognized (filters noise)."
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
    st.subheader("💾 Logging & Storage")
    auto_save = st.checkbox("Auto-save Detections to Disk", value=True, help="Saves cropped plate image and appends to detection_results.csv")

    st.divider()
    st.subheader("🖥️ Hardware & Status")
    st.success("🟢 Detector: Ready (ckpt-11)")
    st.success("🟢 OCR: Ready (EasyOCR PyTorch)")
    st.info("⚡ Execution Mode: CPU High-Performance")

    if st.button("🗑️ Clear Session History", use_container_width=True):
        st.session_state.session_history = []
        st.rerun()


# Hero Banner
st.markdown("""
<div class="hero-header">
    <div>
        <h1 class="hero-title">🚗 ANPR Sentinel AI</h1>
        <div class="hero-subtitle">Real-Time Automatic Number Plate Detection & Character Extraction</div>
    </div>
    <div style="text-align: right;">
        <span style="background: rgba(0, 242, 254, 0.1); border: 1px solid #00f2fe; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; color: #00f2fe; font-weight: 600;">
            ● SSD MobileNet V2 + EasyOCR
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# Main Application Navigation Tabs
tab_image, tab_video, tab_camera, tab_logs, tab_info = st.tabs([
    "📸 Image Recognition",
    "📹 Video Stream",
    "🎥 Live Camera",
    "📊 Detection History",
    "ℹ️ Architecture & Info"
])


# -------------------------------------------------------------
# TAB 1: Image Recognition
# -------------------------------------------------------------
with tab_image:
    col_input, col_preset = st.columns([2, 1])

    with col_input:
        upload_choice = st.radio(
            "Select Image Source:",
            ["Upload Your Vehicle Photo", "Test with Sample Vehicles"],
            horizontal=True
        )

    image_to_process = None

    if upload_choice == "Upload Your Vehicle Photo":
        uploaded_file = st.file_uploader(
            "Upload car / motorcycle image (JPG, PNG, JPEG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"]
        )
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    else:
        sample_folder = "sample_images"
        sample_files = [f for f in os.listdir(sample_folder) if f.endswith((".jpg", ".png", ".jpeg"))] if os.path.exists(sample_folder) else []
        
        with col_preset:
            selected_sample = st.selectbox("Choose demo preset:", sample_files)
        
        if selected_sample:
            sample_path = os.path.join(sample_folder, selected_sample)
            image_to_process = cv2.imread(sample_path)

    if image_to_process is not None:
        st.divider()

        with st.spinner("Processing image through detection and OCR pipeline..."):
            start_time = time.time()
            results = engine.detect(
                image_to_process,
                detection_threshold=det_thresh,
                region_threshold=region_thresh,
                ocr_confidence_threshold=ocr_thresh
            )
            elapsed_ms = (time.time() - start_time) * 1000

        # Layout: Visual Display vs Results Overview
        col_img_orig, col_img_det = st.columns(2)

        with col_img_orig:
            st.markdown("##### 🚙 Original Image")
            st.image(cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col_img_det:
            st.markdown("##### 🎯 Detection & Bounding Box")
            st.image(cv2.cvtColor(results["annotated_image"], cv2.COLOR_BGR2RGB), use_container_width=True)

        st.caption(f"⚡ Inference Latency: {elapsed_ms:.1f} ms | Found {results['count']} license plate(s)")

        # Render Detected Plates Showcase
        if results["count"] > 0:
            st.subheader(f"🔍 Extracted License Plates ({results['count']})")
            
            plate_cols = st.columns(min(results["count"], 3))
            
            for idx, plate in enumerate(results["plates"]):
                with plate_cols[idx % len(plate_cols)]:
                    st.markdown(f"""
                    <div class="plate-badge-container">
                        <div class="plate-flag">IND</div>
                        <div class="plate-string">{plate['plate_text']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Show Plate Crop
                    st.image(
                        cv2.cvtColor(plate["plate_crop"], cv2.COLOR_BGR2RGB),
                        caption=f"Plate Crop #{idx + 1}",
                        width=240
                    )

                    c1, c2 = st.columns(2)
                    c1.metric("Detector Score", f"{plate['detection_score'] * 100:.1f}%")
                    c2.metric("OCR Confidence", f"{plate['ocr_confidence'] * 100:.1f}%")

                    ymin, xmin, ymax, xmax = plate["box"]
                    st.caption(f"📍 Bounding Box: [{ymin}, {xmin}, {ymax}, {xmax}]")

                    # Handle Auto-save or Manual Log
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
# TAB 2: Video Stream Analysis
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
            frame_skip = st.slider("Frame Skip Rate (Process every N frames)", min_value=1, max_value=30, value=5, help="Increases processing speed by sampling every N-th frame.")
        with col_v2:
            st.info(f"Video Stats: {total_frames} frames | ~{fps} FPS | Duration: {total_frames / fps:.1f}s")

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

                    # Update preview
                    annotated_rgb = cv2.cvtColor(results["annotated_image"], cv2.COLOR_BGR2RGB)
                    st_frame.image(annotated_rgb, channels="RGB", use_container_width=True)

                    progress_pct = min(1.0, frame_idx / max(total_frames, 1))
                    st_progress.progress(progress_pct)
                    st_stats.markdown(f"**Processed Frame:** {frame_idx}/{total_frames} | **Unique Plates Discovered:** `{len(unique_plates)}` {list(unique_plates)[-5:]}")

                frame_idx += 1

            cap.release()
            st.success(f"🎉 Video Analysis Complete! Found {len(unique_plates)} unique vehicle license plate(s): {', '.join(unique_plates)}")


# -------------------------------------------------------------
# TAB 3: Live Camera Input
# -------------------------------------------------------------
with tab_camera:
    st.markdown("#### 🎥 Live Camera Snapshot Recognition")
    st.caption("Point your webcam or phone camera towards a vehicle license plate and capture a snapshot.")

    camera_photo = st.camera_input("Capture license plate")

    if camera_photo is not None:
        bytes_data = camera_photo.getvalue()
        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner("Analyzing camera snapshot..."):
            results = engine.detect(
                cv_img,
                detection_threshold=det_thresh,
                region_threshold=region_thresh,
                ocr_confidence_threshold=ocr_thresh
            )

        col_cam1, col_cam2 = st.columns(2)
        with col_cam1:
            st.markdown("##### 📷 Snapshot")
            st.image(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col_cam2:
            st.markdown("##### 🎯 Detection Result")
            st.image(cv2.cvtColor(results["annotated_image"], cv2.COLOR_BGR2RGB), use_container_width=True)

        if results["count"] > 0:
            st.success(f"Recognized {results['count']} Plate(s):")
            for plate in results["plates"]:
                st.markdown(f"""
                <div class="plate-badge-container">
                    <div class="plate-flag">IND</div>
                    <div class="plate-string">{plate['plate_text']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No license plates detected in camera snapshot. Adjust distance or lighting.")


# -------------------------------------------------------------
# TAB 4: Detection History & Logs
# -------------------------------------------------------------
with tab_logs:
    st.markdown("#### 📊 Real-Time Detection Registry & Logs")

    # Load persistent CSV logs if available
    persistent_records = []
    if os.path.exists("detection_results.csv"):
        try:
            with open("detection_results.csv", "r", encoding="utf-8") as f:
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

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.markdown(f"""
    <div class="stat-card">
        <div class="stat-lbl">Session Detections</div>
        <div class="stat-val">{len(st.session_state.session_history)}</div>
    </div>
    """, unsafe_allow_html=True)

    col_m2.markdown(f"""
    <div class="stat-card">
        <div class="stat-lbl">Persistent Logged Plates</div>
        <div class="stat-val">{len(persistent_records)}</div>
    </div>
    """, unsafe_allow_html=True)

    col_m3.markdown(f"""
    <div class="stat-card">
        <div class="stat-lbl">Output Folder Assets</div>
        <div class="stat-val">{len(os.listdir('Detection_Images')) if os.path.exists('Detection_Images') else 0}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.session_history:
        st.markdown("##### 🕒 Session Detection Feed")
        df_session = pd.DataFrame([
            {
                "Timestamp": item["timestamp"],
                "Recognized Plate": item["plate_text"],
                "Detector Score": item["det_score"],
                "OCR Confidence": item["ocr_conf"],
                "Saved File": item["image_name"]
            }
            for item in st.session_state.session_history
        ])
        st.dataframe(df_session, use_container_width=True)

        # Download Button
        csv_data = df_session.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Session Report (CSV)",
            data=csv_data,
            file_name=f"anpr_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No plates recognized during this active session yet. Run an image or video detection to populate records.")

    # Show Persistent Records Table
    if persistent_records:
        with st.expander("📂 View Full Disk Log (`detection_results.csv`)", expanded=False):
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
        - **Fine-Tuning Dataset**: Kaggle Car License Plate Dataset (433 vehicle images)
        - **Checkpoint Restored**: `ckpt-11` (10,000+ training steps)
        - **Feature Extractor**: MobileNetV2 with Feature Pyramid Network (FPN)
        - **Loss Functions**: Smooth L1 (Localization) + Sigmoid Focal Loss (Classification)
        """)

    with col_a2:
        st.markdown("""
        ##### 2️⃣ Text Recognition (OCR) Stage
        - **Model**: `EasyOCR (PyTorch Engine)`
        - **Architecture**: CRAFT (Character Region Awareness for Text Detection) + ResNet/CRNN + CTC Decoder
        - **Text Filtering**: Custom relative bounding box area filter `filter_text()`
        - **Character Set**: Alphanumeric Latin (English)
        - **Confidence Metric**: Multi-token averaged probability
        """)

    st.markdown("---")
    st.markdown("##### 📁 Project Pipeline Schema")
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
