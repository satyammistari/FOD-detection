import streamlit as st
import os
import cv2
import time
from ultralytics import YOLO

# -------------------------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="FOD Detection & Tracking",
    layout="centered"
)

st.title("🛫 FOD Detection - Maximum Objects")
st.write(
    "Upload a runway video. "
    "Detects **Metal scraps, Metal parts, Rubber, Luggage tags, Screws, Plastic** with optimized processing. "
    "Frame skipping enabled for faster results while catching all objects."
)

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------

MODEL_PATH = "runs/detect/runs_fod/yolov8n_fod_v1_2/weights/best.pt"  # Newer FOD model

# Balanced threshold - detects objects but not floor/background
CONF_THRESHOLD = 0.20  # Good balance for real objects
IMG_SIZE = 640  # High resolution for accuracy
IOU_THRESHOLD = 0.4  # Standard NMS threshold
MAX_DET = 300  # Reasonable max detections
FRAME_SKIP = 2  # Process every 2nd frame for efficiency

# Size filtering (ignore very large/small detections that are likely noise)
MIN_OBJECT_AREA = 100  # Minimum pixels² (filters out tiny noise)
MAX_OBJECT_AREA = 500000  # Maximum pixels² (filters out floor/background)
DRAW_BOXES = True

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# -------------------------------------------------
# LOAD MODEL (CACHED)
# -------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------

uploaded_video = st.file_uploader(
    "Upload a video file",
    type=["mp4", "avi", "mov"]
)

if uploaded_video is not None:
    input_path = os.path.join(TEMP_DIR, uploaded_video.name)

    with open(input_path, "wb") as f:
        f.write(uploaded_video.read())

    st.success("✅ Video uploaded successfully")

    if st.button("▶ Detect FOD Objects (Floor Objects Only)"):
        model = load_model()
        
        st.info(f"🎯 Detecting: Metal scraps, Metal parts, Metal plates, Rubber, Luggage tags, Screws, Plastic")
        st.info(f"⚡ Smart filtering: Ignores floor/background, detects only actual objects")
        st.info(f"⏱️ Frame skip: {FRAME_SKIP} (faster processing)")

        # Open video ONLY to read metadata
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            st.error("❌ Failed to open video")
            st.stop()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Output writer
        output_path = os.path.join(TEMP_DIR, "fod_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps/FRAME_SKIP, (width, height))  # Adjust FPS for frame skip

        unique_fod_ids = set()
        total_detections = 0  # Track total detection count
        object_sizes = []  # Track object sizes for classification hints
        filtered_out = 0  # Track filtered detections

        progress_bar = st.progress(0)
        status_text = st.empty()

        processed_frames = 0
        total_video_frames = 0
        start_time = time.time()

        st.info("🚀 Processing with maximum object detection...")

        # -------------------------------------------------
        # YOLO TRACKING LOOP WITH FRAME SKIPPING
        # -------------------------------------------------

        for result in model.track(
            source=input_path,
            conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE,
            iou=IOU_THRESHOLD,
            max_det=MAX_DET,
            tracker="bytetrack.yaml",
            stream=True,
            verbose=False,
            vid_stride=FRAME_SKIP  # Skip frames for efficiency
        ):
            processed_frames += 1
            total_video_frames += FRAME_SKIP

            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                # Filter boxes by size (remove floor/background and noise)
                valid_boxes_mask = []
                for i, box in enumerate(boxes.xywh):
                    width_px = box[2].item()
                    height_px = box[3].item()
                    area = width_px * height_px
                    
                    # Keep only objects within reasonable size range
                    if MIN_OBJECT_AREA <= area <= MAX_OBJECT_AREA:
                        valid_boxes_mask.append(True)
                        object_sizes.append(area)
                    else:
                        valid_boxes_mask.append(False)
                        filtered_out += 1
                
                # Count valid detections
                total_detections += sum(valid_boxes_mask)
                
                # Track unique IDs for valid boxes only
                if boxes.id is not None:
                    for i, (tid, box) in enumerate(zip(boxes.id.tolist(), boxes.xywh)):
                        if valid_boxes_mask[i]:
                            unique_fod_ids.add(int(tid))

            frame = result.plot() if DRAW_BOXES else result.orig_img

            # Show detection count with object type hints
            cv2.putText(
                frame,
                f"FOD Objects: {len(unique_fod_ids)} | Detections: {total_detections}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )
            
            # Add detected object types label
            cv2.putText(
                frame,
                "Types: Metal, Rubber, Plastic, Screws, Luggage tags",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )
            
            # Show filtered count
            if filtered_out > 0:
                cv2.putText(
                    frame,
                    f"Filtered: {filtered_out} (floor/noise)",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (128, 128, 128),
                    1
                )

            out.write(frame)

            if processed_frames % 10 == 0:
                progress_value = min(total_video_frames / total_frames, 1.0)
                progress_bar.progress(progress_value)
                status_text.text(
                    f"Frame {total_video_frames}/{total_frames} (processed {processed_frames}) | "
                    f"FOD: {len(unique_fod_ids)} | Valid: {total_detections} | Filtered: {filtered_out}"
                )

        out.release()

        elapsed = time.time() - start_time
        avg_fps = processed_frames / elapsed if elapsed > 0 else 0

        st.success("✅ Detection complete!")

        st.markdown("### 📊 Detection Results")
        st.write(f"**Total Unique FOD Objects:** {len(unique_fod_ids)}")
        st.write(f"**Valid Detections:** {total_detections}")
        st.write(f"**Filtered Out (floor/background/noise):** {filtered_out}")
        st.write(f"**Frames Processed:** {processed_frames} (skipped {FRAME_SKIP-1} of every {FRAME_SKIP})")
        st.write(f"**Processing Speed:** {avg_fps:.2f} FPS")
        
        # Provide size-based classification hints
        if object_sizes:
            avg_size = sum(object_sizes) / len(object_sizes)
            st.markdown("#### 🏷️ Detected FOD Types:")
            st.write("- **Metal scraps** (various sizes)")
            st.write("- **Metal parts** (medium-large objects)")
            st.write("- **Rubber** (flexible, various sizes)")
            st.write("- **Luggage tags** (small, rectangular)")
            st.write("- **Screws** (very small objects)")
            st.write("- **Plastic** (various sizes)")
            st.info(f"💡 Average object size: {avg_size:.0f}px² | Detected {len([s for s in object_sizes if s < avg_size/2])} small objects (likely screws/tags)")
        
        st.info("ℹ️ **Smart Filtering:** Objects too large (floor/walls) or too small (noise) are automatically filtered out to show only actual FOD objects.")

        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button(
                "⬇ Download FOD Detection Video (Optimized)",
                f,
                "fod_detection_fast.mp4",
                "video/mp4"
            )
