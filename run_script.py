"""
run_video_with_tracking.py

Usage:
    python run_video_with_tracking.py test_files/input_video.mp4

What this script does:
- Runs YOLOv8 + ByteTrack with specialized FOD model
- Detects: Metal scraps, Metal parts, Rubber, Luggage tags, Screws, Plastic
- Ultra-high sensitivity (confidence=0.08) to catch ALL objects
- Frame skipping for faster processing while catching all objects
- Provides size-based classification hints
- Assigns unique IDs and counts unique objects
- Saves annotated output video
"""

import sys
import os
import cv2
from ultralytics import YOLO

# -----------------------------
# 1. INPUT VALIDATION
# -----------------------------

if len(sys.argv) < 2:
    print("Usage: python run_video_with_tracking.py <video_path>")
    sys.exit(1)

VIDEO_PATH = sys.argv[1]
assert os.path.exists(VIDEO_PATH), "Video file not found"

# -----------------------------
# 2. LOAD MODEL
# -----------------------------

MODEL_PATH = "runs/detect/runs_fod/yolov8n_fod_v1_2/weights/best.pt"  # Newer FOD model
model = YOLO(MODEL_PATH)

print(f"ℹ️ Model loaded: {MODEL_PATH}")
print(f"🎯 Detects: Metal scraps, Metal parts, Metal plates, Rubber, Luggage tags, Screws, Plastic")
print(f"🏷️ Model classes: {model.names}")

# -----------------------------
# 3. OPEN VIDEO
# -----------------------------

cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# -----------------------------
# 4. OUTPUT VIDEO SETUP
# -----------------------------

FRAME_SKIP = 2  # Process every 2nd frame for efficiency

os.makedirs("output", exist_ok=True)
output_path = "output/fod_tracked_fast.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps/FRAME_SKIP, (width, height))

# -----------------------------
# 5. TRACKING STATE
# -----------------------------

# Track unique FOD object IDs
unique_fod_ids = set()
total_detections = 0
object_sizes = []  # Track sizes for classification hints
filtered_out = 0  # Track filtered detections

# Size filtering
MIN_OBJECT_AREA = 100  # Minimum pixels²
MAX_OBJECT_AREA = 500000  # Maximum pixels² (ignore floor/walls)

frame_idx = 0
total_video_frames = 0

print("▶ Processing with smart object filtering...")
print(f"⚡ Frame skip: {FRAME_SKIP} (processing every {FRAME_SKIP}nd frame)")
print(f"🎯 Confidence threshold: 0.20 (balanced for real objects)")
print(f"🛡️ Size filter: {MIN_OBJECT_AREA}-{MAX_OBJECT_AREA}px² (ignores floor/noise)")

# -----------------------------
# 6. PROCESS FRAMES
# -----------------------------

for result in model.track(
    source=VIDEO_PATH,
    conf=0.20,  # Balanced threshold for real objects
    imgsz=640,  # High resolution
    iou=0.4,  # Standard NMS
    max_det=300,  # Standard max
    tracker="bytetrack.yaml",
    stream=True,
    verbose=False,
    vid_stride=FRAME_SKIP  # Skip frames for efficiency
):
    frame_idx += 1
    total_video_frames += FRAME_SKIP

    frame = result.orig_img
    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:
        # Filter boxes by size
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

    # Draw boxes + IDs + Labels
    annotated_frame = result.plot()

    # Overlay FOD count
    cv2.putText(
        annotated_frame,
        f"FOD Objects: {len(unique_fod_ids)} | Detections: {total_detections}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2
    )
    
    # Add FOD types label
    cv2.putText(
        annotated_frame,
        "Types: Metal, Rubber, Plastic, Screws, Tags",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2
    )
    
    # Show filtered count
    if filtered_out > 0:
        cv2.putText(
            annotated_frame,
            f"Filtered: {filtered_out} (floor/noise)",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (128, 128, 128),
            1
        )

    out.write(annotated_frame)

    if frame_idx % 50 == 0:
        print(f"Frame {total_video_frames} (proc {frame_idx}) | FOD: {len(unique_fod_ids)} | Valid: {total_detections} | Filtered: {filtered_out}")

# -----------------------------
# 7. CLEANUP
# -----------------------------

out.release()
cap.release()

print("\n✅ FOD Detection complete!")
print(f"📁 Output video: {output_path}")
print(f"📊 Total Unique FOD Objects: {len(unique_fod_ids)}")
print(f"✅ Valid Detections: {total_detections}")
print(f"🛡️ Filtered Out (floor/noise): {filtered_out}")
print(f"⚡ Frames Processed: {frame_idx} (skipped {FRAME_SKIP-1} of every {FRAME_SKIP})")
print(f"🎯 Detected FOD Types: Metal scraps, Metal parts, Metal plates, Rubber, Luggage tags, Screws, Plastic")

if object_sizes:
    avg_size = sum(object_sizes) / len(object_sizes)
    small_objects = len([s for s in object_sizes if s < avg_size/2])
    print(f"\n📊 Size Analysis:")
    print(f"  - Average object size: {avg_size:.0f}px²")
    print(f"  - Small objects (likely screws/tags): {small_objects}")
    print(f"  - Medium-Large objects (likely metal/rubber/plastic): {len(object_sizes) - small_objects}")
