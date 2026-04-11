#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:55:02 2026

@author: manish

Real-time video tracking using Ultralytics YOLO with built-in ByteTrack

"""

import os
import cv2
import numpy as np
import time
from collections import defaultdict, deque
from ultralytics import YOLO

# ================= HARDCODED CONFIGURATION =================
# Input/Output paths
VIDEO_PATH = "/home/manish/Mee/codes/vayudh_task/video1.avi"
IMAGE_DIR = "/home/manish/Mee/codes/vayudh_task/main/test/VisDrone2019-VID-test-dev/sequences/uav0000297_02761_v"
MODEL_PATH = "/home/manish/Mee/codes/vayudh_task/visdrone_finetune22/weights/best.pt"
OUTPUT_DIR = "/home/manish/Mee/codes/vayudh_task/main/output/byteTrck"

# Processing mode: Set to True for video, False for image directory
PROCESS_VIDEO = True

# ByteTrack configuration file (create this file)
BYTETRACK_CONFIG = "/home/manish/Mee/codes/vayudh_task/bytetrack_custom.yaml"

# YOLO detection parameters
CONF = 0.45               # Detection confidence threshold
IOU = 0.30                # IoU threshold for NMS
CLASSES = [0]  # COCO classes (all)

# Output options
SAVE_VIDEO = True
SAVE_DEBUG = False
SHOW_TRAILS = True
TRAIL_LEN = 30
NO_DISPLAY = False

# Colors (BGR format)
BOX_COLOR = (0, 255, 0)          # Green
TRAIL_COLOR = (255, 255, 255)    # White
TEXT_BG_COLOR = (0, 255, 0)      # Green background for text
TEXT_COLOR = (255, 255, 255)     # White text
FPS_COLOR = (0, 255, 0)          # Green FPS text

# Display settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
ALPHA = 0.1  # Transparency for bounding boxes
# ===========================================================


def create_output_directory(output_path):
    """Create a directory if it doesn't exist."""
    os.makedirs(output_path, exist_ok=True)
    print(f"Directory created at: {output_path}")


def draw_trail(frame, trail_points, color, trail_thick=2):
    """Draw a fading polyline trail behind an object."""
    pts = list(trail_points)
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        faded = tuple(int(c * alpha) for c in color)
        thickness = max(1, int(trail_thick * alpha))
        cv2.line(frame, pts[i - 1], pts[i], faded, thickness)


if __name__ == "__main__":
    # Create output directories
    create_output_directory(OUTPUT_DIR)
    
    debug_dir = None
    if SAVE_DEBUG:
        debug_dir = os.path.join(OUTPUT_DIR, "debug")
        create_output_directory(debug_dir)
    
    output_video_dir = os.path.join(OUTPUT_DIR, "videos")
    if SAVE_VIDEO:
        create_output_directory(output_video_dir)

    # Check if ByteTrack config exists
    if not os.path.exists(BYTETRACK_CONFIG):
        print(f"Warning: ByteTrack config not found at {BYTETRACK_CONFIG}")
        print("Using default tracking parameters")
        use_custom_tracker = False
    else:
        use_custom_tracker = True
        print(f"Using ByteTrack config: {BYTETRACK_CONFIG}")

    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    
    # Warm up model
    dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy_input, verbose=False)

    # Initialize video capture or image list
    if PROCESS_VIDEO:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")
        
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video info: {width}x{height} @ {fps_src:.2f} fps, {total_frames} frames")
    else:
        # Process image directory
        image_files = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) 
                              if f.endswith(('png', 'jpg', 'jpeg'))])
        if not image_files:
            raise FileNotFoundError(f"No images found in: {IMAGE_DIR}")
        
        # Read first image to get dimensions
        sample_img = cv2.imread(image_files[0])
        height, width = sample_img.shape[:2]
        fps_src = 30  # Default for image sequences
        total_frames = len(image_files)
        print(f"Image directory info: {width}x{height}, {total_frames} images")

    # Video writer
    out = None
    if SAVE_VIDEO:
        out_path = os.path.join(output_video_dir, "bytetrack_output_vid.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(out_path, fourcc, fps_src, (width, height))
        print(f"Output video will be saved to: {out_path}")

    # Tracking state
    trails = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
    results_txt = []
    frame_id = 0
    prev_time = time.time()
    fps_disp = 0  # Initialize FPS

    print("Starting ByteTrack tracking... Press 'ESC' or 'q' to quit")

    while True:
        # Get frame (either from video or image directory)
        if PROCESS_VIDEO:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            if frame_id >= len(image_files):
                break
            frame = cv2.imread(image_files[frame_id])
            if frame is None:
                break
        
        frame_id += 1
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
        
        # FPS calculation with exponential smoothing
        now = time.time()
        if frame_id == 1:
            fps_disp = 1.0 / (now - prev_time + 1e-9)
        else:
            instant_fps = 1.0 / (now - prev_time + 1e-9)
            fps_disp = 0.9 * fps_disp + 0.1 * instant_fps
        prev_time = now

        # Run YOLO with built-in ByteTrack
        if use_custom_tracker:
            results = model.track(
                frame,
                tracker=BYTETRACK_CONFIG,
                persist=True,
                conf=CONF,
                iou=IOU,
                verbose=False,
                classes=CLASSES
            )
        else:
            # Use default tracking (no custom config)
            results = model.track(
                frame,
                persist=True,
                conf=CONF,
                iou=IOU,
                verbose=False,
                classes=CLASSES
            )
        
        # Extract tracking results
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            # Create overlay for transparent boxes
            overlay = frame.copy()
            
            # Draw tracked objects
            for i, (box, tid, conf) in enumerate(zip(boxes, track_ids, confs)):
                x1, y1, x2, y2 = map(int, box)
                tid = int(tid)
                
                # Save MOT format results
                results_txt.append(
                    f"{frame_id},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{conf:.2f},-1,-1,-1\n"
                )
                
                # Draw filled overlay box (transparent green)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), BOX_COLOR, -1)
                # Draw solid green border
                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
                
                # Draw ID label with green background
                label = f"ID:{tid}"
                (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 2)
                lx, ly = x1, max(y1 - 5, th + 5)
                cv2.rectangle(frame, (lx, ly - th - 4), (lx + tw + 4, ly + 2), TEXT_BG_COLOR, -1)
                cv2.putText(frame, label, (lx + 2, ly - 2), FONT, 0.55, TEXT_COLOR, 2)
                
                # Update and draw white trail
                if SHOW_TRAILS:
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    trails[tid].append((cx, cy))
                    draw_trail(frame, trails[tid], TRAIL_COLOR, trail_thick=2)
            
            # Blend transparent boxes
            cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0, frame)
            
            active_tracks = len(track_ids)
        else:
            active_tracks = 0

        # Draw FPS
        fps_label = f"FPS: {fps_disp:.1f}"
        cv2.putText(frame, fps_label, (12, 32), FONT, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, fps_label, (12, 32), FONT, 0.9, FPS_COLOR, 2, cv2.LINE_AA)
        
        # Draw frame counter
        frame_label = f"Frame: {frame_id}/{total_frames}"
        cv2.putText(frame, frame_label, (12, 65), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, frame_label, (12, 65), FONT, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Draw tracker name
        tracker_label = "Tracker: ByteTrack (Ultralytics)"
        cv2.putText(frame, tracker_label, (12, 95), FONT, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, tracker_label, (12, 95), FONT, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        # Display
        if not NO_DISPLAY:
            cv2.imshow("ByteTrack Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

        # Save video
        if SAVE_VIDEO and out:
            out.write(frame)

        # Progress indicator
        if frame_id % 100 == 0:
            print(f"Processed {frame_id}/{total_frames} frames, {active_tracks} active tracks, FPS: {fps_disp:.1f}")

    # Cleanup
    if PROCESS_VIDEO:
        cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    # Save results
    results_path = os.path.join(OUTPUT_DIR, "bytetrack_output.txt")
    with open(results_path, "w") as f:
        f.writelines(results_txt)

    print(f"\n{'='*50}")
    print(f"BYTETRACK TRACKING COMPLETED")
    print(f"{'='*50}")
    print(f"Processed frames: {frame_id}")
    print(f"Results saved to: {results_path}")
    if SAVE_VIDEO:
        print(f"Video saved to: {out_path}")
    if SAVE_DEBUG:
        print(f"Debug images saved to: {debug_dir}")
    print(f"Average FPS: {fps_disp:.1f}")
    print(f"{'='*50}")