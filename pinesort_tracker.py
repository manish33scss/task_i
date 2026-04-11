#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:42:37 2026

@author: manish 

Real-time video tracking using PineSORT - Optimized hardcoded version

"""

import os
import cv2
import numpy as np
import time
from collections import defaultdict, deque
from ultralytics import YOLO
from tracker.pinesort import PineSORT
from gamma_correction import GammaCorrector

# Monkey patch to fix the debug saving issue
from tracker import utils
original_plot = utils.plot_and_save_image_with_detections

def patched_plot(cv_image, bounding_boxes, detections, output_path, alpha=0.1, thickness=3):
    # Only save if output_path is a valid string (not None, not empty)
    if output_path and output_path != "":
        original_plot(cv_image, bounding_boxes, detections, output_path, alpha, thickness)

utils.plot_and_save_image_with_detections = patched_plot


def create_output_directory(output_path):
    #Create a directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    print(f"Directory created at: {output_path}")


def draw_trail(frame, trail_points, color, trail_thick=2):
    #Draw a fading polyline trail behind an object
    pts = list(trail_points)
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        faded = tuple(int(c * alpha) for c in color)
        thickness = max(1, int(trail_thick * alpha))
        cv2.line(frame, pts[i - 1], pts[i], faded, thickness)


# ================= HARDCODED CONFIGURATION =================
# Input/Output paths
VIDEO_PATH = "/home/manish/Mee/codes/vayudh_task/video5.mp4"
IMAGE_DIR = "/home/manish/Mee/codes/vayudh_task/main/test/VisDrone2019-VID-test-dev/sequences/uav0000297_02761_v"
MODEL_PATH = "/home/manish/Mee/codes/vayudh_task/visdrone_finetune22/weights/best.pt"
OUTPUT_DIR = "/home/manish/Mee/codes/vayudh_task/main/output/pinesort"

# Processing mode: Set to True for video, False for image directory
PROCESS_VIDEO = True  # True: process video file, False: process image directory

# PineSORT parameters 
DET_THRESH = 0.45
MIN_DET_THRESH = 0.30
MAX_AGE = 5
MIN_HITS = 1
FIRST_IOU_THRESHOLD = 0.30
SECOND_IOU_THRESHOLD = 0.30
THIRD_IOU_THRESHOLD = 0.10
OVERLAP_IOU_THRESHOLD = 0.10
DELTA_T = 3
INERTIA = 0.2
ASSO_FUNC = "eiou"
USE_BYTE = True
CAMERA_COMPENSATION = "orb"

# Output options
SAVE_VIDEO = True
SAVE_DEBUG = False
SHOW_TRAILS = True
TRAIL_LEN = 30
NO_DISPLAY = False

# Colors (BGR format)00FF00
BOX_COLOR = (0, 255, 0)          # Green
TRAIL_COLOR = (255, 255, 255)    # White
TEXT_BG_COLOR = (0, 255, 0)      # Green background for text
TEXT_COLOR = (255, 255, 255)     # White text
FPS_COLOR = (0, 255, 0)          # Green FPS text

# Display settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
ALPHA = 0.1  # Transparency for bounding boxes
# ===========================================================


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

    # Warm up YOLO model (initialize GPU)
    print("Loading YOLO model..")
    model = YOLO(MODEL_PATH)
    dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy_input, verbose=False)  # Warm up
    
    print("Initializing PineSort")
    tracker = PineSORT(
        det_thresh=DET_THRESH,
        min_det_thresh=MIN_DET_THRESH,
        max_age=MAX_AGE,
        min_hits=MIN_HITS,
        first_iou_threshold=FIRST_IOU_THRESHOLD,
        second_iou_threshold=SECOND_IOU_THRESHOLD,
        third_iou_threshold=THIRD_IOU_THRESHOLD,
        overlap_iou_threshold=OVERLAP_IOU_THRESHOLD,
        delta_t=DELTA_T,
        inertia=INERTIA,
        asso_func=ASSO_FUNC,
        camera_compensation=CAMERA_COMPENSATION,
        use_byte=USE_BYTE
    )

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
        
        # Read first image to get dimensions
        sample_img = cv2.imread(image_files[0])
        height, width = sample_img.shape[:2]
        fps_src = 30  # Default for image sequences
        total_frames = len(image_files)
        print(f"Image directory info: {width}x{height}, {total_frames} images")

    # Video writer
    out = None
    if SAVE_VIDEO:
        out_path = os.path.join(output_video_dir, "vid_5_pineTrack.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(out_path, fourcc, fps_src, (width, height))
        print(f"Output video will be saved to: {out_path}")

    # Tracking state
    trails = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
    results_txt = []
    frame_id = 0
    prev_time = time.time()
    fps_disp = 0  # Initialize FPS

    print("Starting tracking... Press 'ESC' or 'q' to quit")
    gc = GammaCorrector()

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
        frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_CUBIC)
        frame, gamma_val = gc.process(frame)

        # FPS calculation with exponential smoothing
        now = time.time()
        if frame_id == 1:
            fps_disp = 1.0 / (now - prev_time + 1e-9)
        else:
            instant_fps = 1.0 / (now - prev_time + 1e-9)
            fps_disp = 0.9 * fps_disp + 0.1 * instant_fps
        prev_time = now

        # YOLO inference with optimized detection parsing
        prediction = model.predict(frame, verbose=False)
        boxes = prediction[0].boxes
        
        if boxes is not None and len(boxes) > 0:
            # Extract xyxy and confidence
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy().reshape(-1, 1)
            
            # Apply detection threshold
            mask = conf.flatten() > DET_THRESH
            xyxy = xyxy[mask]
            conf = conf[mask]
            
            if len(xyxy) > 0:
                detections = np.concatenate((xyxy, conf), axis=1)
            else:
                detections = np.empty((0, 5))
        else:
            detections = np.empty((0, 5))

        # Prepare debug path
        path_save_debug = None
        if SAVE_DEBUG and debug_dir:
            path_save_debug = os.path.join(debug_dir, f"frame_{frame_id:06d}.jpg")

        # Update tracker
        online_targets = tracker.update(detections, frame, path_save_debug=path_save_debug)

        # Create overlay for transparent boxes
        overlay = frame.copy()

        # Draw tracked objects
        for t in online_targets:
            x1, y1, x2, y2 = t[0], t[1], t[2], t[3]
            tid = int(t[4])

            # Save MOT format results
            results_txt.append(
                f"{frame_id},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},-1,-1,-1,-1\n"
            )
            
            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])

            # Draw filled overlay box 
            cv2.rectangle(overlay, (x1i, y1i), (x2i, y2i), BOX_COLOR, -1)
            
            cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), BOX_COLOR, 2)

            # Draw ID label 
            label = f"ID:{tid}"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 2)
            lx, ly = x1i, max(y1i - 5, th + 5)
            cv2.rectangle(frame, (lx, ly - th - 4), (lx + tw + 4, ly + 2), TEXT_BG_COLOR, -1)
            cv2.putText(frame, label, (lx + 2, ly - 2), FONT, 0.55, TEXT_COLOR, 2)

            # Update and draw white trail
            if SHOW_TRAILS:
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                trails[tid].append((cx, cy))
                draw_trail(frame, trails[tid], TRAIL_COLOR, trail_thick=2)

        # Blend transparent boxes
        cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0, frame)

        # Draw FPS
        fps_label = f"FPS: {fps_disp:.1f}"
        cv2.putText(frame, fps_label, (12, 32), FONT, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, fps_label, (12, 32), FONT, 0.9, FPS_COLOR, 2, cv2.LINE_AA)
        
        
        frame_label = f"Frame: {frame_id}/{total_frames}"
        cv2.putText(frame, frame_label, (12, 65), FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, frame_label, (12, 65), FONT, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        # Display
        if not NO_DISPLAY:
            cv2.imshow("PineSORT Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

        # Save video
        if SAVE_VIDEO and out:
            out.write(frame)

        # Progress indicator
        if frame_id % 100 == 0:
            print(f"Processed {frame_id}/{total_frames} frames, {len(online_targets)} active tracks, FPS: {fps_disp:.1f}")

    # Cleanup
    if PROCESS_VIDEO:
        cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    # Save results
    results_path = os.path.join(OUTPUT_DIR, "tracking_res_pinesort.txt")
    with open(results_path, "w") as f:
        f.writelines(results_txt)

    print(f"\n{'='*50}")
    print(f"TRACKING COMPLETED")
    print(f"{'='*50}")
    print(f"Processed frames: {frame_id}")
    print(f"Results saved to: {results_path}")
    if SAVE_VIDEO:
        print(f"Video saved to: {out_path}")
    if SAVE_DEBUG:
        print(f"Debug images saved to: {debug_dir}")
    print(f"Average FPS: {fps_disp:.1f}")
    print(f"{'='*50}")
