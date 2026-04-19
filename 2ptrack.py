#!/usr/bin/env python3
# Docker-compatible version — optimized for Jetson Nano

import os
import cv2
import numpy as np
import time
from collections import defaultdict, deque
from ultralytics import YOLO

# Import your custom modules
from tracker.pinesort import PineSORT
from gamma_correction import GammaCorrector

# ================= DOCKER PATHS =================
engine_path = "/workspace/PR_25/vayudh/visdrone_finetune22/weights/best_container.engine"
if os.path.exists(engine_path):
    MODEL_PATH = engine_path
    print(f"Using TensorRT engine: {MODEL_PATH}")
else:
    MODEL_PATH = "/workspace/PR_25/vayudh/visdrone_finetune22/weights/best.pt"
    print(f"Engine not found, falling back to: {MODEL_PATH}")

VIDEO_PATH   = "/workspace/PR_25/vayudh/video2.avi"
IMAGE_DIR    = "/workspace/VisDrone2019-MOT-val/sequences/uav0000339_00001_v"
OUTPUT_DIR   = "/ultralytics/runs/tracking_output/"
filename     = "uav_traffic2"

PROCESS_VIDEO = True

# ── Resolution (must match TRT engine input size) ────────────────────────────
INFER_W, INFER_H = 640, 480   # ← changed from 640×480 to 640×640

# ── PineSORT parameters ──────────────────────────────────────────────────────
DET_THRESH           = 0.40
MIN_DET_THRESH       = 0.30
MAX_AGE              = 30
MIN_HITS             = 1
FIRST_IOU_THRESHOLD  = 0.30
SECOND_IOU_THRESHOLD = 0.30
THIRD_IOU_THRESHOLD  = 0.10
OVERLAP_IOU_THRESHOLD= 0.10
DELTA_T              = 3
INERTIA              = 0.2
ASSO_FUNC            = "eiou"
USE_BYTE             = False
CAMERA_COMPENSATION  = "orb"

# ── Output options ───────────────────────────────────────────────────────────
SAVE_VIDEO  = True
SAVE_DEBUG  = False
SHOW_TRAILS = True
TRAIL_LEN   = 30

# ── Overlay settings ─────────────────────────────────────────────────────────
TRACKER_NAME  = "PineSORT"
BOX_COLOR     = (0, 255, 0)
TRAIL_COLOR   = (255, 255, 255)
TEXT_BG_COLOR = (0, 255, 0)
TEXT_COLOR    = (255, 255, 255)
FPS_COLOR     = (0, 255, 0)
INFO_COLOR    = (0, 200, 255)
FONT          = cv2.FONT_HERSHEY_SIMPLEX
ALPHA         = 0.1          # box fill opacity

# ── FPS smoothing ─────────────────────────────────────────────────────────────
FPS_SMOOTH    = 0.9          # higher = smoother but slower to react

# ── Resize interpolation: INTER_LINEAR is ~3× faster than INTER_CUBIC ────────
RESIZE_INTERP = cv2.INTER_LINEAR


def create_output_directory(path):
    os.makedirs(path, exist_ok=True)
    print(f"Directory ready: {path}")


def draw_trail(frame, trail_points, color, trail_thick=2):
    pts = list(trail_points)
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        faded = tuple(int(c * alpha) for c in color)
        cv2.line(frame, pts[i - 1], pts[i], faded, max(1, int(trail_thick * alpha)))


def put_text_shadowed(frame, text, pos, scale, color, thickness=1):
    """Draw text with a dark shadow for readability on any background."""
    x, y = pos
    cv2.putText(frame, text, (x + 1, y + 1), FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y),         FONT, scale, color,     thickness,     cv2.LINE_AA)


if __name__ == "__main__":
    create_output_directory(OUTPUT_DIR)

    debug_dir = None
    if SAVE_DEBUG:
        debug_dir = os.path.join(OUTPUT_DIR, "debug")
        create_output_directory(debug_dir)

    if SAVE_VIDEO:
        create_output_directory(os.path.join(OUTPUT_DIR, "videos"))

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading YOLO model …")
    model = YOLO(MODEL_PATH)

    # Warm-up with correct input size (avoids slow first inference)
    dummy = np.zeros((INFER_H, INFER_W, 3), dtype=np.uint8)
    for _ in range(2):
        model.predict(dummy, imgsz=INFER_W, verbose=False)
    print("Model warmed up.")

    # ── Init tracker ──────────────────────────────────────────────────────────
    print("Initialising PineSORT …")
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
        use_byte=USE_BYTE,
    )

    # ── Video / image source ──────────────────────────────────────────────────
    if PROCESS_VIDEO:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

        # Use MJPEG decoder — faster on Jetson for some codecs
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        fps_src     = cap.get(cv2.CAP_PROP_FPS) or 30
        src_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Source video: {src_w}×{src_h} @ {fps_src:.1f} fps  ({total_frames} frames)")
    else:
        image_files = sorted(
            [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
             if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        )
        if not image_files:
            raise FileNotFoundError(f"No images in: {IMAGE_DIR}")
        sample = cv2.imread(image_files[0])
        src_h, src_w = sample.shape[:2]
        fps_src      = 30
        total_frames = len(image_files)
        print(f"Image dir: {src_w}×{src_h}, {total_frames} frames")

    # ── Video writer at 640×640 ───────────────────────────────────────────────
    out = None
    if SAVE_VIDEO:
        out_path = os.path.join(OUTPUT_DIR, f"pineTrack_{filename}.avi")
        fourcc   = cv2.VideoWriter_fourcc(*"XVID")
        out      = cv2.VideoWriter(out_path, fourcc, fps_src, (INFER_W, INFER_H))
        print(f"Saving to: {out_path}  ({INFER_W}×{INFER_H})")

    # ── State ─────────────────────────────────────────────────────────────────
    trails    = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
    results_txt = []
    frame_id  = 0
    fps_disp  = 0.0
    gc        = GammaCorrector()

    # Timing buckets for profiling
    t_read = t_pre = t_inf = t_track = t_draw = t_write = 0.0

    print("Tracking started …")
    loop_start = time.time()

    while True:
        # ── Read ──────────────────────────────────────────────────────────────
        t0 = time.time()
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
        t_read += time.time() - t0

        frame_id += 1

        # ── Pre-process ───────────────────────────────────────────────────────
        t0 = time.time()
        # Resize to 640×640 (INTER_LINEAR ≈ 3× faster than INTER_CUBIC)
        frame = cv2.resize(frame, (INFER_W, INFER_H), interpolation=RESIZE_INTERP)
        frame, gamma_val = gc.process(frame)
        t_pre += time.time() - t0

        # ── FPS (measured before inference so it captures the full loop) ──────
        now = time.time()
        if frame_id == 1:
            fps_disp = 1.0
        else:
            instant = 1.0 / (now - prev_time + 1e-9)
            fps_disp = FPS_SMOOTH * fps_disp + (1 - FPS_SMOOTH) * instant
        prev_time = now

        # ── YOLO inference ────────────────────────────────────────────────────
        t0 = time.time()
        prediction = model.predict(frame,  verbose=False)
        t_inf += time.time() - t0

        boxes = prediction[0].boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy().reshape(-1, 1)
            mask = conf.flatten() > DET_THRESH
            xyxy, conf = xyxy[mask], conf[mask]
            detections = np.concatenate((xyxy, conf), axis=1) if len(xyxy) > 0 else np.empty((0, 5))
        else:
            detections = np.empty((0, 5))

        # ── Tracker update ────────────────────────────────────────────────────
        t0 = time.time()
        dbg_path = (os.path.join(debug_dir, f"frame_{frame_id:06d}.jpg")
                    if SAVE_DEBUG and debug_dir else None)
        online_targets = tracker.update(detections, frame, path_save_debug=dbg_path)
        t_track += time.time() - t0

        # ── Draw ──────────────────────────────────────────────────────────────
        t0 = time.time()
        overlay = frame.copy()

        for t in online_targets:
            x1, y1, x2, y2 = t[0], t[1], t[2], t[3]
            tid = int(t[4])

            results_txt.append(
                f"{frame_id},{tid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},-1,-1,-1,-1\n"
            )

            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])

            # Per-ID colour (deterministic)
            color = (int(tid * 37 % 255), int(tid * 97 % 200 + 55), int(tid * 157 % 255))

            cv2.rectangle(overlay, (x1i, y1i), (x2i, y2i), (255,100,0), -1)   # filled (for blend)
            cv2.rectangle(frame,   (x1i, y1i), (x2i, y2i), (0,255,0),  2)   # border

            label = f"ID:{tid}"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
            lx, ly = x1i, max(y1i - 4, th + 4)
            cv2.rectangle(frame, (lx, ly - th - 4), (lx + tw + 4, ly + 2), (144,0,130), -1)
            cv2.putText(frame, label, (lx + 2, ly - 2), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if SHOW_TRAILS:
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                trails[tid].append((cx, cy))
                draw_trail(frame, trails[tid], (255,255,220), trail_thick=2)

        # Blend filled boxes
        cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0, frame)

        # ── HUD overlays ──────────────────────────────────────────────────────
        # FPS
        put_text_shadowed(frame, f"FPS: {fps_disp:.1f}",       (12, 30),  0.8, FPS_COLOR,  2)
        # Tracker name
        put_text_shadowed(frame, f"Tracker: {TRACKER_NAME}",   (12, 60),  0.65, INFO_COLOR, 1)
        # Resolution
        put_text_shadowed(frame, f"Res: {INFER_W}x{INFER_H}",  (12, 88),  0.55, INFO_COLOR, 1)
        # Frame counter
        put_text_shadowed(frame, f"Frame: {frame_id}/{total_frames}", (12, 114), 0.55, (200, 200, 200), 1)
        # Active tracks
        put_text_shadowed(frame, f"Tracks: {len(online_targets)}", (12, 140), 0.55, (200, 200, 200), 1)

        t_draw += time.time() - t0

        # ── Write ─────────────────────────────────────────────────────────────
        t0 = time.time()
        if SAVE_VIDEO and out:
            out.write(frame)
        t_write += time.time() - t0

        if frame_id % 100 == 0:
            print(f"[{frame_id:5d}/{total_frames}]  FPS: {fps_disp:5.1f}  "
                  f"Tracks: {len(online_targets):3d}  γ: {gamma_val:.2f}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if PROCESS_VIDEO:
        cap.release()
    if out:
        out.release()
    # cv2.destroyAllWindows() is intentionally removed — crashes in headless Docker

    # ── Save MOT results ──────────────────────────────────────────────────────
    results_path = os.path.join(OUTPUT_DIR, f"tracking_pinesort_{filename}.txt")
    with open(results_path, "w") as f:
        f.writelines(results_txt)

    total_time = time.time() - loop_start
    avg_fps    = frame_id / (total_time + 1e-9)

    print(f"\n{'='*52}")
    print(f"  TRACKING COMPLETE")
    print(f"{'='*52}")
    print(f"  Frames processed : {frame_id}")
    print(f"  Average FPS      : {avg_fps:.1f}")
    print(f"  Total time       : {total_time:.1f}s")
    print(f"  Output video     : {out_path if SAVE_VIDEO else 'disabled'}")
    print(f"  MOT results      : {results_path}")
    print(f"{'='*52}")
    print(f"  Timing breakdown (avg ms/frame):")
    n = max(frame_id, 1)
    print(f"    Read    : {t_read/n*1000:6.2f} ms")
    print(f"    Preproc : {t_pre/n*1000:6.2f} ms")
    print(f"    Infer   : {t_inf/n*1000:6.2f} ms  ← biggest cost")
    print(f"    Track   : {t_track/n*1000:6.2f} ms")
    print(f"    Draw    : {t_draw/n*1000:6.2f} ms")
    print(f"    Write   : {t_write/n*1000:6.2f} ms")
    print(f"{'='*52}")
