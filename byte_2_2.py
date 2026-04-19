#!/usr/bin/env python3
# ByteTrack version — styled and timed to match 2ptrack.py
# Timing fix: infer+track cost correctly measured via generator gap timing

import os
import cv2
import yaml
import time
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

# ================= PATHS =================
MODEL_PATH = "/workspace/PR_25/vayudh/visdrone_finetune22/weights/best_container.engine"
VIDEO_PATH = "/workspace/PR_25/vayudh/video2.mp4"
OUTPUT_DIR = "/ultralytics/runs/tracking_output/"
filename   = "uav_traffic2_i"

# ── Resolution ───────────────────────────────────────────────────────────────
INFER_W, INFER_H = 640, 480

# ── ByteTrack config ─────────────────────────────────────────────────────────
tracker_config = {
    'tracker_type':      'bytetrack',
    'track_high_thresh': 0.5,
    'track_low_thresh':  0.1,
    'new_track_thresh':  0.6,
    'track_buffer':      60,
    'match_thresh':      0.8,
    'fuse_score':        True,
}
TRACKER_NAME = tracker_config['tracker_type'].upper()
TRACKER_YAML = '/workspace/tracker_config.yaml'

# ── Output options ────────────────────────────────────────────────────────────
SAVE_VIDEO  = True
SHOW_TRAILS = True
TRAIL_LEN   = 30

# ── Colors (BGR) ──────────────────────────────────────────────────────────────
FPS_COLOR  = (0, 255, 0)
INFO_COLOR = (0, 200, 255)
FONT       = cv2.FONT_HERSHEY_SIMPLEX
ALPHA      = 0.1

# ── FPS smoothing ─────────────────────────────────────────────────────────────
FPS_SMOOTH    = 0.9
RESIZE_INTERP = cv2.INTER_LINEAR


# ─────────────────────────── helpers ─────────────────────────────────────────

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
    x, y = pos
    cv2.putText(frame, text, (x+1, y+1), FONT, scale, (0, 0, 0), thickness+2, cv2.LINE_AA)
    cv2.putText(frame, text, (x,   y  ), FONT, scale, color,     thickness,   cv2.LINE_AA)


# ─────────────────────────── main ────────────────────────────────────────────

if __name__ == "__main__":
    create_output_directory(OUTPUT_DIR)

    with open(TRACKER_YAML, 'w') as f:
        yaml.dump(tracker_config, f)

    # ── Load & warm up model ──────────────────────────────────────────────────
    print("Loading YOLO model ...")
    model = YOLO(MODEL_PATH)
    dummy = np.zeros((INFER_H, INFER_W, 3), dtype=np.uint8)
    for _ in range(2):
        model.predict(dummy, imgsz=(INFER_W), verbose=False)
    print("Model warmed up.")

    # ── Video source info ─────────────────────────────────────────────────────
    _cap = cv2.VideoCapture(VIDEO_PATH)
    fps_src      = _cap.get(cv2.CAP_PROP_FPS) or 30
    src_w        = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _cap.release()
    print(f"Source video: {src_w}x{src_h} @ {fps_src:.1f} fps  ({total_frames} frames)")

    # ── Video writer ──────────────────────────────────────────────────────────
    out = None
    out_path = ""
    if SAVE_VIDEO:
        out_path = os.path.join(OUTPUT_DIR, f"byteTrack_{filename}.avi")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"XVID"),
                              fps_src, (INFER_W, INFER_H))
        print(f"Saving to: {out_path}  ({INFER_W}x{INFER_H})")

    # ── State ─────────────────────────────────────────────────────────────────
    trails      = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
    results_txt = []
    frame_id    = 0
    fps_disp    = 0.0
    prev_time   = time.time()

    # Timing buckets
    # t_infer_track: wall-clock gap between loop iterations = true infer+track cost
    # This works because model.track(stream=True) is a generator — it does ALL
    # the GPU inference and ByteTrack assignment BEFORE yielding back to our code.
    # So: iter_end - iter_start = time the generator was in control = infer+track.
    t_infer_track = 0.0
    t_read  = 0.0
    t_pre   = 0.0
    t_parse = 0.0
    t_draw  = 0.0
    t_write = 0.0

    print("Tracking started ...")
    loop_start = time.time()
    iter_start = time.time()   # generator is about to run the first frame

    # ── ByteTrack stream loop ─────────────────────────────────────────────────
    for result in model.track(
        source=VIDEO_PATH,
        tracker=TRACKER_YAML,
        conf=0.4,
        iou=0.4,
        stream=True,
        persist=True,
        show=False,
        verbose=False,
    ):
        # Generator just yielded — measure how long it was in control
        # (that time = TRT inference + ByteTrack data association)
        iter_end = time.time()
        t_infer_track += iter_end - iter_start

        # ── Read ──────────────────────────────────────────────────────────────
        t0 = time.time()
        frame = result.orig_img.copy()
        t_read += time.time() - t0

        frame_id += 1

        # ── Pre-process ───────────────────────────────────────────────────────
        t0 = time.time()
        frame = cv2.resize(frame, (INFER_W, INFER_H), interpolation=RESIZE_INTERP)
        t_pre += time.time() - t0

        # ── FPS ───────────────────────────────────────────────────────────────
        now = time.time()
        if frame_id == 1:
            fps_disp = 1.0
        else:
            instant  = 1.0 / (now - prev_time + 1e-9)
            fps_disp = FPS_SMOOTH * fps_disp + (1 - FPS_SMOOTH) * instant
        prev_time = now

        # ── Parse detections (GPU->CPU tensor copy) ────────────────────────────
        t0 = time.time()
        online_targets = []
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            ids   = result.boxes.id.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            for box, tid, conf in zip(boxes, ids, confs):
                online_targets.append((*box, tid, conf))
        t_parse += time.time() - t0

        # ── Draw ──────────────────────────────────────────────────────────────
        t0 = time.time()
        overlay = frame.copy()

        for entry in online_targets:
            x1, y1, x2, y2, tid, conf = entry

            results_txt.append(
                f"{frame_id},{tid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},-1,-1,-1,-1\n"
            )

            color = (int(tid * 37 % 255), int(tid * 97 % 200 + 55), int(tid * 157 % 255))

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 100, 0), -1)
            cv2.rectangle(frame,   (x1, y1), (x2, y2), (0, 255, 0),   2)

            label = f"ID:{tid}"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
            lx, ly = x1, max(y1 - 4, th + 4)
            cv2.rectangle(frame, (lx, ly - th - 4), (lx + tw + 4, ly + 2), (144, 0, 130), -1)
            cv2.putText(frame, label, (lx + 2, ly - 2), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if SHOW_TRAILS:
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                trails[tid].append((cx, cy))
                draw_trail(frame, trails[tid], (255, 255, 220), trail_thick=2)

        cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0, frame)

        # ── HUD ───────────────────────────────────────────────────────────────
        put_text_shadowed(frame, f"FPS: {fps_disp:.1f}",             (12,  30), 0.80, FPS_COLOR,       2)
        put_text_shadowed(frame, f"Tracker: {TRACKER_NAME}",         (12,  60), 0.65, INFO_COLOR,      1)
        put_text_shadowed(frame, f"Res: {INFER_W}x{INFER_H}",        (12,  88), 0.55, INFO_COLOR,      1)
        put_text_shadowed(frame, f"Frame: {frame_id}/{total_frames}", (12, 114), 0.55, (200, 200, 200), 1)
        put_text_shadowed(frame, f"Tracks: {len(online_targets)}",    (12, 140), 0.55, (200, 200, 200), 1)

        t_draw += time.time() - t0

        # ── Write ─────────────────────────────────────────────────────────────
        t0 = time.time()
        if SAVE_VIDEO and out:
            out.write(frame)
        t_write += time.time() - t0

        if frame_id % 100 == 0:
            print(f"[{frame_id:5d}/{total_frames}]  FPS: {fps_disp:5.1f}  "
                  f"Tracks: {len(online_targets):3d}")

        # Reset: mark when we hand control back to the generator
        iter_start = time.time()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if out:
        out.release()

    # ── Save MOT results ──────────────────────────────────────────────────────
    results_path = os.path.join(OUTPUT_DIR, f"tracking_bytetrack_{filename}.txt")
    with open(results_path, "w") as f:
        f.writelines(results_txt)

    total_time = time.time() - loop_start
    avg_fps    = frame_id / (total_time + 1e-9)
    n          = max(frame_id, 1)

    print(f"\n{'='*52}")
    print(f"  TRACKING COMPLETE")
    print(f"{'='*52}")
    print(f"  Frames processed    : {frame_id}")
    print(f"  Average FPS         : {avg_fps:.1f}")
    print(f"  Total time          : {total_time:.1f}s")
    print(f"  Output video        : {out_path if SAVE_VIDEO else 'disabled'}")
    print(f"  MOT results         : {results_path}")
    print(f"{'='*52}")
    print(f"  Timing breakdown (avg ms/frame):")
    print(f"    Infer+Track : {t_infer_track/n*1000:6.2f} ms  <- TRT infer + ByteTrack (generator gap)")
    print(f"    Read        : {t_read       /n*1000:6.2f} ms  (orig_img copy)")
    print(f"    Preproc     : {t_pre        /n*1000:6.2f} ms  (resize)")
    print(f"    Parse       : {t_parse      /n*1000:6.2f} ms  (GPU->CPU tensor transfer)")
    print(f"    Draw        : {t_draw       /n*1000:6.2f} ms")
    print(f"    Write       : {t_write      /n*1000:6.2f} ms")
    print(f"    {'─'*30}")
    accounted = (t_infer_track + t_read + t_pre + t_parse + t_draw + t_write) / n * 1000
    print(f"    Accounted   : {accounted:6.2f} ms")
    print(f"    Wall total  : {total_time/n*1000:6.2f} ms/frame")
    print(f"{'='*52}")
