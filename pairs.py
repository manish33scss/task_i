"""
ColorChecker Pair Pipeline
──────────────────────────
Directory layout expected:
    images/
        good_<name>.jpg
        tint_<name>.jpg

For each pair:
  1. Load both images
  2. User selects 4 corners on EACH image (GUI)
  3. Perspective transform
  4. Gamma correction
  5. Detect 24 patch corners + centers in warped ROI
  6. Map back to original coords
  7. Build hstack output:
       [good_input | good_warped | good_corrected | good_overlay]
       [tint_input | tint_warped | tint_corrected | tint_overlay]
       ── stacked vertically, one row per image in the pair
"""

import cv2
import numpy as np
import json
import os
import glob

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
OUT_W       = 900
OUT_H       = 600
ROWS, COLS  = 4, 6
TARGET_MEAN = 128.0
CLIP_GAMMA  = (0.5, 2.0)
BORDER_FRAC = 0.03          # inset from chart edge when gridding
DISPLAY_H   = 400           # row height in the final hstack canvas


# ─────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────
def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    return img


def find_pairs(directory: str):
    """
    Return list of (good_path, tint_path) tuples matched by suffix.
    e.g. good_chart01.jpg  ↔  tint_chart01.jpg
    """
    good_files = sorted(glob.glob(os.path.join(directory, "good_*")))
    pairs = []
    for gp in good_files:
        suffix = os.path.basename(gp)[len("good_"):]
        tp = os.path.join(directory, "tint_" + suffix)
        if os.path.exists(tp):
            pairs.append((gp, tp))
        else:
            print(f"[WARN] No tint_ match for {gp}, skipping.")
    return pairs


# ─────────────────────────────────────────────────────────────────
# 2. MANUAL 4-POINT SELECTION
# ─────────────────────────────────────────────────────────────────
class PointSelector:
    """
    Click TL → TR → BR → BL on the displayed image.
    'r' = reset clicks   |   Enter / 'q' = confirm (needs 4 pts)
    """
    LABELS = ["TL", "TR", "BR", "BL"]

    def __init__(self):
        self.points = []

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))

    def select(self, img: np.ndarray, title: str = "") -> np.ndarray:
        self.points = []
        win = f"4-corner select  [{title}]  TL→TR→BR→BL | r=reset | Enter=confirm"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 900, 600)
        cv2.setMouseCallback(win, self._mouse_cb)

        while True:
            vis = img.copy()
            for i, pt in enumerate(self.points):
                cv2.circle(vis, pt, 8, (0, 255, 0), -1)
                cv2.putText(vis, self.LABELS[i], (pt[0]+10, pt[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if len(self.points) == 4:
                poly = np.array(self.points, dtype=np.int32)
                cv2.polylines(vis, [poly.reshape(-1, 1, 2)], True, (0, 255, 255), 2)

            cv2.imshow(win, vis)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('r'):
                self.points = []
            if key in (13, ord('q')) and len(self.points) == 4:
                break

        cv2.destroyWindow(win)
        return np.array(self.points, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
# 3. PERSPECTIVE TRANSFORM
# ─────────────────────────────────────────────────────────────────
def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]     # TL
    rect[2] = pts[np.argmax(s)]     # BR
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect


def perspective_transform(img: np.ndarray,
                           src_pts: np.ndarray,
                           out_w=OUT_W, out_h=OUT_H):
    ordered = order_points(src_pts)
    dst     = np.float32([[0,0],[out_w,0],[out_w,out_h],[0,out_h]])
    M       = cv2.getPerspectiveTransform(ordered, dst)
    M_inv   = np.linalg.inv(M)
    warped  = cv2.warpPerspective(img, M, (out_w, out_h))
    return warped, M, M_inv, ordered


# ─────────────────────────────────────────────────────────────────
# 4. GAMMA CORRECTION  (applied on warped image)
# ─────────────────────────────────────────────────────────────────
class GammaCorrector:
    def __init__(self, target_mean=TARGET_MEAN, clip_gamma=CLIP_GAMMA):
        self.target_mean = target_mean
        self.clip_gamma  = clip_gamma

    def estimate_gamma(self, frame: np.ndarray) -> float:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean  = np.mean(gray)
        gamma = mean / self.target_mean
        return float(np.clip(gamma, *self.clip_gamma))

    def build_lut(self, gamma: float) -> np.ndarray:
        return np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)],
            dtype=np.uint8
        )

    def process(self, frame: np.ndarray):
        gamma     = self.estimate_gamma(frame)
        lut       = self.build_lut(gamma)
        corrected = cv2.LUT(frame, lut)
        return corrected, gamma


# ─────────────────────────────────────────────────────────────────
# 5. DETECT PATCH CORNERS + CENTERS  (inside warped ROI)
# ─────────────────────────────────────────────────────────────────
def detect_patches(warped: np.ndarray,
                   rows=ROWS, cols=COLS,
                   border_frac=BORDER_FRAC):
    """
    Returns dict  idx → {TL, TR, BR, BL, center}
    All coords are in warped-image space.
    """
    h, w   = warped.shape[:2]
    pad_x  = w * border_frac
    pad_y  = h * border_frac
    cell_w = (w - 2*pad_x) / cols
    cell_h = (h - 2*pad_y) / rows

    patches = {}
    for idx in range(rows * cols):
        row = idx // cols
        col = idx  % cols
        x0  = pad_x + col * cell_w
        y0  = pad_y + row * cell_h
        x1  = x0 + cell_w
        y1  = y0 + cell_h
        cx  = (x0 + x1) / 2
        cy  = (y0 + y1) / 2
        patches[idx] = {
            "TL": [x0, y0], "TR": [x1, y0],
            "BR": [x1, y1], "BL": [x0, y1],
            "center": [cx, cy]
        }
    return patches


# ─────────────────────────────────────────────────────────────────
# 6. MAP BACK TO ORIGINAL COORDS
# ─────────────────────────────────────────────────────────────────
def map_to_original(patches_warped: dict, M_inv: np.ndarray) -> dict:
    patches_orig = {}
    for idx, c in patches_warped.items():
        pts_w = np.float32(
            [c["TL"], c["TR"], c["BR"], c["BL"], c["center"]]
        ).reshape(-1, 1, 2)
        pts_o = cv2.perspectiveTransform(pts_w, M_inv).reshape(-1, 2)
        patches_orig[idx] = {
            "TL":     pts_o[0].tolist(),
            "TR":     pts_o[1].tolist(),
            "BR":     pts_o[2].tolist(),
            "BL":     pts_o[3].tolist(),
            "center": pts_o[4].tolist()
        }
    return patches_orig


# ─────────────────────────────────────────────────────────────────
# 7. DRAW OVERLAY  (patches + centers)
# ─────────────────────────────────────────────────────────────────
def draw_overlay(img: np.ndarray, patches: dict,
                 line_color=(0,255,0),
                 dot_color=(0,0,255),
                 center_color=(255,0,255),
                 thickness=2) -> np.ndarray:
    vis = img.copy()
    for idx, c in patches.items():
        # Patch rectangle
        pts = np.array([c["TL"],c["TR"],c["BR"],c["BL"]], dtype=np.int32)
        cv2.polylines(vis, [pts.reshape(-1,1,2)], True, line_color, thickness)
        # Corner dots
        for pt in pts:
            cv2.circle(vis, tuple(pt.tolist()), 4, dot_color, -1)
        # Center cross + filled dot
        cx, cy = int(c["center"][0]), int(c["center"][1])
        cv2.drawMarker(vis, (cx,cy), center_color,
                       cv2.MARKER_CROSS, 14, 2)
        cv2.circle(vis, (cx,cy), 4, center_color, -1)
        # Patch index label
        cv2.putText(vis, str(idx), (cx-10, cy-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)
    return vis


# ─────────────────────────────────────────────────────────────────
# BUILD ONE ROW OF THE OUTPUT  (for a single image)
# ─────────────────────────────────────────────────────────────────
def make_row(label: str,
             original: np.ndarray,
             warped:   np.ndarray,
             corrected:np.ndarray,
             overlay_warped: np.ndarray,
             row_h: int = DISPLAY_H) -> np.ndarray:
    """
    Resize each stage to row_h and hstack:
    [original | warped | corrected | overlay]
    Add a label banner at the top of the row.
    """
    def fit(img):
        h, w = img.shape[:2]
        new_w = int(w * row_h / h)
        return cv2.resize(img, (new_w, row_h))

    cols = [fit(original), fit(warped), fit(corrected), fit(overlay_warped)]

    # Add stage title at top of each panel
    titles = ["Input", "Warped", "Gamma corrected", "Overlay + centers"]
    titled = []
    for panel, title in zip(cols, titles):
        banner = np.zeros((28, panel.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner, title, (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
        titled.append(np.vstack([banner, panel]))

    row = np.hstack(titled)

    # Left label strip
    strip = np.zeros((row.shape[0], 60, 3), dtype=np.uint8)
    cv2.putText(strip, label, (4, row.shape[0]//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,220), 2)
    return np.hstack([strip, row])


# ─────────────────────────────────────────────────────────────────
# PROCESS ONE IMAGE  (returns all stages + patch data)
# ─────────────────────────────────────────────────────────────────
def process_image(img: np.ndarray, src_pts: np.ndarray):
    # Perspective transform (on raw input)
    warped, M, M_inv, ordered = perspective_transform(img, src_pts)

    # Gamma correction on warped
    gc = GammaCorrector()
    corrected, gamma = gc.process(warped)

    # Detect patches on gamma-corrected warped image
    patches_warped = detect_patches(corrected)

    # Map back to original coords
    patches_orig = map_to_original(patches_warped, M_inv)

    # Overlay on corrected warped
    overlay = draw_overlay(corrected, patches_warped)

    return {
        "original":       img,
        "warped":         warped,
        "corrected":      corrected,
        "overlay":        overlay,
        "gamma":          gamma,
        "patches_warped": patches_warped,
        "patches_orig":   patches_orig,
        "M":              M,
        "M_inv":          M_inv,
        "ordered_pts":    ordered,
    }


# ─────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────
def run_pair_pipeline(image_dir: str,
                      output_dir: str,
                      interactive: bool = True,
                      test_pts: dict = None):
    """
    Parameters
    ----------
    image_dir   : folder containing good_* and tint_* images
    output_dir  : where results are saved
    interactive : True  → GUI point picker
                  False → use test_pts = {"good": [...], "tint": [...]}
    """
    os.makedirs(output_dir, exist_ok=True)
    pairs = find_pairs(image_dir)

    if not pairs:
        print("[ERROR] No matching good_*/tint_* pairs found.")
        return

    selector = PointSelector()
    all_results = {}

    for good_path, tint_path in pairs:
        pair_name = os.path.splitext(os.path.basename(good_path)[len("good_"):])[0]
        print(f"\n{'='*60}")
        print(f"  Processing pair: {pair_name}")
        print(f"{'='*60}")

        good_img = load_image(good_path)
        tint_img = load_image(tint_path)

        # ── Point selection ───────────────────────────────────────
        if interactive:
            print("  → Select corners on GOOD image")
            good_pts = selector.select(good_img, title=f"GOOD: {pair_name}")
            print("  → Select corners on TINT image")
            tint_pts = selector.select(tint_img, title=f"TINT: {pair_name}")
        else:
            good_pts = np.float32(test_pts["good"])
            tint_pts = np.float32(test_pts["tint"])

        # ── Process each image ────────────────────────────────────
        print("  Processing good image …")
        good_res = process_image(good_img, good_pts)
        print(f"    γ = {good_res['gamma']:.3f}")

        print("  Processing tint image …")
        tint_res = process_image(tint_img, tint_pts)
        print(f"    γ = {tint_res['gamma']:.3f}")

        # ── Build hstack output ───────────────────────────────────
        row_good = make_row("GOOD", good_res["original"], good_res["warped"],
                            good_res["corrected"], good_res["overlay"])
        row_tint = make_row("TINT", tint_res["original"], tint_res["warped"],
                            tint_res["corrected"], tint_res["overlay"])

        # Ensure same width before vstack
        w = min(row_good.shape[1], row_tint.shape[1])
        canvas = np.vstack([row_good[:, :w], row_tint[:, :w]])

        # Separator line
        sep = np.full((4, canvas.shape[1], 3), 80, dtype=np.uint8)
        canvas = np.vstack([canvas[:DISPLAY_H+28+4], sep,
                            canvas[DISPLAY_H+28+4:]])

        out_img  = os.path.join(output_dir, f"{pair_name}_result.jpg")
        out_json = os.path.join(output_dir, f"{pair_name}_data.json")

        cv2.imwrite(out_img, canvas)
        print(f"  Saved: {out_img}")

        # ── Save JSON ─────────────────────────────────────────────
        data = {
            "pair": pair_name,
            "good": {
                "gamma":          good_res["gamma"],
                "selected_pts":   good_res["ordered_pts"].tolist(),
                "patches_warped": good_res["patches_warped"],
                "patches_orig":   good_res["patches_orig"],
                "centers_warped": {k: v["center"]
                                   for k, v in good_res["patches_warped"].items()},
                "centers_orig":   {k: v["center"]
                                   for k, v in good_res["patches_orig"].items()},
            },
            "tint": {
                "gamma":          tint_res["gamma"],
                "selected_pts":   tint_res["ordered_pts"].tolist(),
                "patches_warped": tint_res["patches_warped"],
                "patches_orig":   tint_res["patches_orig"],
                "centers_warped": {k: v["center"]
                                   for k, v in tint_res["patches_warped"].items()},
                "centers_orig":   {k: v["center"]
                                   for k, v in tint_res["patches_orig"].items()},
            }
        }
        with open(out_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved: {out_json}")

        all_results[pair_name] = data

    print(f"\n✅  Done. Processed {len(pairs)} pair(s).")
    return all_results


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Non-interactive test pts (same chart, both images identical crop)
    TEST_PTS = {
        "good": [[114,24],[481,21],[483,277],[116,279]],
        "tint": [[114,24],[481,21],[483,277],[116,279]],
    }

    run_pair_pipeline(
        image_dir   = "/home/claude/test_images",
        output_dir  = "/mnt/user-data/outputs",
        interactive = False,      # ← set True for live webcam / GUI
        test_pts    = TEST_PTS
    )
