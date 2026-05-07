import cv2
import numpy as np
import json

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
OUT_W, OUT_H = 1200, 800   # warped canvas size (matches 6:4 ratio)
ROWS,  COLS  = 4, 6
TARGET_MEAN  = 128.0       # gamma correction target brightness
CLIP_GAMMA   = (0.5, 2.0)  # min/max gamma to avoid over-correction


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD IMAGE
# ─────────────────────────────────────────────────────────────────────────────
def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — GAMMA CORRECTION
# ─────────────────────────────────────────────────────────────────────────────
class GammaCorrector:
    def __init__(self, target_mean=TARGET_MEAN, clip_gamma=CLIP_GAMMA):
        self.target_mean = target_mean
        self.clip_gamma  = clip_gamma

    def estimate_gamma(self, frame: np.ndarray) -> float:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean  = np.mean(gray)
        gamma = mean / self.target_mean
        gamma = np.clip(gamma, self.clip_gamma[0], self.clip_gamma[1])
        return float(gamma)

    def build_lut(self, gamma: float) -> np.ndarray:
        return np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)],
            dtype=np.uint8
        )

    def process(self, frame: np.ndarray) -> tuple:
        """
        Returns:
            corrected (np.ndarray) : gamma-corrected BGR frame
            gamma     (float)      : gamma value that was applied
        """
        gamma     = self.estimate_gamma(frame)
        lut       = self.build_lut(gamma)
        corrected = cv2.LUT(frame, lut)
        return corrected, gamma


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — MANUAL 4-POINT SELECTION (click on the 4 chart corners)
# ─────────────────────────────────────────────────────────────────────────────
class PointSelector:
    """
    Opens a window and lets the user click exactly 4 points.
    Click order: Top-Left → Top-Right → Bottom-Right → Bottom-Left
    Press 'r' to reset, 'q' / Enter to confirm.
    """
    def __init__(self):
        self.points = []

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))

    def select(self, img: np.ndarray) -> np.ndarray:
        self.points = []
        win = "Select 4 corners: TL → TR → BR → BL  |  r=reset  Enter=confirm"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, self._mouse_cb)

        while True:
            vis = img.copy()
            for i, pt in enumerate(self.points):
                cv2.circle(vis, pt, 8, (0, 255, 0), -1)
                cv2.putText(vis, ["TL","TR","BR","BL"][i], (pt[0]+10, pt[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            if len(self.points) == 4:
                pts = np.array(self.points, dtype=np.int32)
                cv2.polylines(vis, [pts.reshape(-1,1,2)], True, (0,255,255), 2)

            cv2.imshow(win, vis)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('r'):
                self.points = []
            if (key == 13 or key == ord('q')) and len(self.points) == 4:
                break

        cv2.destroyAllWindows()
        return np.array(self.points, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — PERSPECTIVE TRANSFORM  (manual pts → flat rectangle)
# ─────────────────────────────────────────────────────────────────────────────
def order_points(pts: np.ndarray) -> np.ndarray:
    """Sort 4 pts into TL, TR, BR, BL regardless of click order."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]    # TL
    rect[2] = pts[np.argmax(s)]    # BR
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmax(diff)] # BL
    return rect


def perspective_transform(img: np.ndarray,
                           src_pts: np.ndarray,
                           out_w=OUT_W, out_h=OUT_H) -> tuple:
    """
    Warp the user-selected quad to a flat out_w × out_h canvas.
    Returns:
        warped  : rectified image
        M       : 3×3 perspective matrix
        M_inv   : inverse matrix (warped → original)
        ordered : src_pts sorted as TL,TR,BR,BL
    """
    ordered = order_points(src_pts)
    dst     = np.float32([[0,0],[out_w,0],[out_w,out_h],[0,out_h]])
    M       = cv2.getPerspectiveTransform(ordered, dst)
    M_inv   = np.linalg.inv(M)
    warped  = cv2.warpPerspective(img, M, (out_w, out_h))
    return warped, M, M_inv, ordered


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — DETECT PATCH CORNERS  (runs on the clean warped ROI)
# ─────────────────────────────────────────────────────────────────────────────
def detect_patch_corners_in_roi(warped: np.ndarray,
                                 rows=ROWS, cols=COLS,
                                 border_frac=0.03) -> dict:
    """
    Divide the warped (flat) image into a rows×cols grid.
    border_frac : fractional inset to skip the chart's black border.
    Returns dict: patch_index → {TL, TR, BR, BL} in warped coords.
    """
    h, w  = warped.shape[:2]
    pad_x = w * border_frac
    pad_y = h * border_frac
    cell_w = (w - 2*pad_x) / cols
    cell_h = (h - 2*pad_y) / rows

    patches = {}
    idx = 0
    for row in range(rows):
        for col in range(cols):
            x0 = pad_x + col * cell_w
            y0 = pad_y + row * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            patches[idx] = {
                "TL": [x0, y0], "TR": [x1, y0],
                "BR": [x1, y1], "BL": [x0, y1]
            }
            idx += 1
    return patches


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — MAP CORNERS BACK TO ORIGINAL IMAGE
# ─────────────────────────────────────────────────────────────────────────────
def map_to_original(patches_warped: dict, M_inv: np.ndarray) -> dict:
    """Project warped patch corners back into original image space."""
    patches_orig = {}
    for idx, c in patches_warped.items():
        pts_w = np.float32([c["TL"], c["TR"], c["BR"], c["BL"]]).reshape(-1,1,2)
        pts_o = cv2.perspectiveTransform(pts_w, M_inv).reshape(-1,2)
        patches_orig[idx] = {
            "TL": pts_o[0].tolist(), "TR": pts_o[1].tolist(),
            "BR": pts_o[2].tolist(), "BL": pts_o[3].tolist()
        }
    return patches_orig


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def draw_patches(vis: np.ndarray, patches: dict,
                 line_color=(0,255,0), dot_color=(0,0,255),
                 thickness=2) -> np.ndarray:
    for idx, c in patches.items():
        pts = np.array([c["TL"],c["TR"],c["BR"],c["BL"]], dtype=np.int32)
        cv2.polylines(vis, [pts.reshape(-1,1,2)], True, line_color, thickness)
        for pt in pts:
            cv2.circle(vis, tuple(pt.tolist()), 5, dot_color, -1)
        cx = int((pts[0][0] + pts[2][0]) / 2)
        cy = int((pts[0][1] + pts[2][1]) / 2)
        cv2.putText(vis, str(idx), (cx-12, cy+6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    return vis


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(image_path: str,
                 out_original: str = "out_original.jpg",
                 out_warped:   str = "out_warped.jpg",
                 out_json:     str = "corners.json",
                 interactive:  bool = True,
                 manual_pts:   np.ndarray = None):
    """
    Parameters
    ----------
    image_path  : path to input image / webcam frame
    interactive : if True, opens GUI for point selection
                  if False, pass manual_pts (4×2 float32 array)
    manual_pts  : used when interactive=False
    """

    # 1. Load
    img = load_image(image_path)
    print(f"[1] Loaded image  {img.shape[1]}×{img.shape[0]}")

    # 2. Gamma correction
    gc             = GammaCorrector()
    img_gc, gamma  = gc.process(img)
    print(f"[2] Gamma correction applied  γ={gamma:.3f}")

    # 3. Select 4 points
    if interactive:
        selector = PointSelector()
        src_pts  = selector.select(img_gc)
    else:
        assert manual_pts is not None, "Provide manual_pts when interactive=False"
        src_pts = np.array(manual_pts, dtype=np.float32)
    print(f"[3] 4 points selected: {src_pts.tolist()}")

    # 4. Perspective transform (on gamma-corrected image)
    warped, M, M_inv, ordered = perspective_transform(img_gc, src_pts)
    print(f"[4] Perspective transform done  →  {OUT_W}×{OUT_H} canvas")

    # 5. Detect patch corners inside the warped ROI
    patches_warped = detect_patch_corners_in_roi(warped)
    print(f"[5] Detected {len(patches_warped)} patch corners in warped ROI")

    # 6. Map back to original coords
    patches_orig = map_to_original(patches_warped, M_inv)
    print(f"[6] Mapped corners back to original image space")

    # Visualise & save
    vis_orig   = draw_patches(img_gc.copy(),  patches_orig)
    vis_warped = draw_patches(warped.copy(),  patches_warped)

    # Draw user-selected quad on original
    cv2.polylines(vis_orig,
                  [ordered.astype(np.int32).reshape(-1,1,2)],
                  True, (0,255,255), 3)

    cv2.imwrite(out_original, vis_orig)
    cv2.imwrite(out_warped,   vis_warped)

    result = {
        "gamma_applied"           : gamma,
        "user_selected_pts"       : ordered.tolist(),
        "patches_original_coords" : patches_orig,
        "patches_warped_coords"   : patches_warped
    }
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✅  Saved:\n   {out_original}\n   {out_warped}\n   {out_json}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── For testing without a display, pass manual_pts ───────────────────────
    # These are the approximate corners of the chart in the sample image
    test_pts = np.float32([
        [114, 24],   # TL
        [481, 21],   # TR
        [483, 277],  # BR
        [116, 279],  # BL
    ])

    run_pipeline(
        image_path   = "/mnt/user-data/uploads/1000278667.jpg",
        out_original = "/mnt/user-data/outputs/out_original.jpg",
        out_warped   = "/mnt/user-data/outputs/out_warped.jpg",
        out_json     = "/mnt/user-data/outputs/corners.json",
        interactive  = False,   # ← set True for GUI point picker
        manual_pts   = test_pts
    )
