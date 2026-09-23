"""
KARIS inference with postprocessing.

Pipeline per frame:
  YOLO seg + ByteTrack -> ROI filter -> mask cleanup -> temporal class voting
  -> 2D pose (minAreaRect) -> optional depth: 3D position + size sanity check

Usage:
  python infer.py --weights runs/segment/karis_yolov8s/weights/best.pt --source 0
  python infer.py --weights best.pt --source video.mp4 --conf 0.4
"""
import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
IMGSZ = 1280
CONF = 0.8               # set from the F1 curve printed by train.py
IOU = 0.6
DEVICE = 0

# Region of interest in pixels (x1, y1, x2, y2). None = whole frame.
# Detections whose mask centroid falls outside are discarded.
ROI = None

MIN_MASK_AREA_PX = 300   # drop tiny blobs
VOTE_WINDOW = 8          # frames used for majority vote
MIN_STABLE_FRAMES = 4    # frames a track must be seen before it is reported

# Optional: known object sizes in meters (long side, short side) per class name.
# Detections whose measured size deviates more than SIZE_TOL are rejected.
# Only used when a depth image is supplied.
KNOWN_SIZE_M: dict = {
    # "bolt_m8": (0.04, 0.015),
}
SIZE_TOL = 0.35
DEPTH_ERODE_PX = 3       # erode mask before sampling depth (removes flying pixels)


@dataclass
class Detection:
    track_id: int
    cls_id: int
    cls_name: str
    conf: float
    box: tuple                    # x1, y1, x2, y2
    mask: np.ndarray              # uint8 HxW, full resolution, 0/255
    center_px: tuple              # (u, v)
    size_px: tuple                # (long, short) of the min area rectangle
    angle_deg: float              # orientation of the long axis, [0, 180)
    xyz: Optional[tuple] = None   # meters, camera frame (if depth given)


# ----------------------------------------------------------------------------
# Mask utilities
# ----------------------------------------------------------------------------
def clean_mask(mask: np.ndarray, k: int = 3) -> np.ndarray:
    """Opening (remove specks), closing (fill gaps), keep largest component,
    fill interior holes."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (labels == largest).astype(np.uint8) * 255

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def min_area_pose(mask: np.ndarray):
    """Center, (long, short) side and long-axis angle from the mask contour."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    (cx, cy), (w, h), ang = cv2.minAreaRect(cnt)
    if w < h:                      # make w the long side
        w, h = h, w
        ang += 90.0
    return (cx, cy), (w, h), ang % 180.0


def depth_position(mask, depth_m, intr):
    """Median depth inside the eroded mask -> 3D point of the mask centroid.
    intr = (fx, fy, cx, cy). depth_m is a float array in meters aligned to color."""
    fx, fy, cx0, cy0 = intr
    if DEPTH_ERODE_PX > 0:
        ker = np.ones((2 * DEPTH_ERODE_PX + 1,) * 2, np.uint8)
        eroded = cv2.erode(mask, ker)
        if eroded.sum() > 0:
            mask = eroded
    z_vals = depth_m[mask > 0]
    z_vals = z_vals[(z_vals > 0.1) & np.isfinite(z_vals)]
    if z_vals.size < 20:
        return None
    z = float(np.median(z_vals))
    ys, xs = np.nonzero(mask)
    u, v = xs.mean(), ys.mean()
    return ((u - cx0) * z / fx, (v - cy0) * z / fy, z)


# ----------------------------------------------------------------------------
# Main class
# ----------------------------------------------------------------------------
class KarisSegmenter:
    def __init__(self, weights: str, conf=CONF, iou=IOU, imgsz=IMGSZ, device=DEVICE):
        self.model = YOLO(weights)
        self.names = self.model.names
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device
        self.history = defaultdict(lambda: deque(maxlen=VOTE_WINDOW))  # id -> (cls, conf)

    def reset(self):
        self.history.clear()
        self.model.predictor = None  # resets the tracker state

    def process(self, frame, depth_m=None, intrinsics=None, track=True):
        """track=True for video/live streams (tracking + temporal voting).
        track=False for independent still images (plain predict, no voting)."""
        common = dict(
            conf=self.conf, iou=self.iou, imgsz=self.imgsz, device=self.device,
            agnostic_nms=True,      # one object cannot be two classes
            retina_masks=True,      # full-resolution masks
            verbose=False,
        )
        if track:
            r = self.model.track(frame, persist=True, tracker="bytetrack.yaml", **common)[0]
        else:
            r = self.model.predict(frame, **common)[0]

        if r.masks is None or r.boxes is None or len(r.boxes) == 0:
            return []

        h, w = frame.shape[:2]
        ids = r.boxes.id
        ids = ids.int().cpu().numpy() if ids is not None else -np.arange(1, len(r.boxes) + 1)
        cls = r.boxes.cls.int().cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()
        masks = r.masks.data.cpu().numpy()

        out = []
        for i in range(len(cls)):
            m = (masks[i] > 0.5).astype(np.uint8) * 255
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

            m = clean_mask(m)
            if int(np.count_nonzero(m)) < MIN_MASK_AREA_PX:
                continue

            pose = min_area_pose(m)
            if pose is None:
                continue
            (cu, cv_), (long_px, short_px), ang = pose

            if ROI is not None:
                x1, y1, x2, y2 = ROI
                if not (x1 <= cu <= x2 and y1 <= cv_ <= y2):
                    continue

            # temporal voting: weighted majority over the last frames
            tid = int(ids[i])
            if tid >= 0:
                self.history[tid].append((int(cls[i]), float(conf[i])))
                hist = self.history[tid]
                if len(hist) < MIN_STABLE_FRAMES:
                    continue
                votes = Counter()
                for c, s in hist:
                    votes[c] += s
                voted_cls = votes.most_common(1)[0][0]
                voted_conf = float(np.mean([s for c, s in hist if c == voted_cls]))
            else:
                voted_cls, voted_conf = int(cls[i]), float(conf[i])

            name = self.names[voted_cls]

            xyz = None
            if depth_m is not None and intrinsics is not None:
                xyz = depth_position(m, depth_m, intrinsics)
                if xyz is None:
                    continue
                if name in KNOWN_SIZE_M:
                    fx = intrinsics[0]
                    long_m = long_px * xyz[2] / fx
                    short_m = short_px * xyz[2] / fx
                    exp_long, exp_short = KNOWN_SIZE_M[name]
                    if (abs(long_m - exp_long) / exp_long > SIZE_TOL
                            or abs(short_m - exp_short) / exp_short > SIZE_TOL):
                        continue

            out.append(Detection(
                track_id=tid, cls_id=voted_cls, cls_name=name, conf=voted_conf,
                box=tuple(boxes[i]), mask=m, center_px=(cu, cv_),
                size_px=(long_px, short_px), angle_deg=ang, xyz=xyz,
            ))
        return out


def draw(frame, dets):
    vis = frame.copy()
    for d in dets:
        color = tuple(int(x) for x in np.random.RandomState(d.cls_id).randint(60, 255, 3))
        overlay = vis.copy()
        overlay[d.mask > 0] = color
        vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
        cnts, _ = cv2.findContours(d.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, color, 2)

        # long axis arrow
        u, v = d.center_px
        a = np.deg2rad(d.angle_deg)
        p2 = (int(u + 0.5 * d.size_px[0] * np.cos(a)), int(v + 0.5 * d.size_px[0] * np.sin(a)))
        cv2.arrowedLine(vis, (int(u), int(v)), p2, (0, 255, 255), 2)

        label = f"#{d.track_id} {d.cls_name} {d.conf:.2f} {d.angle_deg:.0f}deg"
        if d.xyz:
            label += f" z={d.xyz[2]:.2f}m"
        cv2.putText(vis, label, (int(d.box[0]), max(15, int(d.box[1]) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if ROI is not None:
        cv2.rectangle(vis, (ROI[0], ROI[1]), (ROI[2], ROI[3]), (255, 255, 255), 1)
    return vis


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", default="0", help="camera index or video/image path")
    ap.add_argument("--conf", type=float, default=CONF)
    args = ap.parse_args()

    seg = KarisSegmenter(args.weights, conf=args.conf)
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Without a depth camera we only get 2D results. With one, pass
        # depth_m=<HxW float meters aligned to color>, intrinsics=(fx, fy, cx, cy)
        dets = seg.process(frame)
        cv2.imshow("KARIS", draw(frame, dets))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
