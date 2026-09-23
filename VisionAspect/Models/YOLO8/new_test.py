
"""
KARIS test script, self-contained (does not need infer.py).
 
1. Quantitative evaluation on the test split (per-class mask mAP, best F1 threshold)
2. Raw YOLO predictions (baseline)
3. Postprocessed predictions (mask cleanup, min area, 2D pose) + CSV of detections
"""
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_ROOT = Path(
    r"C:\Users\kimsv\OneDrive - Mälardalens universitet\Desktop\KARIS.v1-test1.yolo26"
)
DATA_YAML = DATASET_ROOT / "data.yaml"
TEST_IMAGES = DATASET_ROOT / "test" / "images"

MODEL_PATH = Path(
    r"C:\Users\kimsv\PycharmProjects\KARIS_2\VisionAspect\Models\YOLO8\runs\segment\runs\segment\karis_yolov8s\weights\best.pt"
)
OUTPUT_DIR = Path(
    r"C:\Users\kimsv\OneDrive - Mälardalens universitet\Desktop\temp8"
)

IMAGE_SIZE = 1280
BATCH_SIZE = 4
DEVICE = 0

# None = use the confidence that maximizes mask F1 on the test set; or set a number
CONFIDENCE = None
FALLBACK_CONFIDENCE = 0.25
IOU = 0.6

# postprocessing
MIN_MASK_AREA_PX = 300
ROI = None            # (x1, y1, x2, y2) in pixels, or None for the whole image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ============================================================
# Helpers
# ============================================================
def imread_unicode(path: Path):
    """cv2.imread fails on Windows paths with non-ASCII characters (e.g. 'ä')."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img):
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if ok:
        buf.tofile(str(path))


def best_f1_confidence(metrics):
    try:
        f1 = metrics.seg.f1_curve.mean(0)
        i = int(f1.argmax())
        return float(metrics.seg.px[i]), float(f1[i])
    except Exception:
        return None, None


def clean_mask(mask: np.ndarray, k: int = 3) -> np.ndarray:
    """Opening, closing, keep the largest component, fill holes."""
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
    """Center, (long, short) side, long-axis angle in [0, 180)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    (cx, cy), (w, h), ang = cv2.minAreaRect(cnt)
    if w < h:
        w, h = h, w
        ang += 90.0
    return (cx, cy), (w, h), ang % 180.0


def postprocess(result, names, frame_shape):
    """Turn one ultralytics result into a list of cleaned detection dicts."""
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return []
    h, w = frame_shape[:2]
    cls = result.boxes.cls.int().cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    masks = result.masks.data.cpu().numpy()

    dets = []
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
        dets.append({
            "cls_id": int(cls[i]), "cls_name": names[int(cls[i])], "conf": float(conf[i]),
            "box": boxes[i], "mask": m, "center": (cu, cv_),
            "size": (long_px, short_px), "angle": ang,
        })
    return dets


def draw(frame, dets):
    vis = frame.copy()
    for d in dets:
        color = tuple(int(x) for x in np.random.RandomState(d["cls_id"]).randint(60, 255, 3))
        overlay = vis.copy()
        overlay[d["mask"] > 0] = color
        vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
        cnts, _ = cv2.findContours(d["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, color, 2)

        u, v = d["center"]
        a = np.deg2rad(d["angle"])
        p2 = (int(u + 0.5 * d["size"][0] * np.cos(a)), int(v + 0.5 * d["size"][0] * np.sin(a)))
        cv2.arrowedLine(vis, (int(u), int(v)), p2, (0, 255, 255), 2)

        label = f"{d['cls_name']} {d['conf']:.2f} {d['angle']:.0f}deg"
        cv2.putText(vis, label, (int(d["box"][0]), max(15, int(d["box"][1]) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if ROI is not None:
        cv2.rectangle(vis, (ROI[0], ROI[1]), (ROI[2], ROI[3]), (255, 255, 255), 1)
    return vis


# ============================================================
# TESTING
# ============================================================
def test():
    for p in (DATA_YAML, MODEL_PATH, TEST_IMAGES):
        if not p.exists():
            raise FileNotFoundError(f"Could not find:\n{p}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = YOLO(str(MODEL_PATH))

    # 1. Quantitative evaluation
    print("\nEvaluating model on test set...")
    metrics = model.val(
        data=str(DATA_YAML), split="test", imgsz=IMAGE_SIZE, batch=BATCH_SIZE,
        device=DEVICE, conf=0.001, project=str(OUTPUT_DIR), name="metrics",
        exist_ok=True, plots=True,
    )
    print(f"\nBox  mAP50: {metrics.box.map50:.3f}   mAP50-95: {metrics.box.map:.3f}")
    print(f"Mask mAP50: {metrics.seg.map50:.3f}   mAP50-95: {metrics.seg.map:.3f}")

    print("\nPer-class mask mAP50-95 (weakest first):")
    for idx, ap in sorted(zip(metrics.seg.ap_class_index, metrics.seg.maps), key=lambda t: t[1]):
        print(f"  {metrics.names[int(idx)]:<25s} {ap:.3f}")

    best_conf, best_f1 = best_f1_confidence(metrics)
    if CONFIDENCE is not None:
        conf = CONFIDENCE
    elif best_conf is not None:
        conf = best_conf
        print(f"\nBest mask-F1 confidence: {best_conf:.3f} (F1 = {best_f1:.3f})")
    else:
        conf = FALLBACK_CONFIDENCE
        print(f"\nCould not compute the best threshold, using {conf}")
    print(f"Using confidence {conf:.3f} for visualizations")

    # 2. Raw predictions
    print("\nGenerating raw predictions...")
    model.predict(
        source=str(TEST_IMAGES), imgsz=IMAGE_SIZE, conf=conf, device=DEVICE,
        show_labels=True, show_conf=True, boxes=True, save=True,
        project=str(OUTPUT_DIR), name="predictions_raw", exist_ok=True,
    )

    # 3. Postprocessed predictions
    print("\nGenerating postprocessed predictions...")
    post_dir = OUTPUT_DIR / "predictions_post"
    post_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    images = sorted(p for p in TEST_IMAGES.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for img_path in images:
        frame = imread_unicode(img_path)
        if frame is None:
            print(f"  Could not read {img_path.name}, skipping")
            continue

        result = model.predict(
            frame, imgsz=IMAGE_SIZE, conf=conf, iou=IOU, device=DEVICE,
            agnostic_nms=True, retina_masks=True, verbose=False,
        )[0]
        dets = postprocess(result, model.names, frame.shape)

        imwrite_unicode(post_dir / f"{img_path.stem}.jpg", draw(frame, dets))
        for d in dets:
            rows.append({
                "image": img_path.name, "class": d["cls_name"], "conf": round(d["conf"], 3),
                "center_u": round(d["center"][0], 1), "center_v": round(d["center"][1], 1),
                "long_px": round(d["size"][0], 1), "short_px": round(d["size"][1], 1),
                "angle_deg": round(d["angle"], 1),
            })
        print(f"  {img_path.name}: {len(dets)} objects")

    csv_path = OUTPUT_DIR / "detections_post.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "class", "conf", "center_u", "center_v", "long_px", "short_px", "angle_deg"])
        writer.writeheader()
        writer.writerows(rows)

    print("\nTesting complete.")
    print(f"Metrics + curves:        {OUTPUT_DIR / 'metrics'}")
    print(f"Raw predictions:         {OUTPUT_DIR / 'predictions_raw'}")
    print(f"Postprocessed images:    {post_dir}")
    print(f"Detections CSV:          {csv_path}")
    return metrics


if __name__ == "__main__":
    test()





