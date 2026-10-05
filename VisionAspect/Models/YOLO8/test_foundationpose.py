"""
KARIS FoundationPose test, self-contained.

For every test image:
  1. YOLO-seg detects and classifies all objects (same mask cleanup as new_test.py).
  2. FoundationPose registers every detected object whose class has a CAD mesh
     -> 6D pose (4x4, object frame in camera optical frame).
     Classes without a mesh (e.g. tray slots) get a 3D centre point from depth.
  3. Self-check without ground truth: the mesh is rendered at the estimated pose
     and compared with the YOLO mask (silhouette IoU) and the observed depth
     (median depth error in mm). Bad poses show up as low IoU / large error.

Outputs (OUTPUT_DIR):
  annotated/<image>.jpg   masks, 3D boxes, axes, rendered silhouette, side panel
  poses.csv               one row per object (xyz, rpy, quaternion, checks)
  poses.json              same plus the full 4x4 matrices and K

Requirements:
  - Linux with FoundationPose built (nvdiffrast + its extensions), i.e. run it in
    the perception container, not on Windows.
  - Per image: RGB + depth ALIGNED to RGB (same resolution) + intrinsics K.
    A plain RGB test split (e.g. a Roboflow export) is not enough: FoundationPose
    needs depth, and resized/cropped exports no longer match the camera's K.

Pose convention: camera optical frame (x right, y down, z forward), translation
in metres, origin and axes = the CAD mesh's own frame (same as the ROS pipeline's
TF camera -> karis/<object_id>).
"""
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================
FOUNDATIONPOSE_DIR = Path("/opt/FoundationPose")      # repo root (contains estimater.py, weights/)

DATASET_ROOT = Path("/data/karis_fp_test")
TEST_IMAGES = DATASET_ROOT / "rgb"                    # colour images
DEPTH_DIR = DATASET_ROOT / "depth"                    # <same stem>.png (uint16) or .npy
DEPTH_SCALE_PNG = 0.001                               # uint16 PNG units -> metres (RealSense: 1 mm)
DEPTH_SCALE_NPY = 1.0                                 # .npy already in metres
MIN_DEPTH_M, MAX_DEPTH_M = 0.1, 3.0

# Intrinsics: cam_K.txt (3x3, FoundationPose demo format). If missing, CAM_K below is used.
# A per-image file DATASET_ROOT/cam_K/<stem>.txt overrides both.
CAM_K_FILE = DATASET_ROOT / "cam_K.txt"
CAM_K = np.array([[615.0, 0.0, 320.0],                # nominal D435i colour 640x480, replace
                  [0.0, 615.0, 240.0],                # with the values from camera_info
                  [0.0, 0.0, 1.0]])

MODEL_PATH = Path("/models/karis_yolov8s/best.pt")
OUTPUT_DIR = Path("/data/karis_fp_test/out")

# Detector class name -> CAD mesh. Names must match the YOLO class names exactly.
# scale: 0.001 for CAD in mm, 1.0 for metres.
MESHES = {
    "rund_mutter": {"mesh": "/meshes/rund_mutter.obj", "scale": 0.001},
    # "class_name": {"mesh": "/meshes/part.stl", "scale": 0.001},
}

IMAGE_SIZE = 1280
DEVICE = 0
CONFIDENCE = 0.25        # use the best-F1 value that new_test.py prints
IOU = 0.6
MIN_MASK_AREA_PX = 300
MIN_VALID_DEPTH_PX = 50  # skip registration if the mask has fewer valid depth pixels
ROI = None               # (x1, y1, x2, y2) or None

REG_ITERATIONS = 5       # FoundationPose refinement iterations (demo default 5)
RENDER_CHECK = True      # silhouette IoU + depth error self-check

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ============================================================
# I/O helpers
# ============================================================
def imread_unicode(path: Path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: Path, img):
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if ok:
        buf.tofile(str(path))


def load_K(stem: str) -> np.ndarray:
    per_image = DATASET_ROOT / "cam_K" / f"{stem}.txt"
    for p in (per_image, CAM_K_FILE):
        if p is not None and p.exists():
            return np.loadtxt(p).reshape(3, 3).astype(np.float64)
    return CAM_K.astype(np.float64)


def load_depth(stem: str, shape) -> np.ndarray | None:
    for ext in (".png", ".npy", ".tif", ".tiff"):
        p = DEPTH_DIR / f"{stem}{ext}"
        if p.exists():
            break
    else:
        return None
    if p.suffix == ".npy":
        d = np.load(p).astype(np.float32) * DEPTH_SCALE_NPY
    else:
        d = imread_unicode(p, cv2.IMREAD_UNCHANGED).astype(np.float32) * DEPTH_SCALE_PNG
    if d.ndim == 3:
        d = d[..., 0]
    if d.shape != tuple(shape[:2]):
        raise ValueError(
            f"Depth {p.name} is {d.shape}, colour is {shape[:2]}. Depth must be aligned to "
            f"colour at the same resolution (aligned_depth_to_color).")
    d[(d < MIN_DEPTH_M) | (d > MAX_DEPTH_M)] = 0.0
    return d


def check_K(K, shape, name):
    h, w = shape[:2]
    if abs(2 * K[0, 2] - w) > 0.25 * w or abs(2 * K[1, 2] - h) > 0.25 * h:
        print(f"  WARNING {name}: principal point ({K[0, 2]:.0f}, {K[1, 2]:.0f}) does not fit "
              f"image {w}x{h}. Was the image resized/cropped? Poses will be wrong.")


# ============================================================
# Detection (same cleanup as new_test.py)
# ============================================================
def clean_mask(mask: np.ndarray, k: int = 3) -> np.ndarray:
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


def detect(model, frame_bgr):
    result = model.predict(frame_bgr, imgsz=IMAGE_SIZE, conf=CONFIDENCE, iou=IOU, device=DEVICE,
                           agnostic_nms=True, retina_masks=True, verbose=False)[0]
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return []
    h, w = frame_bgr.shape[:2]
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
        ys, xs = np.nonzero(m)
        cu, cv_ = float(xs.mean()), float(ys.mean())
        if ROI is not None:
            x1, y1, x2, y2 = ROI
            if not (x1 <= cu <= x2 and y1 <= cv_ <= y2):
                continue
        dets.append({"cls_id": int(cls[i]), "cls_name": model.names[int(cls[i])],
                     "conf": float(conf[i]), "box": boxes[i], "mask": m > 0,
                     "center_px": (cu, cv_)})
    # stable ids: class, then left to right
    dets.sort(key=lambda d: (d["cls_name"], d["center_px"][0]))
    counters = {}
    for d in dets:
        counters[d["cls_name"]] = counters.get(d["cls_name"], 0) + 1
        d["object_id"] = f"{d['cls_name']}_{counters[d['cls_name']]}"
    return dets


# ============================================================
# FoundationPose
# ============================================================
def build_estimators(class_names):
    sys.path.insert(0, str(FOUNDATIONPOSE_DIR))
    import trimesh
    import nvdiffrast.torch as dr
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    import Utils as fpu

    fpu.set_seed(0)
    debug_dir = OUTPUT_DIR / "fp_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # scorer/refiner networks and the GL context are shared by all classes
    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    ests = {}
    for cls_name, cfg in MESHES.items():
        if cls_name not in class_names:
            print(f"  WARNING: mesh class '{cls_name}' is not a YOLO class {sorted(class_names)}")
            continue
        mesh = trimesh.load(cfg["mesh"], force="mesh")
        mesh.apply_scale(cfg.get("scale", 1.0))
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        if extents.max() > 1.0:
            print(f"  WARNING: '{cls_name}' mesh is {extents.max():.2f} m across. Wrong scale?")
        est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                             mesh=mesh, scorer=scorer, refiner=refiner,
                             debug_dir=str(debug_dir), debug=0, glctx=glctx)
        ests[cls_name] = {"est": est, "to_origin": to_origin,
                          "bbox": np.stack([-extents / 2, extents / 2], axis=0),
                          "extent": float(extents.max())}
        print(f"  {cls_name}: {Path(cfg['mesh']).name}, extents {np.round(extents * 1000, 1)} mm")
    return ests, fpu, glctx


def render_check(entry, pose, K, depth, mask, fpu, glctx):
    """Render the mesh at `pose`; return (silhouette IoU, median |depth error| mm, rendered mask)."""
    try:
        import torch
        est = entry["est"]
        # FoundationPose stores a centred copy of the mesh; register() returns the pose
        # of the original mesh frame, so shift back to the centred one for rendering.
        pose_c = pose.copy()
        pose_c[:3, 3] = pose[:3, 3] + pose[:3, :3] @ np.asarray(est.model_center).reshape(3)
        h, w = depth.shape
        _, rdepth, _ = fpu.nvdiffrast_render(
            K=K, H=h, W=w, glctx=glctx, mesh_tensors=est.mesh_tensors, output_size=(h, w),
            ob_in_cams=torch.as_tensor(pose_c[None], device="cuda", dtype=torch.float),
            use_light=False)
        rdepth = rdepth[0].detach().cpu().numpy()
        rmask = rdepth > 0
        union = np.count_nonzero(rmask | mask)
        iou = np.count_nonzero(rmask & mask) / union if union else 0.0
        both = rmask & mask & (depth > 0)
        err = float(np.median(np.abs(rdepth[both] - depth[both])) * 1000) if both.any() else None
        return float(iou), err, rmask
    except Exception as e:  # never let the self-check kill the run
        print(f"    render check failed: {e}")
        return None, None, None


def centre_from_depth(mask, depth, K):
    valid = mask & (depth > 0)
    if np.count_nonzero(valid) < MIN_VALID_DEPTH_PX:
        return None
    ys, xs = np.nonzero(mask)
    u, v = xs.mean(), ys.mean()
    z = float(np.median(depth[valid]))
    return np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])


# ============================================================
# Rotation helpers
# ============================================================
def rot_to_rpy_deg(R):
    """R = Rz(yaw) Ry(pitch) Rx(roll), same convention as ROS / tf."""
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([roll, pitch, yaw])


def rot_to_quat_xyzw(R):
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


# ============================================================
# Drawing
# ============================================================
def class_color(cls_id):
    return tuple(int(x) for x in np.random.RandomState(cls_id).randint(60, 255, 3))


def draw(frame_rgb, objs, ests, K, fpu):
    vis = frame_rgb.copy()
    for o in objs:
        color = class_color(o["cls_id"])
        overlay = vis.copy()
        overlay[o["mask"]] = color
        vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
        if o.get("render_mask") is not None:   # rendered mesh silhouette in white
            cnts, _ = cv2.findContours(o["render_mask"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (255, 255, 255), 1)
    for o in objs:
        color = class_color(o["cls_id"])
        if o["pose"] is not None:
            e = ests[o["cls_name"]]
            center_pose = o["pose"] @ np.linalg.inv(e["to_origin"])
            vis = fpu.draw_posed_3d_box(K, img=vis, ob_in_cam=center_pose, bbox=e["bbox"])
            vis = fpu.draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.6 * e["extent"], K=K,
                                    thickness=2, transparency=0, is_input_rgb=True)
        x1, y1 = int(o["box"][0]), int(o["box"][1])
        cv2.putText(vis, o["object_id"], (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # side panel with xyz / rpy per object
    h = vis.shape[0]
    panel = np.full((h, 430, 3), 30, np.uint8)
    y = 22
    for o in objs:
        if y > h - 10:
            break
        if o["pose"] is not None:
            t = o["pose"][:3, 3]
            r = rot_to_rpy_deg(o["pose"][:3, :3])
            lines = [f"{o['object_id']} ({o['conf']:.2f})",
                     f"  xyz [m] {t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}",
                     f"  rpy [deg] {r[0]:+.0f} {r[1]:+.0f} {r[2]:+.0f}"]
            if o["mask_iou"] is not None:
                err = f"{o['depth_err_mm']:.1f}" if o["depth_err_mm"] is not None else "-"
                lines.append(f"  IoU {o['mask_iou']:.2f}  dz {err} mm")
        elif o["centre"] is not None:
            t = o["centre"]
            lines = [f"{o['object_id']} ({o['conf']:.2f}) centre only",
                     f"  xyz [m] {t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}"]
        else:
            lines = [f"{o['object_id']} ({o['conf']:.2f})", f"  {o['note']}"]
        for i, line in enumerate(lines):
            cv2.putText(panel, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        class_color(o["cls_id"]) if i == 0 else (220, 220, 220), 1)
            y += 18
        y += 6
    return np.hstack([vis, panel])


# ============================================================
# Main
# ============================================================
def test():
    for p in (MODEL_PATH, TEST_IMAGES, DEPTH_DIR, FOUNDATIONPOSE_DIR):
        if not p.exists():
            raise FileNotFoundError(f"Could not find:\n{p}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ann_dir = OUTPUT_DIR / "annotated"
    ann_dir.mkdir(exist_ok=True)

    print("Loading YOLO...")
    model = YOLO(str(MODEL_PATH))
    print("Loading FoundationPose + meshes...")
    ests, fpu, glctx = build_estimators(set(model.names.values()))
    no_mesh = sorted(set(model.names.values()) - set(ests))
    if no_mesh:
        print(f"  Classes without mesh (centre point only): {no_mesh}")

    rows, json_out = [], {}
    images = sorted(p for p in TEST_IMAGES.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    print(f"\nProcessing {len(images)} images...")
    for img_path in images:
        frame_bgr = imread_unicode(img_path)
        if frame_bgr is None:
            print(f"  {img_path.name}: could not read, skipping")
            continue
        depth = load_depth(img_path.stem, frame_bgr.shape)
        if depth is None:
            print(f"  {img_path.name}: no depth file in {DEPTH_DIR}, skipping")
            continue
        K = load_K(img_path.stem)
        check_K(K, frame_bgr.shape, img_path.name)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        objs = detect(model, frame_bgr)
        for o in objs:
            o.update(pose=None, centre=None, mask_iou=None, depth_err_mm=None,
                     render_mask=None, reg_ms=None, note="")
            n_valid = int(np.count_nonzero(o["mask"] & (depth > 0)))
            if o["cls_name"] in ests:
                if n_valid < MIN_VALID_DEPTH_PX:
                    o["note"] = f"no pose: only {n_valid} depth px in mask"
                    continue
                e = ests[o["cls_name"]]
                t0 = time.perf_counter()
                pose = e["est"].register(K=K, rgb=rgb, depth=depth, ob_mask=o["mask"],
                                         iteration=REG_ITERATIONS)
                o["reg_ms"] = (time.perf_counter() - t0) * 1000
                o["pose"] = np.asarray(pose, dtype=np.float64).reshape(4, 4)
                o["centre"] = o["pose"][:3, 3]
                if RENDER_CHECK:
                    o["mask_iou"], o["depth_err_mm"], o["render_mask"] = render_check(
                        e, o["pose"], K, depth, o["mask"], fpu, glctx)
            else:
                o["centre"] = centre_from_depth(o["mask"], depth, K)
                o["note"] = "no mesh: centre only" if o["centre"] is not None \
                    else "no mesh and no depth in mask"

        vis = draw(rgb, objs, ests, K, fpu)
        imwrite_unicode(ann_dir / f"{img_path.stem}.jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        json_out[img_path.name] = {"K": K.tolist(), "objects": []}
        for o in objs:
            row = {"image": img_path.name, "object_id": o["object_id"], "class": o["cls_name"],
                   "conf": round(o["conf"], 3), "has_6d_pose": o["pose"] is not None}
            t = o["centre"] if o["centre"] is not None else [np.nan] * 3
            row.update(x_m=round(float(t[0]), 4), y_m=round(float(t[1]), 4),
                       z_m=round(float(t[2]), 4))
            if o["pose"] is not None:
                r = rot_to_rpy_deg(o["pose"][:3, :3])
                q = rot_to_quat_xyzw(o["pose"][:3, :3])
                row.update(roll_deg=round(r[0], 1), pitch_deg=round(r[1], 1), yaw_deg=round(r[2], 1),
                           qx=round(q[0], 5), qy=round(q[1], 5), qz=round(q[2], 5), qw=round(q[3], 5))
            row.update(mask_iou=None if o["mask_iou"] is None else round(o["mask_iou"], 3),
                       depth_err_mm=None if o["depth_err_mm"] is None else round(o["depth_err_mm"], 1),
                       reg_ms=None if o["reg_ms"] is None else round(o["reg_ms"], 1),
                       note=o["note"])
            rows.append(row)
            json_out[img_path.name]["objects"].append({
                **{k: v for k, v in row.items() if k != "image"},
                "pose_cam_obj": None if o["pose"] is None else o["pose"].tolist()})

        n_pose = sum(o["pose"] is not None for o in objs)
        print(f"  {img_path.name}: {len(objs)} objects, {n_pose} with 6D pose")

    fields = ["image", "object_id", "class", "conf", "has_6d_pose", "x_m", "y_m", "z_m",
              "roll_deg", "pitch_deg", "yaw_deg", "qx", "qy", "qz", "qw",
              "mask_iou", "depth_err_mm", "reg_ms", "note"]
    with open(OUTPUT_DIR / "poses.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(OUTPUT_DIR / "poses.json", "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2)

    # summary per class
    print("\nSummary per class (median over objects with a 6D pose):")
    print(f"  {'class':<22s} {'dets':>5s} {'posed':>6s} {'IoU':>6s} {'dz mm':>7s} {'reg ms':>7s}")
    for cls_name in sorted({r["class"] for r in rows}):
        rs = [r for r in rows if r["class"] == cls_name]
        posed = [r for r in rs if r["has_6d_pose"]]

        def med(key):
            vals = [r[key] for r in posed if r[key] is not None]
            return f"{np.median(vals):.2f}" if vals else "-"
        print(f"  {cls_name:<22s} {len(rs):>5d} {len(posed):>6d} {med('mask_iou'):>6s} "
              f"{med('depth_err_mm'):>7s} {med('reg_ms'):>7s}")

    print("\nTesting complete.")
    print(f"Annotated images: {ann_dir}")
    print(f"Poses CSV:        {OUTPUT_DIR / 'poses.csv'}")
    print(f"Poses JSON:       {OUTPUT_DIR / 'poses.json'}")


if __name__ == "__main__":
    test()
