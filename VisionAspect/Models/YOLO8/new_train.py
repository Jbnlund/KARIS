import yaml
from pathlib import Path
from ultralytics import YOLO  # pip install ultralytics

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DATASET_ROOT = Path(
    r"C:\Users\kimsv\OneDrive - Mälardalens universitet\Desktop\KARIS.v1-test1.yolo26"
)
DATA_YAML = DATASET_ROOT / "data.yaml"

# n = nano, s = small, m = medium. Compare s/m against your n baseline.
# "yolo26s-seg.pt" is also worth a run (NMS-free, needs a recent ultralytics).
MODEL = "yolov8s-seg.pt"
RUN_NAME = "karis_yolov8s"

# training
EPOCHS = 50
PATIENCE = 10            # stop early if validation does not improve for 10 epochs
IMAGE_SIZE = 1280
BATCH_SIZE = 2           # lower if you run out of GPU memory
DEVICE = 0
WORKERS = 4              # dataloader workers (lower to 2 or 0 if Windows complains)
CACHE = False            # True/"ram" is faster but 1280 px images need a lot of RAM
SEED = 0                 # fixed seed so runs are comparable
COS_LR = True            # cosine learning rate schedule
CLOSE_MOSAIC = 10        # turn mosaic off for the last N epochs

AUGMENTATION_ON = True

# color
HSV_H = 0.015
HSV_S = 0.4
HSV_V = 0.3

# geometry
DEGREES = 15.0           # fine for top-down views, lower it for angled cameras
TRANSLATE = 0.15
SCALE = 0.3
PERSPECTIVE = 0.0005

# flips: keep at 0 if parts are asymmetric (a mirrored part is a different shape)
FLIPLR = 0.25
FLIPUD = 0.25

# compositing, helps a lot for cluttered scenes
MOSAIC = 1.0
MIXUP = 0.0
COPY_PASTE = 0.3         # segmentation only: pastes object instances into other images


def build_aug_args() -> dict:
    if AUGMENTATION_ON:
        return dict(
            hsv_h=HSV_H, hsv_s=HSV_S, hsv_v=HSV_V,
            degrees=DEGREES, translate=TRANSLATE, scale=SCALE,
            perspective=PERSPECTIVE, fliplr=FLIPLR, flipud=FLIPUD,
            mosaic=MOSAIC, mixup=MIXUP, copy_paste=COPY_PASTE,
        )
    return dict(
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, perspective=0.0,
        fliplr=0.0, flipud=0.0, mosaic=0.0, mixup=0.0, copy_paste=0.0,
    )


def train():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Could not find dataset configuration:\n{DATA_YAML}")

    print("Starting training")
    print(f"Model:   {MODEL}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Epochs:  {EPOCHS} (patience {PATIENCE})")
    print(f"Image:   {IMAGE_SIZE}")
    print(f"Batch:   {BATCH_SIZE}")
    print(f"Augmentation: {'on' if AUGMENTATION_ON else 'off'}")

    model = YOLO(MODEL)
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        cache=CACHE,
        seed=SEED,
        cos_lr=COS_LR,
        close_mosaic=CLOSE_MOSAIC,
        project="runs/segment",
        name=RUN_NAME,
        **build_aug_args(),
    )
    return results


def evaluate(save_dir: Path):
    """Evaluate best.pt on the test split (falls back to val), print per-class
    mask metrics and the confidence threshold that maximizes F1."""
    best = Path(save_dir) / "weights" / "best.pt"
    model = YOLO(str(best))

    with open(DATA_YAML, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    split = "test" if cfg.get("test") else "val"
    print(f"\nEvaluating {best.name} on the '{split}' split")

    m = model.val(
        data=str(DATA_YAML), split=split, imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE, device=DEVICE, conf=0.001, plots=True,
    )

    print(f"Box  mAP50: {m.box.map50:.3f}   mAP50-95: {m.box.map:.3f}")
    print(f"Mask mAP50: {m.seg.map50:.3f}   mAP50-95: {m.seg.map:.3f}")

    print("\nPer-class mask mAP50-95 (weakest first):")
    per_class = sorted(zip(m.seg.ap_class_index, m.seg.maps), key=lambda t: t[1])
    for idx, ap in per_class:
        print(f"  {m.names[int(idx)]:<25s} {ap:.3f}")

    try:
        f1 = m.seg.f1_curve.mean(0)      # mean over classes, shape (1000,)
        i = int(f1.argmax())
        print(f"\nBest confidence threshold (mask F1): {m.seg.px[i]:.3f}  (F1 = {f1[i]:.3f})")
        print("Use this as CONF in infer.py")
    except Exception as e:  # attribute names differ between versions
        print(f"\nCould not compute best threshold automatically ({e}). "
              f"Read it from MaskF1_curve.png in {save_dir}")


if __name__ == "__main__":
    res = train()
    evaluate(Path(res.save_dir))
