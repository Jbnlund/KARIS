from pathlib import Path
from ultralytics import YOLO


# CONFIGURATION

DATASET_ROOT = Path(
    r"C:\Users\kimsv\OneDrive - Mälardalens universitet\Desktop\KARIS.v1-test1.yolo26"
)

DATA_YAML = DATASET_ROOT / "data.yaml"

# Model
MODEL = "yolo26m-seg.pt"

# Training settings
EPOCHS = 100
IMAGE_SIZE = 1280
BATCH_SIZE = 1
DEVICE = 0


# DATA AUGMENTATION

# Colour / lighting
HSV_H = 0.015
HSV_S = 0.4
HSV_V = 0.3

# Geometric
DEGREES = 15.0
TRANSLATE = 0.1
SCALE = 0.3
PERSPECTIVE = 0.0005

# Flipping
FLIPLR = 0.5
FLIPUD = 0.0

# YOLO-specific
MOSAIC = 0.5
MIXUP = 0.0
COPY_PASTE = 0.0


# TRAINING

def train():

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Could not find dataset configuration:\n{DATA_YAML}"
        )

    print("Starting YOLO26 instance-segmentation training")
    print(f"Model:   {MODEL}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Epochs:  {EPOCHS}")
    print(f"Image:   {IMAGE_SIZE}")
    print(f"Batch:   {BATCH_SIZE}")

    # Load pretrained YOLO26m segmentation model
    model = YOLO(MODEL)

    # Train
    results = model.train(
        data=str(DATA_YAML),

        # Training
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,

        # Output
        project="runs/segment",
        name="karis_yolo26m",

        # Colour / lighting augmentation
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,

        # Geometric augmentation
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        perspective=PERSPECTIVE,

        # Flipping
        fliplr=FLIPLR,
        flipud=FLIPUD,

        # YOLO-specific augmentation
        mosaic=MOSAIC,
        mixup=MIXUP,
        copy_paste=COPY_PASTE,
    )

    return results


# MAIN

if __name__ == "__main__":
    train()