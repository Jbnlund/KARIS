from pathlib import Path

from ultralytics import YOLO

from data_augmentation import AUGMENTATION_CONFIG


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\gusta\Desktop\Skolarbete\Skolarbete\KARISprojektBilder\KARIS.v1-test1.yolo26"
)

DATA_YAML = DATASET_ROOT / "data.yaml"

# Pretrained YOLO26 Medium instance-segmentation model
MODEL = "yolo26m-seg.pt"

# Training settings
EPOCHS = 100
IMAGE_SIZE = 1280
BATCH_SIZE = 8

# GPU
DEVICE = 0


# ============================================================
# TRAINING
# ============================================================

def train():

    # Check dataset configuration
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

    # Fine-tune on our dataset
    results = model.train(
        data=str(DATA_YAML),

        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,

        # Save training results
        project="runs/segment",
        name="karis_yolo26m",

        # Data augmentation
        **AUGMENTATION_CONFIG,
    )

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train()