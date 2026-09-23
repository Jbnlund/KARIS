from pathlib import Path
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
    r"C:\Users\kimsv\PycharmProjects\KARIS\VisionAspect\Models\YOLO8\runs\segment\runs\segment\karis_yolov8n-2\weights\best.pt"
)

OUTPUT_DIR = Path(
    r"C:\Users\kimsv\OneDrive - Mälardalens universitet\Desktop\temp8"
)

IMAGE_SIZE = 1280
BATCH_SIZE = 2
DEVICE = 0

# Minimum confidence required to visualize a prediction
CONFIDENCE = 0.25


# ============================================================
# TESTING
# ============================================================

def test():

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Could not find:\n{DATA_YAML}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not find:\n{MODEL_PATH}")

    print("Loading model...")
    model = YOLO(str(MODEL_PATH))

    # --------------------------------------------------------
    # 1. Quantitative evaluation
    # --------------------------------------------------------

    print("\nEvaluating model on test set...")

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(OUTPUT_DIR),
        name="metrics",
        plots=True,
    )

    # --------------------------------------------------------
    # 2. Visual predictions
    # --------------------------------------------------------

    print("\nGenerating test-set visualizations...")

    model.predict(
        source=str(TEST_IMAGES),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        device=DEVICE,

        # Visualization
        show_labels=True,
        show_conf=True,
        boxes=True,

        # Save predictions
        save=True,
        project=str(OUTPUT_DIR),
        name="predictions",
    )

    print("\nTesting complete.")
    print(f"Results saved to:\n{OUTPUT_DIR}")

    return metrics


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    test()