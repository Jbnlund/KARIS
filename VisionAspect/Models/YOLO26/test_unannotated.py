from pathlib import Path
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

# Trained YOLO model
MODEL_PATH = Path(
    r"C:\Users\gusta\Desktop\Skolarbete\Skolarbete\Åk5\Testrepot\KARIS\VisionAspect\Models\YOLO26\runs\karis_yolo26m\weights\best.pt"
)

# Folder containing UNANNOTATED images
IMAGE_FOLDER = Path(
    r"C:\Users\gusta\Desktop\Skolarbete\Skolarbete\KARISprojectBilder\Unannotated"
)

# Where visualized predictions will be saved
OUTPUT_DIR = Path(
    r"C:\Users\gusta\Desktop\Skolarbete\Skolarbete\Åk5\Testrepot\KARIS\VisionAspect\Models\YOLO26\runs\unannotated_test"
)

IMAGE_SIZE = 1280
DEVICE = 0

# Minimum confidence required for a prediction to be shown
CONFIDENCE = 0.25


# ============================================================
# INFERENCE
# ============================================================

def test_unannotated_images():

    # Check that model exists
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find model:\n{MODEL_PATH}"
        )

    # Check that image folder exists
    if not IMAGE_FOLDER.exists():
        raise FileNotFoundError(
            f"Could not find image folder:\n{IMAGE_FOLDER}"
        )

    print("Loading trained model...")
    model = YOLO(str(MODEL_PATH))

    print("\nRunning inference on unannotated images...")

    results = model.predict(
        source=str(IMAGE_FOLDER),

        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        device=DEVICE,

        # Visualization
        show_labels=True,
        show_conf=True,
        boxes=True,

        # Save visualized predictions
        save=True,
        project=str(OUTPUT_DIR),
        name="predictions",
    )

    print("\nInference complete.")
    print(f"Processed {len(results)} images.")
    print(f"Visualizations saved to:\n{OUTPUT_DIR / 'predictions'}")

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    test_unannotated_images()