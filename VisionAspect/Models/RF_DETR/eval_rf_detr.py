from __future__ import annotations

import argparse
from pathlib import Path
import json
import torch
from PIL import Image
import torchvision.transforms as T
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import rfdetr


DEFAULT_DATASET_ROOT = Path(
    r"D:\Robotik Projektkurs\repository\private\KARIS\VisionAspect\Data\KARIS.v8i.coco-segmentation"
)
DEFAULT_CHECKPOINT = Path("runs/rf_detr/checkpoint_best_total.pth")


def load_model(checkpoint: Path, device: str | None = None):
    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = rfdetr.from_checkpoint(str(checkpoint), device=device_name)
    model.to(device_name)
    model.eval()
    return model


def predict_on_folder(model, images_dir: Path, device: torch.device, imgsz=(640, 640)):
    transforms = T.Compose([T.Resize(imgsz), T.ToTensor()])
    results = []
    for p in sorted(images_dir.glob("*")):
        if p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        img = Image.open(p).convert("RGB")
        t = transforms(img).unsqueeze(0).to(device)
        with torch.inference_mode():
            preds = model.predict(t)
        results.append({"file": str(p.name), "preds": preds})
    return results


def predict_on_validation(model, image_dir: Path, limit: int = 5):
    predictions = []
    count = 0
    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        if count >= limit:
            break
        count += 1
        result = model.predict(str(image_path), threshold=0.5)
        predictions.append({"file": image_path.name, "num_detections": len(result) if hasattr(result, '__len__') else 0})
    return predictions


def evaluate_coco(coco_gt: Path, coco_results: list):
    coco = COCO(str(coco_gt))
    res_file = coco_gt.parent / "_tmp_results.json"
    with open(res_file, "w", encoding="utf-8") as f:
        json.dump(coco_results, f)

    coco_dt = coco.loadRes(str(res_file))
    coco_eval = COCOeval(coco, coco_dt, iouType="segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a saved RF-DETR segmentation checkpoint over validation images.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    valid_dir = args.dataset_root / "valid"
    if not valid_dir.exists():
        raise FileNotFoundError(f"Validation split not found: {valid_dir}")

    model = load_model(args.checkpoint, args.device)
    preds = predict_on_validation(model, valid_dir, limit=args.limit)
    print(f"Evaluated {len(preds)} validation images with checkpoint: {args.checkpoint}")
    for item in preds:
        print(item)

    # Uncomment the following lines to run COCO evaluation
    # ANN_TEST = args.dataset_root / "valid" / "_annotations.coco.json"
    # evaluate_coco(ANN_TEST, preds)
