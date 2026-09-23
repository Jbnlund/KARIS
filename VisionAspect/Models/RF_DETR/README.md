# RF-DETR (medium) instance-segmentation

This folder now targets the real `rfdetr` package and the `RFDETRSegMedium` segmentation model.

Required dataset format
- Use a Roboflow-exported COCO instance-segmentation dataset.
- The project dataset is already in this format:
  `VisionAspect/Data/KARIS.v8i.coco-segmentation`
- The expected root structure is:
  - `train/_annotations.coco.json`
  - `valid/_annotations.coco.json`
  - `test/_annotations.coco.json`
  - image folders under each split
- Instance masks must be present, not just boxes.

Training
- Use the real RF-DETR API rather than a placeholder custom PyTorch dataset adapter.
- The script supports a tiny smoke run via `--subset-samples` so you can verify training end-to-end before training on the full dataset.

Example:

```bash
python VisionAspect/Models/RF_DETR/train_rf_detr.py --dataset-root "VisionAspect/Data/KARIS.v8i.coco-segmentation" --output-dir "runs/rf_detr" --epochs 1 --batch-size 1 --subset-samples 4
```

This creates a small temporary dataset subset and runs a one-epoch smoke training pass.

Full run:

```bash
python VisionAspect/Models/RF_DETR/train_rf_detr.py --dataset-root "VisionAspect/Data/KARIS.v8i.coco-segmentation" --output-dir "runs/rf_detr" --epochs 10 --batch-size 1
```

Evaluation

```bash
python VisionAspect/Models/RF_DETR/eval_rf_detr.py --dataset-root "VisionAspect/Data/KARIS.v8i.coco-segmentation" --checkpoint "runs/rf_detr/checkpoint_best_total.pth" --limit 5
```

Notes
- Install extra training dependencies if needed:
  `python -m pip install "rfdetr[train,loggers]"`
- RF-DETR expects a COCO-style root with `train` / `valid` / `test` splits; a plain YOLO folder is not the correct direct training input for this model unless it is first converted to COCO format.