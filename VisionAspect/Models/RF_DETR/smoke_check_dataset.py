from pathlib import Path
import json
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from PIL import Image, ImageDraw
import numpy as np


def decode_segmentation(seg, height, width):
    # seg can be polygon (list) or RLE
    if isinstance(seg, dict) and "counts" in seg:
        # RLE
        m = mask_utils.decode(seg)
        return m.astype(np.uint8)
    elif isinstance(seg, list):
        # polygon(s)
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        for poly in seg:
            if len(poly) >= 6:
                xy = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
                draw.polygon(xy, outline=1, fill=1)
        return np.array(mask, dtype=np.uint8)
    else:
        raise ValueError("Unsupported segmentation format")


def smoke_check(dataset_root: Path):
    ann_train = dataset_root / "train" / "_annotations.coco.json"
    if not ann_train.exists():
        ann_train = dataset_root / "train" / "annotations" / "instances_train.json"
    if not ann_train.exists():
        raise FileNotFoundError(f"Could not find train annotation json under {dataset_root}")

    coco = COCO(str(ann_train))
    img_ids = coco.getImgIds()
    print(f"Images: {len(img_ids)}")
    ann_ids = coco.getAnnIds()
    print(f"Annotations: {len(ann_ids)}")
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    print(f"Categories: {len(cats)}")

    # pick a sample image with at least one segmentation
    sample_img = None
    for iid in img_ids:
        a_ids = coco.getAnnIds(imgIds=iid)
        anns = coco.loadAnns(a_ids)
        if any("segmentation" in a and a["segmentation"] for a in anns):
            sample_img = coco.loadImgs(iid)[0]
            sample_anns = anns
            break

    if sample_img is None:
        print("No segmentation annotations found in training set.")
        return

    print(f"Sample image: {sample_img['file_name']} (id={sample_img['id']})")
    img_path = dataset_root / "train" / sample_img["file_name"]
    if not img_path.exists():
        # try parent dataset root
        img_path = dataset_root / sample_img["file_name"]
    print(f"Image path resolved to: {img_path}")

    # Decode first segmentation and write mask to disk
    seg = sample_anns[0].get("segmentation")
    h = sample_img["height"]
    w = sample_img["width"]
    mask = decode_segmentation(seg, h, w)
    out_dir = Path("runs/rf_detr")
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img.save(out_dir / "smoke_mask.png")
    print(f"Wrote smoke mask to: {out_dir / 'smoke_mask.png'}")


if __name__ == "__main__":
    ds = Path(r"d:/Robotik Projektkurs/repository/private/KARIS/VisionAspect/Data/KARIS.v8i.coco-segmentation")
    smoke_check(ds)
