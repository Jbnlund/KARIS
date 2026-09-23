from PIL import Image
from pathlib import Path

img_path = Path(r"VisionAspect/Data/KARIS.v8i.coco-segmentation/train/IMG_2284_jpg.rf.0e6a9d84c2cc4202d2fe400cab020c79.jpg")
mask_path = Path(r"runs/rf_detr/smoke_mask.png")
out_path = Path(r"runs/rf_detr/overlay.png")

img = Image.open(img_path).convert('RGBA')
mask = Image.open(mask_path).convert('L').resize(img.size)

overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
for y in range(img.size[1]):
    for x in range(img.size[0]):
        val = mask.getpixel((x, y))
        if val > 10:
            overlay.putpixel((x, y), (255, 0, 0, 60))

img = Image.alpha_composite(img, overlay)
out_path.parent.mkdir(parents=True, exist_ok=True)
img.save(out_path)
print(f"Wrote overlay to: {out_path}")
