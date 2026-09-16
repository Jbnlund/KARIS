"""
Data augmentation configuration for YOLO instance-segmentation training.

The augmentations are applied dynamically by Ultralytics during training.
No augmented images are saved to disk.
"""


AUGMENTATION_CONFIG = {

    # --------------------------------------------------------
    # Colour / lighting augmentation
    # --------------------------------------------------------

    # Hue variation
    "hsv_h": 0.015,

    # Saturation variation
    "hsv_s": 0.4,

    # Brightness variation
    "hsv_v": 0.3,


    # --------------------------------------------------------
    # Geometric augmentation
    # --------------------------------------------------------

    # Maximum rotation in degrees
    "degrees": 15.0,

    # Horizontal/vertical translation
    # Fraction of image dimensions
    "translate": 0.1,

    # Random scaling
    "scale": 0.3,

    # Perspective transformation
    "perspective": 0.0005,


    # --------------------------------------------------------
    # Flipping
    # --------------------------------------------------------

    # Horizontal flip probability
    "fliplr": 0.5,

    # Vertical flip probability
    "flipud": 0.0,


    # --------------------------------------------------------
    # YOLO-specific augmentation
    # --------------------------------------------------------

    # Combine four training images into one
    "mosaic": 0.5,

    # Mix two images together
    "mixup": 0.0,

    # Copy segmented objects between images
    "copy_paste": 0.0,
}