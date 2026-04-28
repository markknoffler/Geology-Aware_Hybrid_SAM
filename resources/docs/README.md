# Segment Anything Model (SAM) Usage Examples

This repository contains examples for using the Segment Anything Model (SAM) automatic mask generator.

## Overview

The Segment Anything Model (SAM) is a state-of-the-art segmentation model that can generate masks for any object in an image. This repository includes examples showing how to use the automatic mask generator functionality.

## Installation

Before using these examples, you need to install the required dependencies:

```bash
pip install torch torchvision numpy opencv-python
```

You also need to download the SAM model checkpoint from the official repository:
https://github.com/facebookresearch/segment-anything#model-checkpoints

## Files

- `example_usage.py`: Simple example showing basic usage
- `detailed_example.py`: More comprehensive example with various parameters
- `automatic_mask_generator.py`: Core implementation of the automatic mask generator
- `predictor.py`: Core prediction functionality
- `build_sam.py`: Model building and checkpoint loading

## Usage

### Basic Usage

```python
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np

# Load the model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")

# Create or load an image
image = cv2.imread("your_image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Generate masks
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

# Process the masks
for mask in masks:
    print(f"Mask area: {mask['area']}")
    print(f"Bounding box: {mask['bbox']}")
```

### Custom Parameters

You can customize the mask generation with various parameters:

```python
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=16,           # Sample 16x16 grid points
    pred_iou_thresh=0.85,         # Filter masks with predicted IoU < 0.85
    stability_score_thresh=0.90,  # Filter masks with stability score < 0.90
    min_mask_region_area=100,     # Remove masks smaller than 100 pixels
    output_mode="binary_mask"     # Output as binary masks
)
```

## Parameters

- `points_per_side`: Number of points to sample along one side of the image (default: 32)
- `pred_iou_thresh`: Threshold for filtering masks by predicted IoU (default: 0.88)
- `stability_score_thresh`: Threshold for filtering masks by stability score (default: 0.95)
- `min_mask_region_area`: Minimum area for masks (default: 0)
- `output_mode`: Output format ('binary_mask', 'uncompressed_rle', or 'coco_rle')

## Output Format

Each mask record contains:
- `segmentation`: The mask (binary mask, RLE, or COCO RLE)
- `bbox`: Bounding box in XYWH format
- `area`: Area of the mask in pixels
- `predicted_iou`: Model's prediction of mask quality
- `point_coords`: Coordinates of points used for mask generation
- `stability_score`: Measure of mask quality
- `crop_box`: Crop box used for mask generation

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- OpenCV (for image processing)
- pycocotools (for COCO RLE output, optional)

## License

This code is based on the Segment Anything Model from Meta Platforms, Inc. and affiliates.