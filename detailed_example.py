#!/usr/bin/env python3
"""
Detailed example showing how to use the automatic mask generator with various parameters.
"""

import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import os

def create_sample_image():
    """Create a sample image with different shapes for testing."""
    image = np.zeros((512, 512, 3), dtype=np.uint8)

    # Add some sample shapes
    cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle
    cv2.circle(image, (300, 300), 50, (0, 255, 0), -1)  # Green circle
    cv2.rectangle(image, (400, 100), (450, 150), (0, 0, 255), -1)  # Red rectangle

    return image

def main():
    # Create a sample image for demonstration
    print("Creating sample image...")
    image = create_sample_image()

    # Define checkpoint path (you need to download this from the official repository)
    checkpoint_path = "sam_vit_h_4b8939.pth"  # Replace with your actual checkpoint path
    model_type = "vit_h"

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file {checkpoint_path} not found!")
        print("Please download the checkpoint from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
        print("For this example, we'll use default parameters without loading the model.")
        return

    # Load the model
    print("Loading model...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

    # Example 1: Basic automatic mask generation with default parameters
    print("\n=== Basic Automatic Mask Generation ===")
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    print(f"Generated {len(masks)} masks with default parameters")

    # Example 2: Automatic mask generation with custom parameters
    print("\n=== Custom Parameters ===")
    custom_mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,           # Sample 16x16 grid points
        pred_iou_thresh=0.85,         # Filter masks with predicted IoU < 0.85
        stability_score_thresh=0.90,  # Filter masks with stability score < 0.90
        min_mask_region_area=100,     # Remove masks smaller than 100 pixels
        output_mode="binary_mask"     # Output as binary masks
    )

    masks_custom = custom_mask_generator.generate(image)
    print(f"Generated {len(masks_custom)} masks with custom parameters")

    # Example 3: Using different output modes
    print("\n=== Different Output Modes ===")

    # COCO-style RLE output (requires pycocotools)
    try:
        rle_mask_generator = SamAutomaticMaskGenerator(sam, output_mode="coco_rle")
        masks_rle = rle_mask_generator.generate(image)
        print(f"Generated {len(masks_rle)} masks with COCO RLE output")
    except ImportError:
        print("pycocotools not installed, skipping COCO RLE output example")

    # Uncompressed RLE output
    uncompressed_mask_generator = SamAutomaticMaskGenerator(sam, output_mode="uncompressed_rle")
    masks_uncompressed = uncompressed_mask_generator.generate(image)
    print(f"Generated {len(masks_uncompressed)} masks with uncompressed RLE output")

    # Display mask information
    print("\n=== Mask Information ===")
    for i, mask in enumerate(masks[:3]):  # Show info for first 3 masks
        print(f"Mask {i}:")
        print(f"  Area: {mask['area']} pixels")
        print(f"  Bounding box: {mask['bbox']}")
        print(f"  Predicted IoU: {mask['predicted_iou']:.3f}")
        print(f"  Stability score: {mask['stability_score']:.3f}")
        print(f"  Point coordinates: {mask['point_coords'][0]}")
        print()

    # Save masks
    output_dir = "detailed_masks"
    os.makedirs(output_dir, exist_ok=True)

    for i, mask in enumerate(masks[:5]):  # Save first 5 masks
        mask_image = mask['segmentation'].astype(np.uint8) * 255
        cv2.imwrite(os.path.join(output_dir, f"mask_{i}.png"), mask_image)

    print(f"Sample masks saved to {output_dir}")

if __name__ == "__main__":
    main()