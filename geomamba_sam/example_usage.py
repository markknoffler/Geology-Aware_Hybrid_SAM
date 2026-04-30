#!/usr/bin/env python3
"""
Example script demonstrating how to use the Segment Anything Model (SAM) automatic mask generator.

This script shows how to:
1. Load a SAM model
2. Use the automatic mask generator to generate masks for an image
3. Save the generated masks
"""

import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import os

def main():
    # Define the checkpoint path and model type
    # You need to download the model checkpoint from:
    # https://github.com/facebookresearch/segment-anything#model-checkpoints
    checkpoint_path = "sam_vit_h_4b8939.pth"  # Replace with your actual checkpoint path
    model_type = "vit_h"  # Options: "vit_h", "vit_l", "vit_b", "default"

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file {checkpoint_path} not found!")
        print("Please download the checkpoint from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
        return

    # Load the model
    print("Loading model...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

    # For demonstration, we'll create a simple test image
    # In practice, you would load your own image
    test_image = np.zeros((512, 512, 3), dtype=np.uint8)
    test_image[100:200, 100:200] = [255, 0, 0]  # Red square
    test_image[300:400, 300:400] = [0, 255, 0]  # Green square

    # Generate masks
    print("Generating masks...")
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(test_image)

    print(f"Generated {len(masks)} masks")

    # Display information about the generated masks
    for i, mask in enumerate(masks):
        print(f"Mask {i}:")
        print(f"  Area: {mask['area']}")
        print(f"  Bounding box: {mask['bbox']}")
        print(f"  Predicted IoU: {mask['predicted_iou']}")
        print(f"  Stability score: {mask['stability_score']}")
        print(f"  Point coordinates: {mask['point_coords']}")
        print()

    # Save masks as images (if needed)
    output_dir = "generated_masks"
    os.makedirs(output_dir, exist_ok=True)

    for i, mask in enumerate(masks):
        # Save the mask as an image
        mask_image = mask['segmentation'].astype(np.uint8) * 255
        cv2.imwrite(os.path.join(output_dir, f"mask_{i}.png"), mask_image)

    print(f"Masks saved to {output_dir}")

if __name__ == "__main__":
    main()