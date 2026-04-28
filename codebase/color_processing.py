import os
import tempfile
import shutil
from PIL import Image

def process_image(image_path):
    """Process an image to adjust specific colors"""
    try:
        ext = os.path.splitext(image_path)[1].lower()
        with Image.open(image_path) as img:
            # Convert to RGB if not already
            if img.mode != 'RGB':
                rgb_img = img.convert('RGB')
            else:
                rgb_img = img.copy()
            
            # Convert to HSV color space
            hsv_img = rgb_img.convert('HSV')
            pixels = hsv_img.load()
            width, height = hsv_img.size
            
            # Process each pixel
            for x in range(width):
                for y in range(height):
                    h, s, v = pixels[x, y]
                    
                    # Hue ranges (scaled to 0-255)
                    # Warm colors: brown, yellow, orange, muddy (hue 10-60° → ~7-43)
                    if 7 <= h <= 43 and v >= 30:
                        # Enhance saturation and brightness
                        s = min(255, int(s * 1.5))
                        v = min(255, int(v * 1.1))
                        pixels[x, y] = (h, s, v)
                    
                    # Green colors (hue 70-160° → ~50-113)
                    elif 50 <= h <= 113 and s > 50 and 50 <= v <= 220:
                        # Reduce saturation
                        s = int(s * 0.5)
                        pixels[x, y] = (h, s, v)
                    
                    # Grey colors (low saturation)
                    elif s < 50 and 30 <= v <= 220:
                        # Reduce saturation
                        s = int(s * 0.5)
                        pixels[x, y] = (h, s, v)
            
            # Convert back to RGB
            result_img = hsv_img.convert('RGB')
        
        # Save with original format via temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir=os.path.dirname(image_path)) as tmpfile:
            tmp_path = tmpfile.name
            if ext in ('.jpg', '.jpeg'):
                result_img.save(tmp_path, format='JPEG', quality=95)
            elif ext == '.png':
                result_img.save(tmp_path, format='PNG')
            elif ext == '.bmp':
                result_img.save(tmp_path, format='BMP')
            elif ext in ('.tif', '.tiff'):
                result_img.save(tmp_path, format='TIFF')
            else:
                result_img.save(tmp_path)
        
        # Replace original file
        shutil.move(tmp_path, image_path)
        return True
    
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

# Get directories from user
dirs = []
for i in range(3):
    dir_path = input(f"Enter directory path {i+1}: ").strip()
    if os.path.isdir(dir_path):
        dirs.append(dir_path)
    else:
        print(f"Invalid directory: {dir_path}")

# Process all images in directories
extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
processed_count = 0

for dir_path in dirs:
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(extensions):
                file_path = os.path.join(root, file)
                if process_image(file_path):
                    processed_count += 1

print(f"Processing complete! Modified {processed_count} images.")
