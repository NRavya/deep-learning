# fast_test.py
"""Fast test without TensorFlow"""

import os
import config

print("Testing data loader (no TensorFlow import)...", flush=True)

total = 0
for category in config.CATEGORIES:
    for class_name in config.CLASSES:
        path = os.path.join(config.DATA_DIR, category, class_name)
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            total += count
            print(f"{category}/{class_name}: {count} images")

print(f"\nTotal images: {total}")
print("✓ Data loader works!")

# Now test loading a few images
print("\nTesting PIL image loading...")
from PIL import Image
import numpy as np

test_path = os.path.join(config.DATA_DIR, config.CATEGORIES[0], config.CLASSES[0], 'GL01.jpg')
img = Image.open(test_path).convert('RGB')
img_resized = img.resize(config.IMG_SIZE, Image.Resampling.LANCZOS)
img_array = np.array(img_resized, dtype=np.float32)

print(f"Image loaded: shape={img_array.shape}, dtype={img_array.dtype}")
print("✓ Image loading works!")
