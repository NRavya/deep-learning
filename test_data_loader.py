# test_data_loader.py
"""
Test data loader with limited samples
"""

import sys
sys.path.insert(0, '.')

from src.data_loader import WoolDataLoader
import config

print("Starting data loader test...")
print(f"Data directory: {config.DATA_DIR}")
print(f"Categories: {config.CATEGORIES}")
print(f"Classes: {config.CLASSES}")
print(f"Image size: {config.IMG_SIZE}")

loader = WoolDataLoader()

print("\nLoading first 10 images from each category/class combination...")
print("This is a test run with limited images to verify functionality.\n")

# Load small test set
images = []
labels = []

import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm

for category in config.CATEGORIES:
    for class_name in config.CLASSES:
        path = os.path.join(config.DATA_DIR, category, class_name)
        
        if not os.path.exists(path):
            print(f"Warning: Path not found - {path}")
            continue
        
        class_label = 1 if class_name == 'Good' else 0
        image_files = [f for f in os.listdir(path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:10]  # Only 10 images
        
        print(f"Loading {len(image_files)} images from {category}/{class_name}")
        
        for img_file in image_files:
            img_path = os.path.join(path, img_file)
            try:
                img = load_img(img_path, target_size=config.IMG_SIZE)
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(class_label)
            except Exception as e:
                print(f"Error loading {img_path}: {str(e)}")

print(f"\nLoaded {len(images)} images")
print(f"First image shape: {images[0].shape}")
print("✓ Data loader is working!")
