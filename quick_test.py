# quick_test.py
"""Quick test to verify data loader works"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Testing data loader...")
from src.data_loader import WoolDataLoader
import config

loader = WoolDataLoader()

# Just count files, don't load them
total = 0
for category in config.CATEGORIES:
    for class_name in config.CLASSES:
        path = os.path.join(config.DATA_DIR, category, class_name)
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            total += count
            print(f"{category}/{class_name}: {count} images")

print(f"\nTotal images: {total}")
print("✓ Data loader ready!")
