# minimal_test.py
"""Minimal test - no imports from project"""
import os

print("Starting minimal test...")

base_path = r"e:\QualityControl\dataset"
categories = ['Lattice', 'Solid', 'Stripped', 'Printed']
classes = ['Good', 'Bad']

print(f"Checking {base_path}...")

if os.path.exists(base_path):
    print("✓ Dataset directory exists")
    
    for category in categories:
        cat_path = os.path.join(base_path, category)
        if os.path.exists(cat_path):
            for cls in classes:
                cls_path = os.path.join(cat_path, cls)
                if os.path.exists(cls_path):
                    count = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    print(f"  {category}/{cls}: {count} images")
else:
    print("✗ Dataset directory not found")

print("\n✓ Basic file system test complete!")
