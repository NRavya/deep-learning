# safe_test.py
"""Safe test with error handling"""

try:
    import sys
    print(f"Python: {sys.version}")
    print("Starting test...")
    
    import os
    print("os imported")
    
    import config
    print(f"config imported, DATA_DIR={config.DATA_DIR}")
    
    # Test file counting
    path = os.path.join(config.DATA_DIR, config.CATEGORIES[0], config.CLASSES[0])
    print(f"Checking path: {path}")
    
    if os.path.exists(path):
        files = os.listdir(path)
        print(f"Found {len(files)} items")
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(image_files)} images")
    else:
        print(f"Path does not exist: {path}")
    
    print("Test complete!")

except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    print(traceback.format_exc())
