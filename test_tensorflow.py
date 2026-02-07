# test_tensorflow.py
"""Test TensorFlow import directly"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only

print("Attempting TensorFlow import...")
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print("✓ TensorFlow import successful!")
