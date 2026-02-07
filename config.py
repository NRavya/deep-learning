# config.py
"""
Configuration file for Wool Quality Classification Project
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset Configuration
CATEGORIES = ['Lattice', 'Solid', 'Stripped', 'Printed']
CLASSES = ['Good', 'Bad']
NUM_CLASSES = 2

# Image Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Training Configuration
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Data Augmentation
USE_AUGMENTATION = True
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'vertical_flip': False,
    'fill_mode': 'nearest'
}

# Model Configuration
AVAILABLE_MODELS = [
    'cnn_scratch',
    'vgg16',
    'resnet50',
    'efficientnet',
    'mobilenet'
]

# Default model (recommended)
DEFAULT_MODEL = 'efficientnet'

# Callbacks Configuration
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# Random Seed for Reproducibility
RANDOM_SEED = 42
