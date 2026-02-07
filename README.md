# Wool Quality Classification - Deep Learning Project

## Project Overview
This project implements a deep learning model to classify wool quality across different patterns (Lattice, Solid, Stripped, Printed) with binary classification (Good/Bad) for each pattern type.

**Dataset Structure:**
```
dataset/
├── Lattice/
│   ├── Good/
│   └── Bad/
├── Solid/
│   ├── Good/
│   └── Bad/
├── Stripped/
│   ├── Good/
│   └── Bad/
└── Printed/
    ├── Good/
    └── Bad/
```

**Total Images:** 1627

## Implementation Steps

### Step 1: Environment Setup
1. Install Python 3.8+ and VS Code
2. Install required extensions in VS Code:
   - Python
   - Jupyter
   - Pylance
3. Create virtual environment
4. Install dependencies

### Step 2: Project Structure Creation
```
wool-quality-classification/
├── data/
│   └── dataset/          # Your dataset folder
├── models/               # Saved models
├── notebooks/            # Jupyter notebooks for experiments
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── results/              # Training results, plots
├── requirements.txt
├── config.py
└── main.py
```

### Step 3: Data Preparation
- Load and organize images
- Apply data augmentation
- Split into train/validation/test sets
- Create data loaders

### Step 4: Model Selection & Training
We'll implement and compare:
1. **CNN from Scratch** - Baseline
2. **VGG16** - Transfer Learning
3. **ResNet50** - Transfer Learning
4. **EfficientNetB0** - Transfer Learning (Recommended for this dataset size)
5. **MobileNetV2** - Lightweight option

### Step 5: Evaluation & Testing
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Classification Report

### Step 6: Deployment Preparation
- Save best model
- Create prediction pipeline

## Quick Start

```bash
# 1. Clone/Create project directory
mkdir wool-quality-classification
cd wool-quality-classification

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run training with best model (EfficientNet)
python main.py --model efficientnet --epochs 50

# 5. Evaluate model
python src/evaluate.py --model_path models/best_model.h5

# 6. Make predictions
python src/predict.py --image_path path/to/image.jpg
```

## Model Selection Recommendation

For 1627 images, **EfficientNetB0** is recommended because:
- Excellent accuracy with limited data
- Efficient architecture (fewer parameters)
- Better generalization
- Fast training and inference

Expected Performance: 90-95% accuracy with proper data augmentation
