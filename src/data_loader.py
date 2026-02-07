# src/data_loader.py
"""
Data loading and preprocessing utilities - Fast, no TensorFlow dependency
"""

import os
import sys

# Add parent directory to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import config
from tqdm import tqdm
from PIL import Image


class WoolDataLoader:
    """Handle data loading and preprocessing for wool quality classification"""
    
    def __init__(self, data_dir=config.DATA_DIR):
        self.data_dir = data_dir
        self.categories = config.CATEGORIES
        self.classes = config.CLASSES
        self.img_size = config.IMG_SIZE
        
    def load_dataset(self):
        """
        Load all images from the dataset directory with caching
        
        Returns:
            X: numpy array of images
            y: numpy array of labels
            df: pandas DataFrame with image metadata
        """
        cache_file = os.path.join(self.data_dir, '..', 'dataset_cache.npz')
        
        # Check if cache exists
        if os.path.exists(cache_file):
            print("Loading from cache...")
            data = np.load(cache_file, allow_pickle=True)
            X = data['X']
            y = data['y']
            df = pd.DataFrame(data['metadata'].item())
            print(f"✓ Cache loaded! {len(X)} images")
            return X, y, df
        
        images = []
        labels = []
        metadata = []
        
        print("Loading dataset (first run - will cache for next time)...")
        
        for category in self.categories:
            for class_name in self.classes:
                path = os.path.join(self.data_dir, category, class_name)
                
                if not os.path.exists(path):
                    print(f"Warning: Path not found - {path}")
                    continue
                
                class_label = 1 if class_name == 'Good' else 0
                image_files = [f for f in os.listdir(path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                print(f"Loading {len(image_files)} images from {category}/{class_name}")
                
                for img_file in tqdm(image_files, desc=f"{category}/{class_name}"):
                    img_path = os.path.join(path, img_file)
                    
                    try:
                        # Load image using PIL (much faster than Keras)
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize(self.img_size, Image.Resampling.LANCZOS)
                        img_array = np.array(img, dtype=np.float32)
                        # Don't normalize here - let the generator handle it
                        
                        images.append(img_array)
                        labels.append(class_label)
                        
                        metadata.append({
                            'filename': img_file,
                            'category': category,
                            'class': class_name,
                            'label': class_label,
                            'path': img_path
                        })
                        
                    except Exception as e:
                        print(f"Error loading {img_path}: {str(e)}")
                        continue
        
        X = np.array(images)
        y = np.array(labels)
        df = pd.DataFrame(metadata)
        
        # Save to cache for next time
        print("\nSaving cache...")
        np.savez_compressed(cache_file, X=X, y=y, metadata={'df': df.to_dict()})
        
        print(f"\nDataset loaded successfully!")
        print(f"Total images: {len(X)}")
        print(f"Image shape: {X[0].shape}")
        print(f"Label distribution:\n{df['class'].value_counts()}")
        
        return X, y, df
    
    def split_data(self, X, y, test_size=config.TEST_SPLIT, 
                   val_size=config.VALIDATION_SPLIT, random_state=config.RANDOM_SEED):
        """
        Split data into train, validation, and test sets
        
        Args:
            X: images array
            y: labels array
            test_size: proportion of test set
            val_size: proportion of validation set from remaining data
            random_state: random seed
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
        
        print(f"\nData split:")
        print(f"Train set: {len(X_train)} images")
        print(f"Validation set: {len(X_val)} images")
        print(f"Test set: {len(X_test)} images")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_class_weights(self, y):
        """
        Calculate class weights for imbalanced datasets
        
        Returns:
            class_weights: dictionary of class weights
        """
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights = dict(zip(classes, weights))
        
        print(f"\nClass weights: {class_weights}")
        
        return class_weights
    
    def normalize_images(self, X):
        """Normalize images to [0, 1] range"""
        return X / 255.0


def main():
    """Test data loader functionality"""
    loader = WoolDataLoader()
    X, y, df = loader.load_dataset()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
    
    # Show class distribution
    print("\nClass distribution:")
    print(f"Train - Good: {sum(y_train)}, Bad: {len(y_train) - sum(y_train)}")
    print(f"Val - Good: {sum(y_val)}, Bad: {len(y_val) - sum(y_val)}")
    print(f"Test - Good: {sum(y_test)}, Bad: {len(y_test) - sum(y_test)}")
    
    # Calculate class weights
    class_weights = loader.get_class_weights(y_train)
    
    print("\n✓ Data loading pipeline complete!")


if __name__ == "__main__":
    main()
