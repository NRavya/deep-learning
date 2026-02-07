# src/train.py
"""
Training script for wool quality classification
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow import keras
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, 
    TensorBoard, CSVLogger
)
import config
from src.data_loader import WoolDataLoader
from src.model import WoolClassifier


class ModelTrainer:
    """Handle model training with callbacks and monitoring"""
    
    def __init__(self, model_name='efficientnet', epochs=config.EPOCHS):
        self.model_name = model_name
        self.epochs = epochs
        self.history = None
        self.model = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def create_callbacks(self):
        """Create training callbacks"""
        
        # Create model-specific directory
        model_dir = os.path.join(config.MODELS_DIR, f"{self.model_name}_{self.timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        callbacks = [
            # Save best model
            ModelCheckpoint(
                filepath=os.path.join(model_dir, 'best_model.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=config.REDUCE_LR_FACTOR,
                patience=config.REDUCE_LR_PATIENCE,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir=os.path.join(config.RESULTS_DIR, 'logs', f"{self.model_name}_{self.timestamp}"),
                histogram_freq=1
            ),
            
            # CSV logging
            CSVLogger(
                filename=os.path.join(model_dir, 'training_log.csv'),
                separator=',',
                append=False
            )
        ]
        
        return callbacks, model_dir
    
    def train(self, X_train, y_train, X_val, y_val, class_weights=None, fine_tune=False):
        """
        Train the model
        
        Args:
            X_train, y_train: training data
            X_val, y_val: validation data
            class_weights: dictionary of class weights for imbalanced data
            fine_tune: whether to unfreeze and fine-tune the model
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_name} model")
        print('='*60)
        
        # Build and compile model
        classifier = WoolClassifier(model_name=self.model_name)
        self.model = classifier.build()
        classifier.compile()
        
        print("\nModel Architecture:")
        classifier.summary()
        
        # Create callbacks
        callbacks, model_dir = self.create_callbacks()
        
        # Initial training
        print("\n" + "="*60)
        print("Starting Initial Training...")
        print("="*60)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Fine-tuning (optional)
        if fine_tune and self.model_name != 'cnn_scratch':
            print("\n" + "="*60)
            print("Starting Fine-Tuning Phase...")
            print("="*60)
            
            # Unfreeze last 20 layers
            classifier.unfreeze_layers(num_layers=20)
            
            # Continue training with lower learning rate
            fine_tune_epochs = self.epochs // 2
            
            history_fine = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=fine_tune_epochs,
                batch_size=config.BATCH_SIZE,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1,
                initial_epoch=len(self.history.history['loss'])
            )
            
            # Combine histories
            for key in self.history.history.keys():
                self.history.history[key].extend(history_fine.history[key])
        
        # Save final model
        final_model_path = os.path.join(model_dir, 'final_model.keras')
        self.model.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
        
        return self.history, model_dir
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        # Check which metrics are available
        available_metrics = list(self.history.history.keys())
        has_precision = 'precision' in available_metrics
        has_recall = 'recall' in available_metrics
        
        if has_precision and has_recall:
            # Plot all 4 metrics
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Accuracy
            axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy')
            axes[0, 0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
            axes[0, 0].set_title('Model Accuracy')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Loss
            axes[0, 1].plot(self.history.history['loss'], label='Train Loss')
            axes[0, 1].plot(self.history.history['val_loss'], label='Val Loss')
            axes[0, 1].set_title('Model Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Precision
            axes[1, 0].plot(self.history.history['precision'], label='Train Precision')
            axes[1, 0].plot(self.history.history['val_precision'], label='Val Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Recall
            axes[1, 1].plot(self.history.history['recall'], label='Train Recall')
            axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        else:
            # Plot only accuracy and loss (2 plots instead of 4)
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Accuracy
            axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
            axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
            axes[0].set_title('Model Accuracy')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Accuracy')
            axes[0].legend()
            axes[0].grid(True)
            
            # Loss
            axes[1].plot(self.history.history['loss'], label='Train Loss')
            axes[1].plot(self.history.history['val_loss'], label='Val Loss')
            axes[1].set_title('Model Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        plt.show()


def main():
    """Main training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Wool Quality Classification Model')
    parser.add_argument('--model', type=str, default=config.DEFAULT_MODEL,
                       choices=config.AVAILABLE_MODELS,
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--fine-tune', action='store_true',
                       help='Enable fine-tuning after initial training')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading dataset...")
    loader = WoolDataLoader()
    X, y, df = loader.load_dataset()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
    
    # Calculate class weights
    class_weights = loader.get_class_weights(y_train)
    
    # Train model
    trainer = ModelTrainer(model_name=args.model, epochs=args.epochs)
    history, model_dir = trainer.train(
        X_train, y_train, 
        X_val, y_val,
        class_weights=class_weights,
        fine_tune=args.fine_tune
    )
    
    # Plot training history
    plot_path = os.path.join(model_dir, 'training_history.png')
    trainer.plot_training_history(save_path=plot_path)
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Models saved in: {model_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
