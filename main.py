# main.py
"""
Main script to train and evaluate wool quality classification models
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
import numpy as np
import tensorflow as tf
import config

np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)

from src.data_loader import WoolDataLoader
from src.train import ModelTrainer


def compare_all_models(X_train, y_train, X_val, y_val, X_test, y_test, 
                       class_weights, epochs=30):
    """
    Train and compare all available models
    
    Args:
        X_train, y_train: training data
        X_val, y_val: validation data
        X_test, y_test: test data
        class_weights: class weights for imbalanced data
        epochs: number of epochs to train
    """
    from src.evaluate import ModelEvaluator
    
    results = {}
    
    print("\n" + "="*70)
    print("COMPARING ALL MODEL ARCHITECTURES")
    print("="*70)
    
    for model_name in config.AVAILABLE_MODELS:
        print(f"\n{'='*70}")
        print(f"Training: {model_name.upper()}")
        print('='*70)
        
        try:
            # Train model
            trainer = ModelTrainer(model_name=model_name, epochs=epochs)
            history, model_dir = trainer.train(
                X_train, y_train,
                X_val, y_val,
                class_weights=class_weights,
                fine_tune=False
            )
            
            # Plot training history
            plot_path = os.path.join(model_dir, 'training_history.png')
            trainer.plot_training_history(save_path=plot_path)
            
            # Evaluate on test set
            best_model_path = os.path.join(model_dir, 'best_model.keras')
            evaluator = ModelEvaluator(best_model_path)
            metrics, y_pred, y_pred_proba = evaluator.evaluate(X_test, y_test)
            
            # Save results
            results[model_name] = {
                'metrics': metrics,
                'model_path': best_model_path,
                'model_dir': model_dir
            }
            
            print(f"\n✓ {model_name} completed successfully!")
            
        except Exception as e:
            print(f"\n✗ Error training {model_name}: {str(e)}")
            results[model_name] = {'error': str(e)}
    
    # Print comparison summary
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*70)
    
    for model_name, result in results.items():
        if 'error' in result:
            print(f"{model_name:<20} Error: {result['error']}")
        else:
            metrics = result['metrics']
            print(f"{model_name:<20} "
                  f"{metrics['accuracy']:<12.4f} "
                  f"{metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} "
                  f"{metrics['f1_score']:<12.4f}")
    
    # Find best model
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        best_model = max(valid_results.items(), 
                        key=lambda x: x[1]['metrics']['accuracy'])
        
        print("\n" + "="*70)
        print(f"BEST MODEL: {best_model[0].upper()}")
        print(f"Accuracy: {best_model[1]['metrics']['accuracy']:.4f}")
        print(f"Model path: {best_model[1]['model_path']}")
        print("="*70)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Wool Quality Classification - Deep Learning Project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single model (recommended: EfficientNet)
  python main.py --model efficientnet --epochs 50
  
  # Train with fine-tuning
  python main.py --model efficientnet --epochs 50 --fine-tune
  
  # Compare all models (takes longer)
  python main.py --compare-all --epochs 30
  
  # Quick test run
  python main.py --model efficientnet --epochs 5
        """
    )
    
    parser.add_argument('--model', type=str, default=config.DEFAULT_MODEL,
                       choices=config.AVAILABLE_MODELS,
                       help='Model architecture to train')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=config.LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--fine-tune', action='store_true',
                       help='Enable fine-tuning after initial training')
    parser.add_argument('--compare-all', action='store_true',
                       help='Train and compare all available models')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    config.USE_AUGMENTATION = not args.no_augmentation
    
    print("="*70)
    print("WOOL QUALITY CLASSIFICATION - DEEP LEARNING PROJECT")
    print("="*70)
    print(f"Configuration:")
    print(f"  Image Size: {config.IMG_SIZE}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    print(f"  Data Augmentation: {config.USE_AUGMENTATION}")
    print(f"  Random Seed: {config.RANDOM_SEED}")
    print("="*70)
    
    # Check if dataset exists
    if not os.path.exists(config.DATA_DIR):
        print(f"\nError: Dataset directory not found at {config.DATA_DIR}")
        print("Please create the dataset directory with the following structure:")
        print("""
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
        """)
        sys.exit(1)
    
    # Load dataset
    print("\nStep 1: Loading Dataset")
    print("-"*70)
    loader = WoolDataLoader()
    X, y, df = loader.load_dataset()
    
    # Split data
    print("\nStep 2: Splitting Data")
    print("-"*70)
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
    
    # Calculate class weights
    class_weights = loader.get_class_weights(y_train)
    
    # Train models
    print("\nStep 3: Training Model(s)")
    print("-"*70)
    
    if args.compare_all:
        # Compare all models
        results = compare_all_models(
            X_train, y_train, X_val, y_val, X_test, y_test,
            class_weights, epochs=args.epochs
        )
    else:
        # Train single model
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
        
        # Evaluate on test set
        print("\nStep 4: Evaluating on Test Set")
        print("-"*70)
        from src.evaluate import ModelEvaluator
        
        best_model_path = os.path.join(model_dir, 'best_model.keras')
        evaluator = ModelEvaluator(best_model_path)
        metrics, y_pred, y_pred_proba = evaluator.evaluate(X_test, y_test)
        
        # Generate evaluation plots
        results_dir = os.path.join(model_dir, 'evaluation_results')
        os.makedirs(results_dir, exist_ok=True)
        
        evaluator.plot_confusion_matrix(
            y_test, y_pred, 
            save_path=os.path.join(results_dir, 'confusion_matrix.png')
        )
        
        evaluator.plot_roc_curve(
            y_test, y_pred_proba,
            save_path=os.path.join(results_dir, 'roc_curve.png')
        )
        
        evaluator.visualize_predictions(
            X_test, y_test, y_pred,
            save_path=os.path.join(results_dir, 'sample_predictions.png')
        )
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Model saved at: {best_model_path}")
        print(f"Results saved at: {model_dir}")
        print("="*70)
        
        print("\nNext steps:")
        print(f"1. Review training results in: {model_dir}")
        print(f"2. Make predictions using:")
        print(f"   python src/predict.py --model_path {best_model_path} --image_path <path_to_image>")
        print(f"3. For batch predictions:")
        print(f"   python src/predict.py --model_path {best_model_path} --directory_path <path_to_directory>")


if __name__ == "__main__":
    main()
