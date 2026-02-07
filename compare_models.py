# compare_models.py
"""
Script to compare all available models and generate comprehensive report
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import config
from src.data_loader import WoolDataLoader
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator


def compare_all_models(epochs=30, save_results=True):
    """
    Train and compare all available models
    
    Args:
        epochs: number of epochs to train each model
        save_results: whether to save comparison results
    """
    # Load data
    print("Loading dataset...")
    loader = WoolDataLoader()
    X, y, df = loader.load_dataset()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
    class_weights = loader.get_class_weights(y_train)
    
    # Results storage
    results = {
        'model_name': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'training_time': [],
        'parameters': [],
        'model_path': []
    }
    
    # Train each model
    for model_name in config.AVAILABLE_MODELS:
        print(f"\n{'='*70}")
        print(f"Training: {model_name.upper()}")
        print('='*70)
        
        try:
            # Track training time
            start_time = datetime.now()
            
            # Train model
            trainer = ModelTrainer(model_name=model_name, epochs=epochs)
            history, model_dir = trainer.train(
                X_train, y_train,
                X_val, y_val,
                class_weights=class_weights,
                fine_tune=False
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Evaluate model
            best_model_path = os.path.join(model_dir, 'best_model.keras')
            evaluator = ModelEvaluator(best_model_path)
            metrics, y_pred, y_pred_proba = evaluator.evaluate(X_test, y_test)
            
            # Store results
            results['model_name'].append(model_name)
            results['accuracy'].append(metrics['accuracy'])
            results['precision'].append(metrics['precision'])
            results['recall'].append(metrics['recall'])
            results['f1_score'].append(metrics['f1_score'])
            results['training_time'].append(training_time)
            results['parameters'].append(trainer.model.count_params())
            results['model_path'].append(best_model_path)
            
            print(f"\n✓ {model_name} completed successfully!")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Training time: {training_time:.1f}s")
            
        except Exception as e:
            print(f"\n✗ Error training {model_name}: {str(e)}")
            continue
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    
    # Print comparison table
    print("\n" + "="*100)
    print("MODEL COMPARISON SUMMARY")
    print("="*100)
    print(df_results.to_string(index=False))
    print("="*100)
    
    # Find best models
    best_accuracy = df_results.loc[df_results['accuracy'].idxmax()]
    best_f1 = df_results.loc[df_results['f1_score'].idxmax()]
    fastest = df_results.loc[df_results['training_time'].idxmin()]
    smallest = df_results.loc[df_results['parameters'].idxmin()]
    
    print("\n" + "="*70)
    print("BEST MODELS BY CATEGORY")
    print("="*70)
    print(f"Best Accuracy: {best_accuracy['model_name']} ({best_accuracy['accuracy']:.4f})")
    print(f"Best F1-Score: {best_f1['model_name']} ({best_f1['f1_score']:.4f})")
    print(f"Fastest Training: {fastest['model_name']} ({fastest['training_time']:.1f}s)")
    print(f"Smallest Model: {smallest['model_name']} ({smallest['parameters']:,} params)")
    print("="*70)
    
    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(config.RESULTS_DIR, f'comparison_{timestamp}')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save CSV
        csv_path = os.path.join(results_dir, 'model_comparison.csv')
        df_results.to_csv(csv_path, index=False)
        
        # Save JSON
        json_path = os.path.join(results_dir, 'model_comparison.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate visualizations
        generate_comparison_plots(df_results, results_dir)
        
        print(f"\nResults saved to: {results_dir}")
    
    return df_results


def generate_comparison_plots(df_results, save_dir):
    """Generate comparison visualizations"""
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Metrics Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        df_sorted = df_results.sort_values(metric, ascending=False)
        bars = ax.barh(df_sorted['model_name'], df_sorted[metric])
        
        # Color bars
        colors = plt.cm.RdYlGn(df_sorted[metric])
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xlabel(title, fontsize=12)
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(df_sorted[metric]):
            ax.text(v + 0.01, i, f'{v:.4f}', va='center')
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance vs Parameters
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(df_results['parameters'], 
                        df_results['accuracy'],
                        s=df_results['training_time']*10,
                        alpha=0.6,
                        c=df_results['f1_score'],
                        cmap='RdYlGn')
    
    for idx, row in df_results.iterrows():
        ax.annotate(row['model_name'], 
                   (row['parameters'], row['accuracy']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Efficiency: Accuracy vs Parameters\n(Bubble size = Training time, Color = F1-Score)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('F1-Score', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'efficiency_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Training Time Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df_sorted = df_results.sort_values('training_time')
    bars = ax.bar(df_sorted['model_name'], df_sorted['training_time'])
    
    for bar, time in zip(bars, df_sorted['training_time']):
        color = 'green' if time < df_sorted['training_time'].median() else 'orange'
        bar.set_color(color)
    
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Model Training Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for i, (name, time) in enumerate(zip(df_sorted['model_name'], df_sorted['training_time'])):
        ax.text(i, time + max(df_sorted['training_time'])*0.02, 
               f'{time:.1f}s', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_time_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Overall Score (weighted)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate weighted score
    df_results['overall_score'] = (
        df_results['accuracy'] * 0.4 +
        df_results['f1_score'] * 0.4 +
        (1 - df_results['training_time'] / df_results['training_time'].max()) * 0.2
    )
    
    df_sorted = df_results.sort_values('overall_score', ascending=False)
    bars = ax.barh(df_sorted['model_name'], df_sorted['overall_score'])
    
    colors = plt.cm.RdYlGn(df_sorted['overall_score'])
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Overall Score', fontsize=12)
    ax.set_title('Overall Model Ranking\n(40% Accuracy + 40% F1-Score + 20% Speed)',
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(df_sorted['overall_score']):
        ax.text(v + 0.01, i, f'{v:.4f}', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overall_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to: {save_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare all model architectures')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs to train each model')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results')
    
    args = parser.parse_args()
    
    results = compare_all_models(
        epochs=args.epochs,
        save_results=not args.no_save
    )
